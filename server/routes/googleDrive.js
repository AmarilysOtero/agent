const express = require('express');
const router = express.Router();
const { google } = require('googleapis');
const { v4: uuidv4 } = require('uuid');

const authState = new Map();

// Generate OAuth URL
router.post('/authenticate', (req, res) => {
  try {
    const clientId = process.env.GOOGLE_CLIENT_ID || req.body.clientId;
    const clientSecret = process.env.GOOGLE_CLIENT_SECRET || req.body.clientSecret;

    if (!clientId || !clientSecret) {
      return res.status(400).json({ 
        error: 'Google Drive API credentials not configured. Please provide clientId and clientSecret.' 
      });
    }

    const oauth2Client = new google.auth.OAuth2(
      clientId,
      clientSecret,
      `${req.protocol}://${req.get('host')}/api/drive/callback`
    );

    const authUrl = oauth2Client.generateAuthUrl({
      access_type: 'offline',
      scope: [
        'https://www.googleapis.com/auth/drive.readonly'
      ],
      prompt: 'consent'
    });

    const stateId = uuidv4();
    authState.set(stateId, { oauth2Client, timestamp: Date.now() });

    res.json({ authUrl, stateId });
  } catch (error) {
    console.error('Google Drive auth error:', error);
    res.status(500).json({ error: error.message });
  }
});

// OAuth callback
router.get('/callback', async (req, res) => {
  try {
    const { code, state } = req.query;
    
    if (!authState.has(state)) {
      return res.send(`
        <html>
          <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h2>❌ Authentication Failed</h2>
            <p>Session expired. Please try again.</p>
          </body>
        </html>
      `);
    }

    const { oauth2Client } = authState.get(state);
    const { tokens } = await oauth2Client.getToken(code);
    oauth2Client.setCredentials(tokens);
    
    // Update auth state with tokens
    authState.set(state, { oauth2Client, tokens, timestamp: Date.now() });

    res.send(`
      <html>
        <head>
          <style>
            body {
              font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
              padding: 40px;
              text-align: center;
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              color: white;
            }
            .container {
              background: rgba(255, 255, 255, 0.1);
              backdrop-filter: blur(10px);
              padding: 40px;
              border-radius: 20px;
              max-width: 400px;
              margin: 0 auto;
            }
            .success-icon {
              font-size: 64px;
              margin-bottom: 20px;
            }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="success-icon">✅</div>
            <h2>Authentication Successful!</h2>
            <p>You can close this window and return to the application.</p>
            <script>
              setTimeout(() => {
                window.close();
              }, 2000);
            </script>
          </div>
        </body>
      </html>
    `);
  } catch (error) {
    res.send(`
      <html>
        <body style="font-family: Arial; padding: 40px;">
          <h2>❌ Error</h2>
          <p>${error.message}</p>
        </body>
      </html>
    `);
  }
});

// Scan Google Drive
router.post('/scan', async (req, res) => {
  try {
    const { stateId } = req.body;
    
    if (!authState.has(stateId)) {
      return res.status(401).json({ error: 'Not authenticated. Please authenticate first.' });
    }

    const { oauth2Client } = authState.get(stateId);
    const drive = google.drive({ version: 'v3', auth: oauth2Client });
    
    console.log('📁 Scanning Google Drive...');
    
    const fileStructure = await scanGoogleDrive(drive, 'root', '');
    
    res.json({
      success: true,
      data: fileStructure,
      source: 'google-drive',
      totalFiles: countFiles(fileStructure),
      totalFolders: countFolders(fileStructure)
    });
  } catch (error) {
    console.error('Error scanning Google Drive:', error);
    res.status(500).json({ error: error.message });
  }
});

async function scanGoogleDrive(drive, folderId, parentPath = '', pageToken = null) {
  const q = folderId === 'root' 
    ? 'parents in root and trashed = false'
    : `parents in '${folderId}' and trashed = false`;
  
  const params = {
    q,
    fields: 'nextPageToken, files(id, name, mimeType, size, modifiedTime, webViewLink)',
    pageSize: 1000
  };

  if (pageToken) {
    params.pageToken = pageToken;
  }

  const response = await drive.files.list(params);
  const children = [];

  for (const file of response.data.files || []) {
    const currentPath = parentPath ? `${parentPath}/${file.name}` : file.name;
    
    if (file.mimeType === 'application/vnd.google-apps.folder') {
      const folderData = await scanGoogleDrive(drive, file.id, currentPath);
      children.push(folderData);
    } else {
      children.push({
        id: file.id || uuidv4(),
        type: 'file',
        name: file.name,
        path: currentPath,
        relativePath: currentPath,
        size: parseInt(file.size || '0'),
        mimeType: file.mimeType,
        extension: extractExtension(file.name),
        modifiedTime: file.modifiedTime,
        webViewLink: file.webViewLink,
        googleDriveId: file.id,
        children: [],
        source: 'google-drive'
      });
    }
  }

  // Handle pagination
  if (response.data.nextPageToken) {
    const moreChildren = await scanGoogleDrive(drive, folderId, parentPath, response.data.nextPageToken);
    children.push(...moreChildren.children);
  }

  const folderName = folderId === 'root' ? 'Google Drive Root' : 
    (await drive.files.get({ fileId: folderId, fields: 'name' })).data.name;

  return {
    id: folderId,
    type: 'directory',
    name: folderName,
    path: parentPath || 'Google Drive',
    relativePath: parentPath || 'Google Drive',
    googleDriveId: folderId,
    modifiedTime: new Date().toISOString(),
    children: children,
    source: 'google-drive'
  };
}

function extractExtension(filename) {
  const lastDot = filename.lastIndexOf('.');
  return lastDot > 0 ? filename.substring(lastDot) : 'no extension';
}

function countFiles(node) {
  if (node.type === 'file') return 1;
  return node.children.reduce((sum, child) => sum + countFiles(child), 0);
}

function countFolders(node) {
  if (node.type === 'file') return 0;
  return node.children.reduce((sum, child) => sum + countFolders(child) + 1, 0);
}

module.exports = router;
