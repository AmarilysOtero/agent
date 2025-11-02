const express = require('express');
const router = express.Router();
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

const authState = new Map();

// Generate authentication URL
router.post('/authenticate', (req, res) => {
  try {
    const { siteUrl, clientId, tenantId } = req.body;

    if (!siteUrl || !clientId || !tenantId) {
      return res.status(400).json({ 
        error: 'Missing required parameters: siteUrl, clientId, tenantId' 
      });
    }

    const stateId = uuidv4();
    authState.set(stateId, {
      siteUrl,
      clientId,
      tenantId,
      timestamp: Date.now()
    });

    const authUrl = `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/authorize?` +
      `client_id=${clientId}&` +
      `response_type=code&` +
      `redirect_uri=${req.protocol}://${req.get('host')}/api/sharepoint/callback&` +
      `response_mode=query&` +
      `scope=https://graph.microsoft.com/Files.Read.All offline_access&` +
      `state=${stateId}`;

    res.json({ authUrl, stateId });
  } catch (error) {
    console.error('SharePoint auth error:', error);
    res.status(500).json({ error: error.message });
  }
});

// OAuth callback
router.get('/callback', (req, res) => {
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

  const credentials = authState.get(state);
  authState.set(state, { ...credentials, authorizationCode: code });

  res.send(`
    <html>
      <head>
        <style>
          body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, #0078d4 0%, #0051a2 100%);
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
});

// Scan SharePoint
router.post('/scan', async (req, res) => {
  try {
    const { stateId } = req.body;
    
    if (!authState.has(stateId)) {
      return res.status(401).json({ error: 'Not authenticated. Please authenticate first.' });
    }

    const credentials = authState.get(stateId);
    
    // Note: This requires implementing Microsoft Graph token exchange
    // For now, return a placeholder response
    res.json({
      success: true,
      message: 'SharePoint scanning requires Microsoft Graph API token exchange',
      note: 'Please implement token exchange using client_id and authorization_code',
      credentials: {
        siteUrl: credentials.siteUrl,
        tenantId: credentials.tenantId,
        hasCode: !!credentials.authorizationCode
      }
    });
  } catch (error) {
    console.error('Error scanning SharePoint:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
