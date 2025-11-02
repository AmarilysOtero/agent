const express = require('express');
const router = express.Router();
const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');

// Scan local directory
router.post('/scan', async (req, res) => {
  try {
    const { directoryPath } = req.body;
    
    if (!directoryPath) {
      return res.status(400).json({ error: 'Directory path is required' });
    }

    console.log(`📁 Scanning directory: ${directoryPath}`);
    
    const fileStructure = await scanDirectory(directoryPath, directoryPath);
    
    console.log(`✅ Scan completed! Sending response with ${countFiles(fileStructure)} files and ${countFolders(fileStructure)} folders`);
    
    const totalFiles = countFiles(fileStructure);
    const totalFolders = countFolders(fileStructure);
    
    res.json({
      success: true,
      data: fileStructure,
      source: 'local',
      totalFiles: totalFiles,
      totalFolders: totalFolders
    });
  } catch (error) {
    console.error('Error scanning local directory:', error);
    res.status(500).json({ error: error.message });
  }
});

async function scanDirectory(dirPath, rootPath) {
  console.log(`🔍 Scanning: ${dirPath}`);
  let stats;
  try {
    stats = await fs.stat(dirPath);
    console.log(`✅ Stats retrieved for: ${dirPath}`);
  } catch (error) {
    console.error(`❌ Error accessing ${dirPath}:`, error.message);
    return null; // Skip this file/directory
  }
  
  if (stats.isFile()) {
    const relativePath = path.relative(rootPath, dirPath);
    return {
      id: uuidv4(),
      type: 'file',
      name: path.basename(dirPath),
      fullPath: dirPath,
      relativePath: relativePath,
      size: stats.size,
      extension: path.extname(dirPath) || 'no extension',
      modifiedTime: stats.mtime.toISOString(),
      createdAt: stats.birthtime.toISOString(),
      children: [],
      source: 'local'
    };
  }

  const entries = await fs.readdir(dirPath);
  console.log(`📂 Found ${entries.length} entries in: ${dirPath}`);
  const children = [];
  
  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry);
    console.log(`🔄 Processing entry: ${entry}`);
    try {
      const child = await scanDirectory(fullPath, rootPath);
      if (child) { // Only add if scanDirectory didn't return null
        children.push(child);
        console.log(`✅ Added child: ${entry}`);
      }
    } catch (error) {
      console.error(`Error scanning ${fullPath}:`, error.message);
    }
  }

  const relativePath = path.relative(rootPath, dirPath);
  return {
    id: uuidv4(),
    type: 'directory',
    name: path.basename(dirPath),
    fullPath: dirPath,
    relativePath: relativePath || '.',
    modifiedTime: stats.mtime.toISOString(),
    createdAt: stats.birthtime.toISOString(),
    children: children,
    source: 'local'
  };
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
