import { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Download, Copy, CheckCircle, FileJson, FolderOpen, Database, Upload } from 'lucide-react';
import { clearResults } from '../store/slices/scannerSlice';
import { Link } from 'react-router-dom';
import { formatBytes } from '../utils/formatters';
import { storeInNeo4j, getNeo4jStats, checkNeo4jHealth, getDirectoryFromNeo4j, uploadDirectoryToNeo4j } from '../services/api';

const Results = () => {
  const { scanResults } = useSelector((state) => state.scanner);
  const [copied, setCopied] = useState(false);
  const [neo4jStatus, setNeo4jStatus] = useState(null);
  const [isStoring, setIsStoring] = useState(false);
  const [storeMessage, setStoreMessage] = useState('');
  const [fileSelections, setFileSelections] = useState({});
  const [fileAlphanumericValues, setFileAlphanumericValues] = useState({});
  const [neo4jDirectoryStructure, setNeo4jDirectoryStructure] = useState(null);
  const [isLoadingNeo4jStructure, setIsLoadingNeo4jStructure] = useState(false);
  const [neo4jFileSelections, setNeo4jFileSelections] = useState({});
  const [neo4jFileAlphanumericValues, setNeo4jFileAlphanumericValues] = useState({});
  const [uploadStatus, setUploadStatus] = useState({}); // key: directory id/fullPath -> message
  const dispatch = useDispatch();

  // Fetch directory structure from Neo4j when scanResults are available
  useEffect(() => {
    const fetchNeo4jStructure = async () => {
      if (!scanResults?.data) return;

      try {
        setIsLoadingNeo4jStructure(true);
        const machineId = localStorage.getItem('machineId');
        
        if (!machineId) {
          setNeo4jDirectoryStructure(null);
          return;
        }

        // Get the root directory's fullPath
        // Use fullPath instead of relativePath to ensure each directory is stored independently
        const rootFullPath = scanResults.data.fullPath || '';
        
        if (!rootFullPath) {
          setNeo4jDirectoryStructure(null);
          return;
        }
        
        // Fetch structure from Neo4j using fullPath
        const result = await getDirectoryFromNeo4j(machineId, rootFullPath);
        
        if (result.found && result.structure) {
          setNeo4jDirectoryStructure(result.structure);
          
          // Initialize RAG state from Neo4j structure
          const initializeNeo4jRAGState = (node) => {
            if (node.type === 'file') {
              // Use fullPath as the key to match backend storage
              const fileKey = node.fullPath || node.id;
              // Ensure we properly handle the RAG values from Neo4j
              const ragSelected = node.hasOwnProperty('ragSelected') ? node.ragSelected : false;
              const ragStatus = node.hasOwnProperty('ragStatus') && node.ragStatus ? node.ragStatus : 'unselected';
              
              setNeo4jFileSelections(prev => ({
                ...prev,
                [fileKey]: ragSelected
              }));
              setNeo4jFileAlphanumericValues(prev => ({
                ...prev,
                [fileKey]: ragStatus === 'unselected' ? '' : ragStatus
              }));
            }
            if (node.children && Array.isArray(node.children)) {
              node.children.forEach(child => initializeNeo4jRAGState(child));
            }
          };
          
          // Clear existing state before initializing
          setNeo4jFileSelections({});
          setNeo4jFileAlphanumericValues({});
          
          initializeNeo4jRAGState(result.structure);
        } else {
          setNeo4jDirectoryStructure(null);
          setNeo4jFileSelections({});
          setNeo4jFileAlphanumericValues({});
        }
      } catch (error) {
        console.error('Error fetching Neo4j directory structure:', error);
        setNeo4jDirectoryStructure(null);
      } finally {
        setIsLoadingNeo4jStructure(false);
      }
    };

    fetchNeo4jStructure();
  }, [scanResults?.data?.fullPath]);

  if (!scanResults) {
    return (
      <div className="card text-center py-12">
        <p className="text-slate-600 mb-4">No scan results available</p>
        <Link to="/scanner" className="btn btn-primary inline-flex items-center">
          Start Scanning
        </Link>
      </div>
    );
  }

  const handleCopy = () => {
    const dataStr = JSON.stringify(scanResults.data, null, 2);
    navigator.clipboard.writeText(dataStr).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = dataStr;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleDownload = () => {
    const dataStr = JSON.stringify(scanResults.data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `scan-results-${new Date().toISOString()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleStoreInNeo4j = async () => {
    try {
      setIsStoring(true);
      setStoreMessage('');
      
      // Check Neo4j health first
      const health = await checkNeo4jHealth();
      if (!health.neo4j_connected) {
        setStoreMessage('❌ Neo4j database is not connected');
        return;
      }
      
      // Get machineId from localStorage - required for directory identity
      const machineId = localStorage.getItem('machineId');
      if (!machineId) {
        setStoreMessage('❌ Machine ID not found. Please refresh the page to register your machine.');
        return;
      }
      
      // Prepare RAG data for all files (with default values)
      // Use machineId:fullPath as key to ensure consistency with backend storage
      const ragData = {};
      
      // Get all files with their full paths and compute keys using machineId:fullPath
      const getAllFilesWithKeys = (node) => {
        const files = [];
        if (node.type === 'file') {
          // Use machineId:fullPath as the key for RAG data
          // fullPath is unique per file and ensures each directory is stored independently
          const fileKey = `${machineId}:${node.fullPath}`;
          files.push({ key: fileKey, originalId: node.id, fullPath: node.fullPath });
        }
        if (node.children) {
          node.children.forEach(child => {
            files.push(...getAllFilesWithKeys(child));
          });
        }
        return files;
      };
      
      const allFiles = getAllFilesWithKeys(scanResults.data);
      
      // Create RAG data using machineId:fullPath keys
      // Also support lookup by original file ID for backward compatibility with UI state
      allFiles.forEach(file => {
        const originalId = file.originalId;
        const fileKey = file.key;
        
        // Use machineId:fullPath as the primary key for RAG data
        ragData[fileKey] = {
          selected: fileSelections[originalId] || false,
          status: fileAlphanumericValues[originalId] || 'unselected'
        };
        
        // Also support lookup by original ID (for UI state management)
        // This allows the UI to continue using original IDs while backend uses machineId:fullPath
      });
      
      // Store the file structure with RAG data and machineId
      const result = await storeInNeo4j(
        scanResults.data, 
        {
          scanTimestamp: new Date().toISOString(),
          source: scanResults.source,
          totalFiles: scanResults.totalFiles,
          totalFolders: scanResults.totalFolders
        }, 
        ragData,
        machineId  // Pass machineId to backend for directory key generation
      );
      
      setStoreMessage(`✅ Successfully stored in Neo4j! Root ID: ${result.root_id}`);
      
      // Get updated stats
      const stats = await getNeo4jStats();
      setNeo4jStatus(stats);
      
      // Refresh Neo4j directory structure after storing
      // Add a small delay to ensure Neo4j transaction is committed
      const rootFullPath = scanResults.data.fullPath || '';
      if (rootFullPath) {
        // Wait a bit to ensure Neo4j has committed the transaction
        await new Promise(resolve => setTimeout(resolve, 500));
        
        try {
          const neo4jResult = await getDirectoryFromNeo4j(machineId, rootFullPath);
          if (neo4jResult.found && neo4jResult.structure) {
            setNeo4jDirectoryStructure(neo4jResult.structure);
            
            // Initialize RAG state from Neo4j structure
            const initializeNeo4jRAGState = (node) => {
              if (node.type === 'file') {
                // Use fullPath as the key to match backend storage
                const fileKey = node.fullPath || node.id;
                // Ensure we properly handle the RAG values from Neo4j
                const ragSelected = node.hasOwnProperty('ragSelected') ? node.ragSelected : false;
                const ragStatus = node.hasOwnProperty('ragStatus') && node.ragStatus ? node.ragStatus : 'unselected';
                
                setNeo4jFileSelections(prev => ({
                  ...prev,
                  [fileKey]: ragSelected
                }));
                setNeo4jFileAlphanumericValues(prev => ({
                  ...prev,
                  [fileKey]: ragStatus === 'unselected' ? '' : ragStatus
                }));
              }
              if (node.children && Array.isArray(node.children)) {
                node.children.forEach(child => initializeNeo4jRAGState(child));
              }
            };
            
            // Clear existing state before initializing
            setNeo4jFileSelections({});
            setNeo4jFileAlphanumericValues({});
            
            initializeNeo4jRAGState(neo4jResult.structure);
          }
        } catch (error) {
          console.error('Error refreshing Neo4j structure:', error);
        }
      }
      
    } catch (error) {
      setStoreMessage(`❌ Error storing in Neo4j: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsStoring(false);
    }
  };

  const handleStoreNeo4jStructure = async () => {
    try {
      setIsStoring(true);
      setStoreMessage('');
      
      if (!neo4jDirectoryStructure) {
        setStoreMessage('❌ No Neo4j directory structure available to store');
        return;
      }
      
      // Check Neo4j health first
      const health = await checkNeo4jHealth();
      if (!health.neo4j_connected) {
        setStoreMessage('❌ Neo4j database is not connected');
        return;
      }
      
      // Get machineId from localStorage
      const machineId = localStorage.getItem('machineId');
      if (!machineId) {
        setStoreMessage('❌ Machine ID not found. Please refresh the page to register your machine.');
        return;
      }
      
      // Prepare RAG data for all files in Neo4j structure using machineId:fullPath keys
      const ragData = {};
      
      // Get all files with their full paths and compute keys using machineId:fullPath
      const getAllFilesWithKeys = (node) => {
        const files = [];
        if (node.type === 'file') {
          // Use machineId:fullPath as the key for RAG data
          const fileKey = `${machineId}:${node.fullPath || node.id}`;
          files.push({ key: fileKey, fullPath: node.fullPath || node.id });
        }
        if (node.children) {
          node.children.forEach(child => {
            files.push(...getAllFilesWithKeys(child));
          });
        }
        return files;
      };
      
      const allFiles = getAllFilesWithKeys(neo4jDirectoryStructure);
      
      // Create RAG data using machineId:fullPath keys from Neo4j structure state
      allFiles.forEach(file => {
        const fileKey = file.key;
        const fullPath = file.fullPath;
        
        // Get values from Neo4j RAG state
        ragData[fileKey] = {
          selected: neo4jFileSelections[fullPath] || false,
          status: neo4jFileAlphanumericValues[fullPath] || 'unselected'
        };
      });
      
      // Store the Neo4j structure with updated RAG data
      const result = await storeInNeo4j(
        neo4jDirectoryStructure,
        {
          scanTimestamp: new Date().toISOString(),
          source: neo4jDirectoryStructure.source || 'neo4j',
          totalFiles: countFilesInStructure(neo4jDirectoryStructure),
          totalFolders: countFoldersInStructure(neo4jDirectoryStructure)
        },
        ragData,
        machineId
      );
      
      setStoreMessage(`✅ Successfully stored Neo4j structure with updated RAG data! Root ID: ${result.root_id}`);
      
      // Get updated stats
      const stats = await getNeo4jStats();
      setNeo4jStatus(stats);
      
      // Refresh Neo4j directory structure after storing
      // Add a small delay to ensure Neo4j transaction is committed
      const rootFullPath = neo4jDirectoryStructure.fullPath || '';
      if (rootFullPath) {
        // Wait a bit to ensure Neo4j has committed the transaction
        await new Promise(resolve => setTimeout(resolve, 500));
        
        try {
          const neo4jResult = await getDirectoryFromNeo4j(machineId, rootFullPath);
          if (neo4jResult.found && neo4jResult.structure) {
            setNeo4jDirectoryStructure(neo4jResult.structure);
            // Re-initialize RAG state from refreshed structure
            const initializeNeo4jRAGState = (node) => {
              if (node.type === 'file') {
                const fileKey = node.fullPath || node.id;
                // Ensure we properly handle the RAG values from Neo4j
                const ragSelected = node.hasOwnProperty('ragSelected') ? node.ragSelected : false;
                const ragStatus = node.hasOwnProperty('ragStatus') && node.ragStatus ? node.ragStatus : 'unselected';
                
                setNeo4jFileSelections(prev => ({
                  ...prev,
                  [fileKey]: ragSelected
                }));
                setNeo4jFileAlphanumericValues(prev => ({
                  ...prev,
                  [fileKey]: ragStatus === 'unselected' ? '' : ragStatus
                }));
              }
              if (node.children && Array.isArray(node.children)) {
                node.children.forEach(child => initializeNeo4jRAGState(child));
              }
            };
            
            // Clear existing state before re-initializing
            setNeo4jFileSelections({});
            setNeo4jFileAlphanumericValues({});
            
            initializeNeo4jRAGState(neo4jResult.structure);
          }
        } catch (error) {
          console.error('Error refreshing Neo4j structure:', error);
        }
      }
      
    } catch (error) {
      setStoreMessage(`❌ Error storing Neo4j structure: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsStoring(false);
    }
  };

  const countFilesInStructure = (node) => {
    if (node.type === 'file') return 1;
    if (!node.children) return 0;
    return node.children.reduce((sum, child) => sum + countFilesInStructure(child), 0);
  };

  const handleUploadDirectory = async (directoryNode) => {
    try {
      const machineId = localStorage.getItem('machineId');
      if (!machineId) return;
      setUploadStatus(prev => ({ ...prev, [directoryNode.fullPath || directoryNode.id]: 'Uploading...' }));
      const res = await uploadDirectoryToNeo4j(machineId, directoryNode.fullPath || directoryNode.id);
      setUploadStatus(prev => ({ ...prev, [directoryNode.fullPath || directoryNode.id]: `Uploaded: ${res.summary.created_chunks} chunks` }));
    } catch (e) {
      setUploadStatus(prev => ({ ...prev, [directoryNode.fullPath || directoryNode.id]: `Error: ${e.response?.data?.detail || e.message}` }));
    }
  };

  const renderNeo4jNodeWithUpload = (node, level = 0) => {
    const isDirectory = node.type === 'directory';
    const Icon = isDirectory ? FolderOpen : FileJson;
    const isFile = !isDirectory;
    return (
      <div key={node.id} className="ml-4 my-2">
        <div className="flex items-center justify-between text-slate-700">
          <div className="flex items-center space-x-2">
            <Icon className={`w-4 h-4 ${isDirectory ? 'text-blue-500' : 'text-slate-500'}`} />
            <span className="font-medium">{node.name}</span>
            {node.size && (
              <span className="text-xs text-slate-500">({formatBytes(node.size)})</span>
            )}
          </div>
          {isDirectory ? (
            <div className="flex items-center space-x-2">
              <button
                onClick={() => handleUploadDirectory(node)}
                className="btn btn-secondary flex items-center"
              >
                <Upload className="w-4 h-4 mr-2" /> Upload Documents
              </button>
              {uploadStatus[node.fullPath || node.id] && (
                <span className="text-xs text-slate-500">{uploadStatus[node.fullPath || node.id]}</span>
              )}
            </div>
          ) : (
            <div className="flex items-center space-x-2">
              <label className="flex items-center space-x-1 cursor-pointer">
                <input type="checkbox" checked={neo4jFileSelections[node.fullPath || node.id] || false} onChange={() => {}} readOnly className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2" />
                <span className="text-xs text-slate-600">RAG</span>
              </label>
              <input type="text" value={neo4jFileAlphanumericValues[node.fullPath || node.id] || ''} readOnly className="px-2 py-1 text-xs border border-gray-300 rounded w-24 bg-gray-50" maxLength={20} />
            </div>
          )}
        </div>
        {node.children && node.children.length > 0 && (
          <div className="border-l-2 border-slate-200">
            {node.children.map(child => renderNeo4jNodeWithUpload(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  const countFoldersInStructure = (node) => {
    if (node.type === 'directory') {
      return 1 + (node.children ? node.children.reduce((sum, child) => sum + countFoldersInStructure(child), 0) : 0);
    }
    return 0;
  };

  const checkNeo4jConnection = async () => {
    try {
      const health = await checkNeo4jHealth();
      setNeo4jStatus(health);
      
      if (health.neo4j_connected) {
        const stats = await getNeo4jStats();
        setNeo4jStatus(stats);
      }
    } catch (error) {
      setNeo4jStatus({ status: 'unhealthy', neo4j_connected: false, error: error.message });
    }
  };

  const handleFileSelection = (fileId, isSelected) => {
    setFileSelections(prev => ({
      ...prev,
      [fileId]: isSelected
    }));
  };

  const handleAlphanumericChange = (fileId, value) => {
    setFileAlphanumericValues(prev => ({
      ...prev,
      [fileId]: value
    }));
  };

  const findFileById = (node, targetId) => {
    if (node.id === targetId) {
      return node;
    }
    if (node.children) {
      for (const child of node.children) {
        const found = findFileById(child, targetId);
        if (found) return found;
      }
    }
    return null;
  };

  // Render function for scanned directory structure (with interactive RAG fields)
  const renderNode = (node, level = 0) => {
    const isDirectory = node.type === 'directory';
    const Icon = isDirectory ? FolderOpen : FileJson;
    const isFile = !isDirectory;
    
    return (
      <div key={node.id} className="ml-4 my-2">
        <div className="flex items-center justify-between text-slate-700">
          <div className="flex items-center space-x-2">
            <Icon className={`w-4 h-4 ${isDirectory ? 'text-blue-500' : 'text-slate-500'}`} />
            <span className="font-medium">{node.name}</span>
            {node.size && (
              <span className="text-xs text-slate-500">
                ({formatBytes(node.size)})
              </span>
            )}
          </div>
          
          {/* Add checkbox and alphanumeric field for files only - aligned to the right */}
          {isFile && (
            <div className="flex items-center space-x-2">
              <label className="flex items-center space-x-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={fileSelections[node.id] || false}
                  onChange={(e) => handleFileSelection(node.id, e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                />
                <span className="text-xs text-slate-600">RAG</span>
              </label>
              
              <input
                type="text"
                placeholder="RAG Status..."
                value={fileAlphanumericValues[node.id] === 'unselected' ? '' : (fileAlphanumericValues[node.id] || '')}
                onChange={(e) => handleAlphanumericChange(node.id, e.target.value)}
                className="px-2 py-1 text-xs border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-24"
                maxLength={20}
              />
            </div>
          )}
        </div>
        {node.children && node.children.length > 0 && (
          <div className="border-l-2 border-slate-200">
            {node.children.map(child => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  // Handlers for Neo4j structure RAG fields
  const handleNeo4jFileSelection = (fileKey, checked) => {
    setNeo4jFileSelections(prev => ({
      ...prev,
      [fileKey]: checked
    }));
  };

  const handleNeo4jAlphanumericChange = (fileKey, value) => {
    setNeo4jFileAlphanumericValues(prev => ({
      ...prev,
      [fileKey]: value
    }));
  };

  // Render function for Neo4j directory structure (interactive, allowing RAG editing)
  const renderNeo4jNode = (node, level = 0) => {
    const isDirectory = node.type === 'directory';
    const Icon = isDirectory ? FolderOpen : FileJson;
    const isFile = !isDirectory;
    
    // Use fullPath as the key to match backend storage format
    const fileKey = node.fullPath || node.id;
    
    return (
      <div key={node.id} className="ml-4 my-2">
        <div className="flex items-center justify-between text-slate-700">
          <div className="flex items-center space-x-2">
            <Icon className={`w-4 h-4 ${isDirectory ? 'text-blue-500' : 'text-slate-500'}`} />
            <span className="font-medium">{node.name}</span>
            {node.size && (
              <span className="text-xs text-slate-500">
                ({formatBytes(node.size)})
              </span>
            )}
          </div>
          
          {/* Editable RAG fields for files only - aligned to the right */}
          {isFile && (
            <div className="flex items-center space-x-2">
              <label className="flex items-center space-x-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={neo4jFileSelections[fileKey] || false}
                  onChange={(e) => handleNeo4jFileSelection(fileKey, e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                />
                <span className="text-xs text-slate-600">RAG</span>
              </label>
              
              <input
                type="text"
                placeholder="RAG Status..."
                value={neo4jFileAlphanumericValues[fileKey] === 'unselected' ? '' : (neo4jFileAlphanumericValues[fileKey] || '')}
                onChange={(e) => handleNeo4jAlphanumericChange(fileKey, e.target.value)}
                className="px-2 py-1 text-xs border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-24"
                maxLength={20}
              />
            </div>
          )}
        </div>
        {node.children && node.children.length > 0 && (
          <div className="border-l-2 border-slate-200">
            {node.children.map(child => renderNeo4jNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };


  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-slate-800">Scan Results</h2>
          <p className="text-slate-600">Source: {scanResults.source}</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={handleCopy}
            className="btn btn-outline flex items-center"
          >
            {copied ? (
              <>
                <CheckCircle className="w-5 h-5 mr-2" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="w-5 h-5 mr-2" />
                Copy JSON
              </>
            )}
          </button>
          <button
            onClick={handleDownload}
            className="btn btn-primary flex items-center"
          >
            <Download className="w-5 h-5 mr-2" />
            Download JSON
          </button>
          <button
            onClick={handleStoreInNeo4j}
            disabled={isStoring}
            className="btn btn-secondary flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isStoring ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-slate-600 mr-2"></div>
                Storing...
              </>
            ) : (
              <>
                <Database className="w-5 h-5 mr-2" />
                Store in Neo4j
              </>
            )}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card bg-gradient-to-br from-blue-50 to-blue-100">
          <p className="text-sm text-slate-600 mb-1">Total Files</p>
          <p className="text-3xl font-bold text-blue-700">
            {scanResults.totalFiles || 0}
          </p>
        </div>
        <div className="card bg-gradient-to-br from-green-50 to-green-100">
          <p className="text-sm text-slate-600 mb-1">Total Folders</p>
          <p className="text-3xl font-bold text-green-700">
            {scanResults.totalFolders || 0}
          </p>
        </div>
        <div className="card bg-gradient-to-br from-primary-50 to-primary-100">
          <p className="text-sm text-slate-600 mb-1">Source Type</p>
          <p className="text-3xl font-bold text-primary-700 capitalize">
            {scanResults.source}
          </p>
        </div>
      </div>

      {/* Directory Structures - Side by Side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Current Scanned Directory Structure */}
        <div className="card max-h-96 overflow-y-auto">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Directory Structure (Scanned)</h3>
          <div className="space-y-2">
            {renderNode(scanResults.data)}
          </div>
        </div>

        {/* Directory Structure from Neo4j */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-800">
              Directory Structure (Neo4j)
              {isLoadingNeo4jStructure && (
                <span className="ml-2 text-sm text-slate-500 font-normal">Loading...</span>
              )}
            </h3>
            {neo4jDirectoryStructure && (
              <button
                onClick={handleStoreNeo4jStructure}
                disabled={isStoring}
                className="btn btn-primary flex items-center disabled:opacity-50"
              >
                <Upload className="w-4 h-4 mr-2" />
                {isStoring ? 'Storing...' : 'Store in Neo4j'}
              </button>
            )}
          </div>
          <div className="max-h-96 overflow-y-auto space-y-2">
            {neo4jDirectoryStructure ? (
              renderNeo4jNodeWithUpload(neo4jDirectoryStructure)
            ) : (
              <div className="text-center text-slate-500 py-8">
                <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>Directory not found in Neo4j</p>
                <p className="text-sm mt-1">Store the directory to Neo4j to see it here</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Selected Files Summary */}
      {Object.keys(fileSelections).length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Selected Files for RAG</h3>
          <div className="space-y-2">
            {Object.entries(fileSelections)
              .filter(([_, isSelected]) => isSelected)
              .map(([fileId, _]) => {
                const file = findFileById(scanResults.data, fileId);
                return file ? (
                  <div key={fileId} className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <FileJson className="w-4 h-4 text-slate-500" />
                      <span className="font-medium text-slate-700">{file.name}</span>
                      <span className="text-xs text-slate-500">({formatBytes(file.size)})</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-slate-600">RAG Status:</span>
                      <span className="px-2 py-1 bg-white border border-slate-300 rounded text-sm font-mono">
                        {fileAlphanumericValues[fileId] || 'No status'}
                      </span>
                    </div>
                  </div>
                ) : null;
              })}
          </div>
        </div>
      )}

      {storeMessage && (
        <div className={`card ${storeMessage.includes('✅') ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <p className={storeMessage.includes('✅') ? 'text-green-700' : 'text-red-700'}>{storeMessage}</p>
        </div>
      )}

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-800">Neo4j Graph Database</h3>
          <button
            onClick={checkNeo4jConnection}
            className="btn btn-outline flex items-center"
          >
            <Upload className="w-4 h-4 mr-2" />
            Check Connection
          </button>
        </div>
        
        {neo4jStatus ? (
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${neo4jStatus.neo4j_connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm font-medium">
                {neo4jStatus.neo4j_connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {neo4jStatus.total_nodes !== undefined && (
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <p className="text-slate-600">Total Nodes</p>
                  <p className="font-semibold text-slate-800">{neo4jStatus.total_nodes}</p>
                </div>
                <div className="text-center">
                  <p className="text-slate-600">Files</p>
                  <p className="font-semibold text-slate-800">{neo4jStatus.total_files}</p>
                </div>
                <div className="text-center">
                  <p className="text-slate-600">Directories</p>
                  <p className="font-semibold text-slate-800">{neo4jStatus.total_directories}</p>
                </div>
              </div>
            )}
            
            {neo4jStatus.sources && (
              <div>
                <p className="text-sm text-slate-600 mb-1">Sources:</p>
                <div className="flex flex-wrap gap-2">
                  {neo4jStatus.sources.map((source, index) => (
                    <span key={index} className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs">
                      {source}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="text-slate-600 text-sm">Click "Check Connection" to see Neo4j status</p>
        )}
      </div>

      <div className="flex justify-center space-x-4">
        <button
          onClick={() => dispatch(clearResults())}
          className="btn btn-secondary"
        >
          Clear Results
        </button>
        <Link to="/scanner" className="btn btn-primary">
          Scan Again
        </Link>
      </div>
    </div>
  );
};

export default Results;
