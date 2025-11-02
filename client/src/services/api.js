import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || '/api';
const NEO4J_API_URL = import.meta.env.VITE_NEO4J_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

const neo4jApi = axios.create({
  baseURL: NEO4J_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Local file system
export const scanLocal = async (directoryPath) => {
  const response = await api.post('/local/scan', { directoryPath });
  return response.data;
};

// Google Drive
export const authenticateGoogleDrive = async (credentials) => {
  const response = await api.post('/drive/authenticate', credentials);
  return response.data;
};

export const scanGoogleDrive = async (stateId) => {
  const response = await api.post('/drive/scan', { stateId });
  return response.data;
};

// SharePoint
export const authenticateSharePoint = async (credentials) => {
  const response = await api.post('/sharepoint/authenticate', credentials);
  return response.data;
};

export const scanSharePoint = async (stateId) => {
  const response = await api.post('/sharepoint/scan', { stateId });
  return response.data;
};

// Machine Registration
export const registerMachine = async () => {
  const response = await neo4jApi.post('/api/register-machine');
  return response.data;
};

// Neo4j Graph Operations
export const storeInNeo4j = async (fileData, metadata = {}, ragData = {}, machineId = null) => {
  const response = await neo4jApi.post('/api/graph/store', {
    data: fileData,
    metadata: metadata,
    rag_data: ragData,
    machine_id: machineId  // Include machineId in the payload
  });
  return response.data;
};

export const getNeo4jStats = async () => {
  const response = await neo4jApi.get('/api/graph/stats');
  return response.data;
};

// Retrieve directory structure from Neo4j
export const getDirectoryFromNeo4j = async (machineId, fullPath) => {
  const response = await neo4jApi.get('/api/graph/directory', {
    params: {
      machine_id: machineId,
      full_path: fullPath
    }
  });
  return response.data;
};

export const uploadDirectoryToNeo4j = async (machineId, fullPath, options = {}) => {
  const params = {
    machine_id: machineId,
    full_path: fullPath,
  };
  if (options.chunk_size) params.chunk_size = options.chunk_size;
  if (options.chunk_overlap) params.chunk_overlap = options.chunk_overlap;
  const response = await neo4jApi.post('/api/graph/upload-directory', null, { params });
  return response.data;
};

export const clearNeo4jGraph = async () => {
  const response = await neo4jApi.post('/api/graph/clear');
  return response.data;
};

export const searchNeo4jFiles = async (name, source = null) => {
  const params = { name };
  if (source) params.source = source;
  const response = await neo4jApi.get('/api/graph/search', { params });
  return response.data;
};

export const getNeo4jDirectoryTree = async (directoryId, maxDepth = 5) => {
  const response = await neo4jApi.get(`/api/graph/tree/${directoryId}`, {
    params: { max_depth: maxDepth }
  });
  return response.data;
};

export const getNeo4jVisualization = async () => {
  const response = await neo4jApi.get('/api/graph/visualization');
  return response.data;
};

export const getNeo4jNodes = async () => {
  const response = await neo4jApi.get('/api/graph/nodes');
  return response.data;
};

export const getNeo4jRelationships = async () => {
  const response = await neo4jApi.get('/api/graph/relationships');
  return response.data;
};

export const checkNeo4jHealth = async () => {
  const response = await neo4jApi.get('/health');
  return response.data;
};

export default api;
