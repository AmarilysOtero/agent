/**
 * Application constants
 */

export const SOURCES = {
  LOCAL: 'local',
  GOOGLE_DRIVE: 'google-drive',
  SHAREPOINT: 'sharepoint'
};

export const SOURCE_CONFIG = {
  [SOURCES.LOCAL]: {
    name: 'Local Directory',
    description: 'Scan files from your local machine',
    icon: 'Folder',
    color: 'blue',
    requiresAuth: false
  },
  [SOURCES.GOOGLE_DRIVE]: {
    name: 'Google Drive',
    description: 'Connect and scan Google Drive',
    icon: 'HardDrive',
    color: 'green',
    requiresAuth: true
  },
  [SOURCES.SHAREPOINT]: {
    name: 'SharePoint',
    description: 'Connect and scan SharePoint',
    icon: 'Share2',
    color: 'primary',
    requiresAuth: true
  }
};

export const API_ENDPOINTS = {
  LOCAL: '/api/local',
  DRIVE: '/api/drive',
  SHAREPOINT: '/api/sharepoint'
};

export const STORAGE_KEYS = {
  SCAN_RESULTS: 'rag_scanner_results',
  AUTH_STATE: 'rag_auth_state',
  SETTINGS: 'rag_settings'
};

export const SUPPORTED_FILE_TYPES = [
  'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
  'jpg', 'jpeg', 'png', 'gif', 'svg', 'webp',
  'mp4', 'avi', 'mov', 'wmv',
  'mp3', 'wav', 'flac',
  'zip', 'rar', '7z',
  'js', 'ts', 'jsx', 'tsx', 'py', 'java', 'cpp', 'c', 'h',
  'html', 'css', 'json', 'xml', 'yaml', 'yml'
];
