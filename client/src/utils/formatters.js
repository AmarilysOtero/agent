/**
 * Format bytes to human readable size
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size string
 */
export const formatBytes = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  if (!bytes || isNaN(bytes)) return 'Unknown size';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
};

/**
 * Format date string
 * @param {string} dateString - ISO date string
 * @returns {string} Formatted date
 */
export const formatDate = (dateString) => {
  if (!dateString) return 'Unknown date';
  try {
    return new Date(dateString).toLocaleString();
  } catch (error) {
    return dateString;
  }
};

/**
 * Get file icon based on extension
 * @param {string} extension - File extension
 * @returns {string} Icon name or type
 */
export const getFileIcon = (extension) => {
  const ext = extension?.toLowerCase() || '';
  
  if (['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'].includes(ext)) {
    return 'image';
  }
  if (['.pdf'].includes(ext)) return 'pdf';
  if (['.doc', '.docx'].includes(ext)) return 'word';
  if (['.xls', '.xlsx'].includes(ext)) return 'excel';
  if (['.zip', '.rar', '.7z'].includes(ext)) return 'archive';
  if (['.mp4', '.avi', '.mov', '.wmv'].includes(ext)) return 'video';
  if (['.mp3', '.wav', '.flac'].includes(ext)) return 'audio';
  return 'document';
};

/**
 * Calculate total size of a node
 * @param {object} node - File tree node
 * @returns {number} Total size in bytes
 */
export const calculateTotalSize = (node) => {
  if (node.type === 'file') {
    return node.size || 0;
  }
  return node.children?.reduce((sum, child) => sum + calculateTotalSize(child), 0) || 0;
};
