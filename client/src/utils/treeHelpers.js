/**
 * Count files in a tree structure
 * @param {object} node - Tree node
 * @returns {number} Total file count
 */
export const countFiles = (node) => {
  if (node.type === 'file') return 1;
  return node.children?.reduce((sum, child) => sum + countFiles(child), 0) || 0;
};

/**
 * Count folders in a tree structure
 * @param {object} node - Tree node
 * @returns {number} Total folder count
 */
export const countFolders = (node) => {
  if (node.type === 'file') return 0;
  return node.children?.reduce((sum, child) => sum + countFolders(child) + 1, 0) || 0;
};

/**
 * Flatten tree structure to array
 * @param {object} node - Tree node
 * @param {array} result - Result array (for recursion)
 * @returns {array} Flattened array
 */
export const flattenTree = (node, result = []) => {
  result.push(node);
  if (node.children) {
    node.children.forEach(child => flattenTree(child, result));
  }
  return result;
};

/**
 * Search/filter tree by name
 * @param {object} node - Tree node
 * @param {string} query - Search query
 * @returns {object|null} Filtered tree
 */
export const filterTreeByName = (node, query) => {
  const matchesQuery = node.name.toLowerCase().includes(query.toLowerCase());
  
  if (node.children) {
    const filteredChildren = node.children
      .map(child => filterTreeByName(child, query))
      .filter(child => child !== null);
    
    if (matchesQuery || filteredChildren.length > 0) {
      return {
        ...node,
        children: filteredChildren
      };
    }
    return null;
  }
  
  return matchesQuery ? node : null;
};
