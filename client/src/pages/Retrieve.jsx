import { useState, useEffect, useRef } from 'react';
import { Database, RefreshCw, Download, Eye, EyeOff, Maximize2, X, Trash2, FolderOpen, FileJson } from 'lucide-react';
import { getNeo4jStats, getNeo4jVisualization } from '../services/api';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data';
import { formatBytes } from '../utils/formatters';

const Retrieve = () => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('2d'); // '2d' or '3d'
  const [selectedNode, setSelectedNode] = useState(null);
  const [stats, setStats] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const networkRef = useRef(null);
  const containerRef = useRef(null);
  const fullscreenContainerRef = useRef(null);

  const fetchGraphData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Get stats first
      const statsData = await getNeo4jStats();
      setStats(statsData);
      
      // Get actual graph data from Neo4j
      const visualizationData = await getNeo4jVisualization();
      setGraphData(visualizationData);
      
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch graph data');
      // Fallback to empty data
      setGraphData({ nodes: [], links: [] });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchGraphData();
  }, []);

  useEffect(() => {
    if (graphData.nodes.length > 0 && containerRef.current) {
      createNetwork(containerRef.current, false);
    }
  }, [graphData]);

  useEffect(() => {
    if (isFullscreen && graphData.nodes.length > 0 && fullscreenContainerRef.current) {
      createNetwork(fullscreenContainerRef.current, true);
    }
  }, [isFullscreen, graphData]);

  const createNetwork = (container, isFullscreen = false) => {
    if (!container) return;

    console.log('Creating network with data:', graphData);
    console.log('Nodes count:', graphData.nodes.length);
    console.log('Links count:', graphData.links.length);

    // Prepare nodes for vis-network with RAG hierarchy - simplified
    const nodes = new DataSet(
      graphData.nodes.map(node => {
        // Simplified color and size assignment
        let color = '#3b82f6'; // Default blue
        let size = 20; // Default size
        
        if (node.type === 'directory') {
          color = '#3b82f6'; // Blue
          size = 50; // Large directories
        } else if (node.type === 'file') {
          color = '#10b981'; // Green
          size = 2; // Extremely small files
        } else if (node.type === 'rag_checkbox') {
          color = '#f59e0b'; // Orange
          size = 20; // Medium
        } else if (node.type === 'rag_status') {
          color = '#ef4444'; // Red
          size = 20; // Medium
        }
        
        console.log('Creating node:', node.name, 'type:', node.type, 'color:', color, 'size:', size, 'final node:', {
          id: node.id,
          label: node.name,
          color: color,
          size: size
        });
        
        return {
          id: node.id,
          label: node.name,
          color: color,
          value: size, // Use value instead of size
          font: { color: '#ffffff', size: 12 },
          borderWidth: 2,
          borderColor: '#000000'
        };
      })
    );

    // Prepare edges for vis-network - simplified
    const edges = new DataSet(
      graphData.links.map(link => ({
        from: link.source,
        to: link.target,
        color: '#848484',
        width: 1 // Much thinner edges
      }))
    );

    // Network options - simplified for better node visibility
    const options = {
      nodes: {
        shape: 'circle',
        font: {
          size: 14,
          color: '#000000',
          strokeWidth: 0,
          strokeColor: '#ffffff'
        },
        borderWidth: 3,
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.2)',
          size: 10,
          x: 5,
          y: 5
        },
        scaling: {
          min: 5,
          max: 70,
          label: {
            enabled: false
          }
        }
      },
      edges: {
        width: 1, // Thinner edges
        color: '#848484',
        smooth: {
          type: 'continuous',
          forceDirection: 'none',
          roundness: 0.5
        }
      },
      physics: {
        enabled: true,
        stabilization: { 
          iterations: 200,
          updateInterval: 50,
          onlyDynamicEdges: false,
          fit: true
        },
        barnesHut: {
          gravitationalConstant: -2000,
          centralGravity: 0.3,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.09,
          avoidOverlap: 0.1
        }
      },
      interaction: {
        hover: true,
        selectConnectedEdges: false,
        tooltipDelay: 200
      },
      layout: {
        improvedLayout: true
      }
    };

    // Ensure we have nodes before creating network
    if (nodes.length === 0) {
      console.warn('No nodes to display');
      return;
    }

    console.log('Creating network with', nodes.length, 'nodes and', edges.length, 'edges');
    
    // Clear any existing network
    if (networkRef.current) {
      networkRef.current.destroy();
    }
    
    // Create network
    const network = new Network(container, { nodes, edges }, options);
    networkRef.current = network;
    
    // Force a redraw to ensure nodes are visible
    setTimeout(() => {
      network.redraw();
      network.fit();
    }, 100);

    // Event handlers
    network.on('selectNode', (params) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const node = graphData.nodes.find(n => n.id === nodeId);
        setSelectedNode(node);
      }
    });

    network.on('deselectNode', () => {
      setSelectedNode(null);
    });

    return network;
  };

  const exportGraphData = () => {
    const dataStr = JSON.stringify(graphData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `graph-data-${new Date().toISOString()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const deleteNode = async () => {
    if (!selectedNode) return;
    
    if (!confirm(`Are you sure you want to delete "${selectedNode.name}"?\n\nThis will remove the node and all its relationships from Neo4j.`)) {
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Call the delete API endpoint
      const response = await fetch(`http://localhost:8000/api/graph/nodes/${selectedNode.id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to delete node: ${response.statusText}`);
      }

      // Refresh the graph data
      await fetchGraphData();
      
      // Clear selected node
      setSelectedNode(null);
      
      alert('Node deleted successfully!');
    } catch (err) {
      setError(err.message || 'Failed to delete node');
      console.error('Delete node error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const deleteAllRecords = async () => {
    if (!confirm('Are you sure you want to delete ALL records from Neo4j?\n\nThis will remove all nodes and relationships. This action cannot be undone.')) {
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Call the clear API endpoint
      const response = await fetch('http://localhost:8000/api/graph/clear', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to clear database: ${response.statusText}`);
      }

      // Refresh the graph data
      await fetchGraphData();
      
      // Clear selected node
      setSelectedNode(null);
      
      alert('All records deleted successfully!');
    } catch (err) {
      setError(err.message || 'Failed to delete all records');
      console.error('Delete all records error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const renderDirectoryStructure = (nodes, links) => {
    // Convert graph nodes to tree structure using relationships
    const buildTreeFromGraph = (nodes, links) => {
      // Filter nodes by type
      const fileNodes = nodes.filter(node => node.type === 'file');
      const directoryNodes = nodes.filter(node => node.type === 'directory');
      const ragCheckboxNodes = nodes.filter(node => node.type === 'rag_checkbox');
      const ragStatusNodes = nodes.filter(node => node.type === 'rag_status');
      
      // Create a map for quick node lookup
      const nodeMap = new Map();
      nodes.forEach(node => {
        nodeMap.set(node.id, node);
      });
      
      // Create a map of parent -> children relationships from CONTAINS links
      // Only consider CONTAINS relationships (not HAS_RAG or HAS_RAG_STATUS)
      const childrenMap = new Map();
      const parentMap = new Map(); // Track which nodes have parents
      
      links.forEach(link => {
        if (link.type === 'CONTAINS') {
          const parentId = link.source;
          const childId = link.target;
          
          if (!childrenMap.has(parentId)) {
            childrenMap.set(parentId, []);
          }
          childrenMap.get(parentId).push(childId);
          
          // Mark this node as having a parent
          parentMap.set(childId, parentId);
        }
      });
      
      // Find root directories (directories that are not children of any other directory)
      const rootDirectories = directoryNodes.filter(dir => !parentMap.has(dir.id));
      
      // Build tree structure recursively
      const buildNode = (nodeId) => {
        const node = nodeMap.get(nodeId);
        if (!node) return null;
        
        if (node.type === 'directory') {
          const dirNode = {
            id: node.id,
            name: node.name,
            type: 'directory',
            size: node.size || 0,
            children: []
          };
          
          // Get children of this directory from CONTAINS relationships
          const childIds = childrenMap.get(nodeId) || [];
          
          childIds.forEach(childId => {
            const child = buildNode(childId);
            if (child) {
              dirNode.children.push(child);
            }
          });
          
          // Sort children: directories first, then files
          dirNode.children.sort((a, b) => {
            if (a.type === 'directory' && b.type === 'file') return -1;
            if (a.type === 'file' && b.type === 'directory') return 1;
            return a.name.localeCompare(b.name);
          });
          
          return dirNode;
        } else if (node.type === 'file') {
          // Find RAG data for this file
          const ragCheckbox = ragCheckboxNodes.find(rag => 
            rag.id.includes(node.id)
          );
          const ragStatus = ragStatusNodes.find(rag => 
            rag.id.includes(node.id)
          );
          
          return {
            id: node.id,
            name: node.name,
            type: 'file',
            size: node.size || 0,
            ragSelected: ragCheckbox ? ragCheckbox.value === '1' : false,
            ragStatus: ragStatus ? ragStatus.value : 'unselected',
            children: []
          };
        }
        
        return null;
      };
      
      // Build tree starting from root directories
      const tree = rootDirectories.map(rootDir => buildNode(rootDir.id)).filter(Boolean);
      
      return tree;
    };
    
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
            
            {/* Add RAG fields for files only - aligned to the right */}
            {isFile && (
              <div className="flex items-center space-x-2">
                <label className="flex items-center space-x-1 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={node.ragSelected || false}
                    readOnly
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                  />
                  <span className="text-xs text-slate-600">RAG</span>
                </label>
                
                <input
                  type="text"
                  value={node.ragStatus === 'unselected' ? '' : (node.ragStatus || '')}
                  readOnly
                  className="px-2 py-1 text-xs border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-24 bg-gray-50"
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
    
    const tree = buildTreeFromGraph(nodes, links);
    return tree.map(node => renderNode(node));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-slate-800">Graph Retrieve</h2>
          <p className="text-slate-600">Visualize your Neo4j graph database</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={toggleFullscreen}
            className="btn btn-outline flex items-center"
            disabled={graphData.nodes.length === 0}
          >
            <Maximize2 className="w-5 h-5 mr-2" />
            Fullscreen
          </button>
          <button
            onClick={() => setViewMode(viewMode === '2d' ? '3d' : '2d')}
            className="btn btn-outline flex items-center"
          >
            {viewMode === '2d' ? <EyeOff className="w-5 h-5 mr-2" /> : <Eye className="w-5 h-5 mr-2" />}
            {viewMode === '2d' ? '3D View' : '2D View'}
          </button>
          <button
            onClick={fetchGraphData}
            disabled={isLoading}
            className="btn btn-outline flex items-center disabled:opacity-50"
          >
            <RefreshCw className={`w-5 h-5 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={exportGraphData}
            className="btn btn-primary flex items-center"
          >
            <Download className="w-5 h-5 mr-2" />
            Export Graph
          </button>
        </div>
      </div>

      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card bg-gradient-to-br from-blue-50 to-blue-100">
            <p className="text-sm text-slate-600 mb-1">Total Nodes</p>
            <p className="text-3xl font-bold text-blue-700">{stats.total_nodes || 0}</p>
          </div>
          <div className="card bg-gradient-to-br from-green-50 to-green-100">
            <p className="text-sm text-slate-600 mb-1">Files</p>
            <p className="text-3xl font-bold text-green-700">{stats.total_files || 0}</p>
          </div>
          <div className="card bg-gradient-to-br from-purple-50 to-purple-100">
            <p className="text-sm text-slate-600 mb-1">Directories</p>
            <p className="text-3xl font-bold text-purple-700">{stats.total_directories || 0}</p>
          </div>
          <div className="card bg-gradient-to-br from-orange-50 to-orange-100">
            <p className="text-sm text-slate-600 mb-1">Sources</p>
            <p className="text-3xl font-bold text-orange-700">{stats.sources?.length || 0}</p>
          </div>
        </div>
      )}

      {error && (
        <div className="card bg-red-50 border-red-200">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Graph Visualization */}
        <div className="lg:col-span-2">
          <div className="card h-96">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">Graph Visualization</h3>
            {isLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
              </div>
            ) : graphData.nodes.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center text-slate-500">
                  <Database className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-medium">No data available</p>
                  <p className="text-sm">Scan some directories first to see the graph</p>
                  <button 
                    onClick={() => {
                      // Create test data to verify visualization works
                      const testData = {
                        nodes: [
                          { id: 'test1', name: 'Test File', type: 'file', size: 2 },
                          { id: 'test2', name: 'Test Directory', type: 'directory', size: 50 },
                          { id: 'test3', name: 'RAG Checkbox', type: 'rag_checkbox', size: 20 },
                          { id: 'test4', name: 'RAG Status', type: 'rag_status', size: 20 }
                        ],
                        links: [
                          { source: 'test2', target: 'test1', type: 'CONTAINS' },
                          { source: 'test1', target: 'test3', type: 'HAS_RAG' },
                          { source: 'test1', target: 'test4', type: 'HAS_RAG_STATUS' }
                        ]
                      };
                      setGraphData(testData);
                    }}
                    className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                  >
                    Test Visualization
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div 
                  ref={containerRef} 
                  className="w-full h-full border border-slate-200 rounded"
                  style={{ minHeight: '300px' }}
                />
                
              </div>
            )}
          </div>
        </div>

        {/* Node Details */}
        <div className="lg:col-span-1">
          <div className="card h-96">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">Node Details</h3>
            {selectedNode ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-slate-700">Name</h4>
                  <p className="text-slate-600">{selectedNode.name}</p>
                </div>
                <div>
                  <h4 className="font-medium text-slate-700">Type</h4>
                  <p className="text-slate-600 capitalize">{selectedNode.type}</p>
                </div>
                {selectedNode.size && (
                  <div>
                    <h4 className="font-medium text-slate-700">Size</h4>
                    <p className="text-slate-600">{selectedNode.size} bytes</p>
                  </div>
                )}
                <div>
                  <h4 className="font-medium text-slate-700">ID</h4>
                  <p className="text-slate-600 text-xs font-mono">{selectedNode.id}</p>
                </div>
                <div className="pt-4 border-t border-slate-200 space-y-2">
                  <button
                    onClick={deleteNode}
                    disabled={isLoading}
                    className="btn btn-danger flex items-center w-full disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    {isLoading ? 'Deleting...' : 'Delete Node'}
                  </button>
                  
                  <button
                    onClick={deleteAllRecords}
                    disabled={isLoading}
                    className="btn btn-outline border-red-500 text-red-500 hover:bg-red-50 flex items-center w-full disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    {isLoading ? 'Clearing...' : 'Delete All Records'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-500">
                <div className="text-center">
                  <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Click on a node to see details</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Debug Info */}
      <div className="card">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Debug Info</h3>
        <div className="space-y-2 text-sm">
          <p><strong>Nodes Count:</strong> {graphData.nodes.length}</p>
          <p><strong>Links Count:</strong> {graphData.links.length}</p>
          <p><strong>Container Ref:</strong> {containerRef.current ? 'Available' : 'Not available'}</p>
          <p><strong>Network Ref:</strong> {networkRef.current ? 'Created' : 'Not created'}</p>
        </div>
      </div>

      {/* Directory Structure */}
      <div className="card">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Directory Structure</h3>
        <div className="max-h-96 overflow-y-auto">
          {graphData.nodes.length > 0 ? (
            <div className="space-y-1">
              {renderDirectoryStructure(graphData.nodes, graphData.links)}
            </div>
          ) : (
            <div className="text-center text-slate-500 py-8">
              <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No directory structure available</p>
              <p className="text-sm">Scan some directories first to see the structure</p>
            </div>
          )}
        </div>
      </div>

      {/* Graph Data Table */}
      <div className="card">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Graph Data</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Size
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Connections
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-slate-200">
              {graphData.nodes.map((node) => (
                <tr key={node.id} className="hover:bg-slate-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">
                    {node.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      node.type === 'directory' 
                        ? 'bg-blue-100 text-blue-800' 
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {node.type}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    {node.size ? `${node.size} bytes` : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    {graphData.links.filter(link => 
                      link.source === node.id || link.target === node.id
                    ).length}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Fullscreen Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg w-full h-full max-w-7xl max-h-full flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-xl font-semibold text-slate-800">Graph Visualization - Fullscreen</h3>
              <button
                onClick={toggleFullscreen}
                className="btn btn-outline flex items-center"
              >
                <X className="w-5 h-5 mr-2" />
                Close
              </button>
            </div>
            <div className="flex-1 p-4">
              <div 
                ref={fullscreenContainerRef} 
                className="w-full h-full border border-slate-200 rounded"
                style={{ minHeight: 'calc(100vh - 200px)' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Retrieve;
