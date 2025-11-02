import json
import requests
from typing import Dict, Any, Optional

class Neo4jAPIClient:
    """Client for interacting with the Neo4j backend API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API and Neo4j are healthy"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def store_file_structure(self, file_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store file structure in Neo4j"""
        payload = {
            "data": file_data,
            "metadata": metadata or {}
        }
        
        response = self.session.post(f"{self.base_url}/api/graph/store", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        response = self.session.get(f"{self.base_url}/api/graph/stats")
        response.raise_for_status()
        return response.json()
    
    def clear_graph(self) -> Dict[str, Any]:
        """Clear all data from the graph"""
        response = self.session.post(f"{self.base_url}/api/graph/clear")
        response.raise_for_status()
        return response.json()
    
    def search_files(self, name: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Search for files by name"""
        params = {"name": name}
        if source:
            params["source"] = source
            
        response = self.session.get(f"{self.base_url}/api/graph/search", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_directory_tree(self, directory_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """Get directory tree structure"""
        params = {"max_depth": max_depth}
        response = self.session.get(f"{self.base_url}/api/graph/tree/{directory_id}", params=params)
        response.raise_for_status()
        return response.json()

def test_api():
    """Test the Neo4j API with sample data"""
    client = Neo4jAPIClient()
    
    try:
        # Health check
        print("🔍 Checking API health...")
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"✅ Neo4j Connected: {health['neo4j_connected']}")
        
        # Load sample data
        with open('test_payload.json', 'r') as f:
            payload = json.load(f)
        
        # Store file structure
        print("\n📁 Storing file structure...")
        result = client.store_file_structure(payload['data'], payload.get('metadata', {}))
        print(f"✅ Stored with root ID: {result['root_id']}")
        
        # Get stats
        print("\n📊 Getting graph statistics...")
        stats = client.get_graph_stats()
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Files: {stats['total_files']}")
        print(f"Total Directories: {stats['total_directories']}")
        print(f"Sources: {stats['sources']}")
        
        # Search files
        print("\n🔍 Searching for files...")
        search_results = client.search_files("Enterprise")
        print(f"Found {search_results['count']} files matching 'Enterprise'")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()
