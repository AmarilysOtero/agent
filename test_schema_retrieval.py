"""Test script to verify schema retrieval from Neo4j backend"""

import sys
import os
import requests
import json
from typing import Dict, Any, List

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_backend_health(neo4j_api_url: str) -> bool:
    """Test if the backend is reachable"""
    print("\n" + "="*80)
    print("TEST 1: Backend Health Check")
    print("="*80)
    
    health_url = f"{neo4j_api_url}/health"
    print(f"Testing: {health_url}")
    
    try:
        response = requests.get(health_url, timeout=5.0)
        print(f"✅ Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ Timeout after 5 seconds")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_list_databases(neo4j_api_url: str) -> List[Dict[str, Any]]:
    """Test listing all databases"""
    print("\n" + "="*80)
    print("TEST 2: List All Databases")
    print("="*80)
    
    url = f"{neo4j_api_url}/api/databases"
    print(f"Testing: {url}")
    
    try:
        print("Sending GET request (timeout: 60s)...")
        response = requests.get(url, timeout=60.0)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data)} database(s)")
            
            if isinstance(data, list):
                for idx, db in enumerate(data, 1):
                    print(f"\n   Database {idx}:")
                    print(f"      ID: {db.get('id', 'N/A')}")
                    print(f"      Name: {db.get('name', 'N/A')}")
                    print(f"      Type: {db.get('database_type', db.get('databaseType', 'N/A'))}")
                    print(f"      Host: {db.get('host', 'N/A')}")
                    print(f"      Port: {db.get('port', 'N/A')}")
                    print(f"      Database Name: {db.get('database_name', db.get('databaseName', 'N/A'))}")
                return data
            else:
                print(f"⚠️  Unexpected response format: {type(data)}")
                print(f"Response: {json.dumps(data, indent=2)}")
                return []
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except requests.exceptions.Timeout:
        print(f"❌ Timeout after 60 seconds")
        return []
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_schema_search(neo4j_api_url: str, database_id: str, query: str = "names") -> Dict[str, Any]:
    """Test schema search for a specific database"""
    print("\n" + "="*80)
    print(f"TEST 3: Schema Search (Database: {database_id}, Query: '{query}')")
    print("="*80)
    
    url = f"{neo4j_api_url}/api/databases/{database_id}/schema/search"
    print(f"Testing: {url}")
    
    payload = {
        "query": query,
        "top_k": 10,
        "similarity_threshold": 0.7,
        "element_types": None,
        "use_keyword_search": True,
        "use_graph_expansion": True,
        "max_hops": 1
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        print("Sending POST request (timeout: 60s)...")
        response = requests.post(url, json=payload, timeout=60.0)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            result_count = data.get("result_count", 0)
            print(f"✅ Found {result_count} schema element(s)")
            
            schema_slice = data.get("schema_slice", {})
            tables = schema_slice.get("tables", [])
            print(f"✅ Found {len(tables)} table(s) in schema_slice")
            
            if tables:
                print("\n   Tables found:")
                for idx, table in enumerate(tables, 1):
                    table_name = table.get("name", "?")
                    columns = table.get("columns", [])
                    print(f"\n   Table {idx}: {table_name}")
                    print(f"      Description: {table.get('description', 'N/A')}")
                    print(f"      Domain: {table.get('domain', 'N/A')}")
                    print(f"      Columns ({len(columns)}):")
                    for col_idx, col in enumerate(columns[:10], 1):  # Show first 10 columns
                        col_name = col.get("name", "?")
                        col_type = col.get("data_type", "?")
                        print(f"         {col_idx}. {col_name} ({col_type})")
                    if len(columns) > 10:
                        print(f"         ... and {len(columns) - 10} more columns")
            else:
                print("⚠️  No tables found in schema_slice")
                if "results" in data:
                    results = data.get("results", [])
                    print(f"   But found {len(results)} result(s) in 'results' field")
                    if results:
                        print(f"   First result: {json.dumps(results[0], indent=2)}")
            
            return data
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except requests.exceptions.Timeout:
        print(f"❌ Timeout after 60 seconds")
        return {}
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_get_database_config(neo4j_api_url: str, database_id: str) -> Dict[str, Any]:
    """Test getting a specific database configuration"""
    print("\n" + "="*80)
    print(f"TEST 4: Get Database Config (ID: {database_id})")
    print("="*80)
    
    url = f"{neo4j_api_url}/api/databases/{database_id}"
    print(f"Testing: {url}")
    
    try:
        print("Sending GET request (timeout: 30s)...")
        response = requests.get(url, timeout=30.0)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Database Configuration:")
            print(f"   ID: {data.get('id', 'N/A')}")
            print(f"   Name: {data.get('name', 'N/A')}")
            print(f"   Type: {data.get('database_type', 'N/A')}")
            print(f"   Host: {data.get('host', 'N/A')}")
            print(f"   Port: {data.get('port', 'N/A')}")
            print(f"   Database Name: {data.get('database_name', 'N/A')}")
            return data
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except requests.exceptions.Timeout:
        print(f"❌ Timeout after 30 seconds")
        return {}
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_supabase_connection(neo4j_api_url: str, database_id: str) -> Dict[str, Any]:
    """Test direct connection to Supabase and execute a query"""
    print("\n" + "="*80)
    print(f"TEST 5: Supabase Connection Test (Database: {database_id})")
    print("="*80)
    
    # First get the database config to get connection details
    config_url = f"{neo4j_api_url}/api/databases/{database_id}"
    print(f"Getting database config from: {config_url}")
    
    try:
        response = requests.get(config_url, timeout=30.0)
        if response.status_code != 200:
            print(f"❌ Failed to get database config: {response.status_code}")
            return {}
        
        db_config = response.json()
        host = db_config.get("host", "")
        port = db_config.get("port", 5432)
        database_name = db_config.get("database_name", "")
        username = db_config.get("username", "")
        
        print(f"✅ Database Config Retrieved:")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Database: {database_name}")
        print(f"   Username: {username}")
        print(f"   Password: {'*' * 10} (hidden)")
        
        # Test connection via backend API
        test_url = f"{neo4j_api_url}/api/databases/{database_id}/test"
        print(f"\nTesting connection via backend API: {test_url}")
        print("Sending POST request (timeout: 30s)...")
        
        test_response = requests.post(test_url, params={"use_env": "false"}, timeout=30.0)
        print(f"✅ Status Code: {test_response.status_code}")
        
        if test_response.status_code == 200:
            test_result = test_response.json()
            success = test_result.get("success", False)
            error = test_result.get("error")
            
            if success:
                print(f"✅ Connection test successful!")
                print(f"   Query Type: {test_result.get('query_type', 'N/A')}")
                return {"success": True, "result": test_result}
            else:
                print(f"❌ Connection test failed: {error}")
                return {"success": False, "error": error}
        else:
            print(f"❌ Unexpected status code: {test_response.status_code}")
            print(f"Response: {test_response.text}")
            return {"success": False, "error": f"HTTP {test_response.status_code}"}
            
    except requests.exceptions.Timeout:
        print(f"❌ Timeout after 30 seconds")
        return {"success": False, "error": "Timeout"}
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def test_sql_execution(neo4j_api_url: str, database_id: str, sql_query: str = 'SELECT "name" FROM "Employee" LIMIT 5') -> Dict[str, Any]:
    """Test executing a SQL query on Supabase"""
    print("\n" + "="*80)
    print(f"TEST 6: SQL Execution Test (Database: {database_id})")
    print("="*80)
    
    url = f"{neo4j_api_url}/api/databases/{database_id}/execute"
    print(f"Testing: {url}")
    print(f"SQL Query: {sql_query}")
    
    payload = {"query": sql_query}
    
    try:
        print("Sending POST request (timeout: 30s)...")
        response = requests.post(url, json=payload, timeout=30.0)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            success = data.get("success", False)
            error = data.get("error")
            
            if success:
                result_data = data.get("data", {})
                row_count = result_data.get("row_count", 0)
                rows = result_data.get("rows", [])
                columns = result_data.get("columns", [])
                
                print(f"✅ SQL execution successful!")
                print(f"   Query Type: {data.get('query_type', 'N/A')}")
                print(f"   Rows Returned: {row_count}")
                print(f"   Columns: {columns}")
                
                if rows:
                    print(f"\n   Results:")
                    for idx, row in enumerate(rows[:10], 1):  # Show first 10 rows
                        row_str = " | ".join([f"{col}: {row.get(col, 'N/A')}" for col in columns])
                        print(f"      {idx}. {row_str}")
                    if len(rows) > 10:
                        print(f"      ... and {len(rows) - 10} more rows")
                
                return {"success": True, "data": result_data}
            else:
                print(f"❌ SQL execution failed: {error}")
                return {"success": False, "error": error}
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        print(f"❌ Timeout after 30 seconds")
        return {"success": False, "error": "Timeout"}
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SCHEMA RETRIEVAL TEST SUITE")
    print("="*80)
    
    # Get Neo4j API URL from environment
    neo4j_api_url = os.getenv("NEO4J_API_URL", "http://localhost:8000")
    
    # Check if we should use Docker URL
    # If NEO4J_API_URL contains 'neo4j-backend' or we're in Docker, use that
    if "neo4j-backend" in neo4j_api_url:
        print(f"Using Docker service URL: {neo4j_api_url}")
    elif os.getenv("DOCKER_ENV") or os.path.exists("/.dockerenv"):
        # Running inside Docker - use service name
        neo4j_api_url = "http://neo4j-backend:8000"
        print(f"Detected Docker environment - using service URL: {neo4j_api_url}")
    else:
        # Try localhost first, but also try Docker service name if localhost fails
        print(f"Using Neo4j API URL: {neo4j_api_url}")
        print("Note: If this fails, the backend might be running in Docker.")
        print("      Try setting NEO4J_API_URL=http://neo4j-backend:8000 or http://localhost:8000")
    
    # Remove trailing slash
    neo4j_api_url = neo4j_api_url.rstrip("/")
    
    # Test 1: Health check - try multiple URLs if needed
    urls_to_try = [neo4j_api_url]
    
    # If using localhost, also try Docker service name
    if "localhost" in neo4j_api_url:
        urls_to_try.append("http://neo4j-backend:8000")
    # If using Docker service name, also try localhost
    elif "neo4j-backend" in neo4j_api_url:
        urls_to_try.append("http://localhost:8000")
    
    backend_reachable = False
    working_url = None
    
    for url in urls_to_try:
        print(f"\nTrying URL: {url}")
        if test_backend_health(url):
            backend_reachable = True
            working_url = url
            neo4j_api_url = url  # Use the working URL for subsequent tests
            break
    
    if not backend_reachable:
        print("\n❌ Backend is not reachable at any of the tried URLs. Please check:")
        print("   1. Is the neo4j-backend service running?")
        print("   2. Is the URL correct?")
        print("   3. Are there any network/firewall issues?")
        print("   4. If running locally, is the backend accessible at http://localhost:8000?")
        print("   5. If running in Docker, is the service name 'neo4j-backend' correct?")
        return
    
    # Test 2: List databases
    databases = test_list_databases(neo4j_api_url)
    
    if not databases:
        print("\n❌ No databases found. Please check:")
        print("   1. Are there any database configurations in Neo4j?")
        print("   2. Is the Neo4j connection working?")
        print("   3. Check the backend logs for errors")
        return
    
    # Test 3: Get database config for first database
    if databases:
        first_db = databases[0]
        db_id = first_db.get("id") or first_db.get("database_id") or first_db.get("_id")
        if db_id:
            test_get_database_config(neo4j_api_url, db_id)
    
    # Test 4: Schema search
    if databases:
        first_db = databases[0]
        db_id = first_db.get("id") or first_db.get("database_id") or first_db.get("_id")
        if db_id:
            # Test with "names" query
            schema_result = test_schema_search(neo4j_api_url, db_id, "names")
            
            if schema_result and schema_result.get("schema_slice", {}).get("tables"):
                print("\n✅ Schema retrieval is working!")
            else:
                print("\n⚠️  Schema retrieval returned no tables. This might indicate:")
                print("   1. No schema has been stored for this database")
                print("   2. The query doesn't match any schema elements")
                print("   3. The similarity threshold is too high")
    
    # Test 5: Supabase Connection Test
    if databases:
        first_db = databases[0]
        db_id = first_db.get("id") or first_db.get("database_id") or first_db.get("_id")
        db_type = first_db.get("database_type") or first_db.get("databaseType", "").lower()
        
        if db_id and ("postgresql" in db_type or "postgres" in db_type):
            print("\n" + "="*80)
            print("SUPABASE CONNECTION TESTS")
            print("="*80)
            
            # Test connection
            connection_result = test_supabase_connection(neo4j_api_url, db_id)
            
            # Test SQL execution if connection test passed
            if connection_result.get("success"):
                # Test with the query that was generated earlier
                sql_query = 'SELECT "name" FROM "Employee" LIMIT 5'
                execution_result = test_sql_execution(neo4j_api_url, db_id, sql_query)
                
                if execution_result.get("success"):
                    print("\n✅ Supabase connection and SQL execution are working!")
                else:
                    print("\n⚠️  Connection test passed but SQL execution failed:")
                    print(f"   Error: {execution_result.get('error', 'Unknown error')}")
            else:
                print("\n❌ Supabase connection test failed:")
                print(f"   Error: {connection_result.get('error', 'Unknown error')}")
                print("\n   Possible issues:")
                print("   1. Supabase server is not accessible from the backend container")
                print("   2. Network/firewall blocking the connection")
                print("   3. Incorrect credentials in Neo4j database config")
                print("   4. Supabase instance is down or paused")
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

