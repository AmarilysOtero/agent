"""Integration tests for Text-to-SQL tool

These tests require:
- Neo4j backend API running
- Valid database configuration in Neo4j
- Test database with sample data
"""

import pytest
import os
from typing import Dict, Any

try:
    from src.news_reporter.tools_sql.text_to_sql_tool import TextToSQLTool, query_database
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))
    from src.news_reporter.tools_sql.text_to_sql_tool import TextToSQLTool, query_database


# Skip integration tests if environment variable is not set
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true",
    reason="Integration tests skipped (set SKIP_INTEGRATION_TESTS=false to run)"
)


class TestTextToSQLIntegration:
    """Integration tests for TextToSQLTool"""
    
    @pytest.fixture
    def backend_url(self):
        """Get backend URL from environment or use default"""
        return os.getenv("NEO4J_BACKEND_URL", "http://localhost:8000")
    
    @pytest.fixture
    def database_id(self):
        """Get test database ID from environment"""
        db_id = os.getenv("TEST_DATABASE_ID")
        if not db_id:
            pytest.skip("TEST_DATABASE_ID environment variable not set")
        return db_id
    
    def test_query_database_integration(self, backend_url, database_id):
        """Test end-to-end query_database integration"""
        tool = TextToSQLTool(backend_url=backend_url)
        
        result = tool.query_database(
            natural_language_query="list the names",
            database_id=database_id
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "generated_sql" in result
        assert "explanation" in result
        assert "confidence" in result
        assert "results" in result
        assert "error" in result
        
        # If successful, verify SQL was generated
        if result["success"]:
            assert result["generated_sql"] is not None
            assert isinstance(result["generated_sql"], str)
            assert len(result["generated_sql"]) > 0
            assert result["confidence"] >= 0.0
            assert result["confidence"] <= 1.0
    
    def test_query_database_function_integration(self, backend_url, database_id):
        """Test query_database Foundry function integration"""
        import json
        
        result_str = query_database("list the names", database_id)
        
        # Verify result is JSON string
        assert isinstance(result_str, str)
        
        # Parse and verify structure
        result = json.loads(result_str)
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_error_handling_invalid_database(self, backend_url):
        """Test error handling for invalid database ID"""
        tool = TextToSQLTool(backend_url=backend_url)
        
        result = tool.query_database(
            natural_language_query="list the names",
            database_id="invalid-database-id-12345"
        )
        
        # Should handle error gracefully
        assert isinstance(result, dict)
        assert "success" in result
        # May succeed in SQL generation but fail in execution, or fail completely
        # Either way, error should be handled


class TestEndToEndFlow:
    """Test complete end-to-end flow: natural language → SQL → execution → results"""
    
    @pytest.fixture
    def backend_url(self):
        """Get backend URL from environment or use default"""
        return os.getenv("NEO4J_BACKEND_URL", "http://localhost:8000")
    
    @pytest.fixture
    def database_id(self):
        """Get test database ID from environment"""
        db_id = os.getenv("TEST_DATABASE_ID")
        if not db_id:
            pytest.skip("TEST_DATABASE_ID environment variable not set")
        return db_id
    
    def test_complete_flow_simple_query(self, backend_url, database_id):
        """Test complete flow with simple SELECT query"""
        tool = TextToSQLTool(backend_url=backend_url)
        
        # Step 1: Natural language query
        query = "list the names"
        
        # Step 2: Generate SQL and execute
        result = tool.query_database(query, database_id)
        
        # Step 3: Verify results
        if result["success"]:
            # SQL should be generated
            assert result["generated_sql"] is not None
            
            # Results should be available
            if result["results"]:
                assert "columns" in result["results"]
                assert "rows" in result["results"]
                assert "row_count" in result["results"]
        else:
            # If failed, error should be present
            assert result["error"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])














