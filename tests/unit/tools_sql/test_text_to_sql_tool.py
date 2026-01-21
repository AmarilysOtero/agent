"""Unit tests for Text-to-SQL tool"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

try:
    from src.news_reporter.tools_sql.text_to_sql_tool import TextToSQLTool, query_database
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))
    from src.news_reporter.tools_sql.text_to_sql_tool import TextToSQLTool, query_database


class TestTextToSQLTool:
    """Test cases for TextToSQLTool class"""
    
    def test_init_default_backend_url(self):
        """Test initialization with default backend URL"""
        with patch('src.news_reporter.tools_sql.text_to_sql_tool.Settings') as mock_settings:
            mock_settings.load.return_value.neo4j_api_url = None
            with patch.dict('os.environ', {}, clear=True):
                tool = TextToSQLTool()
                assert tool.backend_url == "http://localhost:8000"
    
    def test_init_custom_backend_url(self):
        """Test initialization with custom backend URL"""
        tool = TextToSQLTool(backend_url="http://custom-backend:9000")
        assert tool.backend_url == "http://custom-backend:9000"
    
    def test_init_from_env_var(self):
        """Test initialization from environment variable"""
        with patch.dict('os.environ', {'NEO4J_BACKEND_URL': 'http://env-backend:8000'}):
            with patch('src.news_reporter.tools_sql.text_to_sql_tool.Settings') as mock_settings:
                mock_settings.load.return_value.neo4j_api_url = None
                tool = TextToSQLTool()
                assert tool.backend_url == "http://env-backend:8000"
    
    @patch('src.news_reporter.tools_sql.text_to_sql_tool.requests')
    def test_query_database_success(self, mock_requests):
        """Test successful SQL generation and execution"""
        # Mock SQL generation
        mock_sql_generator = Mock()
        mock_sql_generator.generate_sql.return_value = {
            "sql": 'SELECT "name" FROM "Employee"',
            "explanation": "This query retrieves all names from the Employee table",
            "confidence": 0.9,
            "error": None
        }
        
        # Mock backend API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "query_type": "SELECT",
            "data": {
                "columns": ["name"],
                "rows": [{"name": "Kevin"}, {"name": "Anthony"}],
                "row_count": 2
            },
            "error": None
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        # Create tool and test
        tool = TextToSQLTool(backend_url="http://test-backend:8000")
        tool.sql_generator = mock_sql_generator
        
        result = tool.query_database("list the names", "db-123")
        
        assert result["success"] is True
        assert result["generated_sql"] == 'SELECT "name" FROM "Employee"'
        assert result["explanation"] == "This query retrieves all names from the Employee table"
        assert result["confidence"] == 0.9
        assert result["results"]["row_count"] == 2
        assert len(result["results"]["rows"]) == 2
    
    @patch('src.news_reporter.tools_sql.text_to_sql_tool.requests')
    def test_query_database_sql_generation_failure(self, mock_requests):
        """Test handling of SQL generation failure"""
        # Mock SQL generation failure
        mock_sql_generator = Mock()
        mock_sql_generator.generate_sql.return_value = {
            "sql": None,
            "error": "No relevant schema elements found",
            "confidence": 0.0
        }
        
        tool = TextToSQLTool(backend_url="http://test-backend:8000")
        tool.sql_generator = mock_sql_generator
        
        result = tool.query_database("invalid query", "db-123")
        
        assert result["success"] is False
        assert result["generated_sql"] is None
        assert "No relevant schema elements found" in result["error"]
        assert result["results"] is None
    
    @patch('src.news_reporter.tools_sql.text_to_sql_tool.requests')
    def test_query_database_execution_failure(self, mock_requests):
        """Test handling of SQL execution failure"""
        # Mock SQL generation success
        mock_sql_generator = Mock()
        mock_sql_generator.generate_sql.return_value = {
            "sql": 'SELECT "name" FROM "Employee"',
            "explanation": "Test query",
            "confidence": 0.9,
            "error": None
        }
        
        # Mock backend API error
        mock_requests.post.side_effect = Exception("Connection error")
        
        tool = TextToSQLTool(backend_url="http://test-backend:8000")
        tool.sql_generator = mock_sql_generator
        
        result = tool.query_database("list the names", "db-123")
        
        assert result["success"] is False
        assert result["generated_sql"] == 'SELECT "name" FROM "Employee"'
        assert result["error"] is not None
    
    @patch('src.news_reporter.tools_sql.text_to_sql_tool.requests')
    def test_execute_sql_timeout(self, mock_requests):
        """Test handling of backend API timeout"""
        import requests
        mock_requests.post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        tool = TextToSQLTool(backend_url="http://test-backend:8000")
        result = tool._execute_sql("db-123", "SELECT * FROM table")
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()
    
    @patch('src.news_reporter.tools_sql.text_to_sql_tool.requests')
    def test_execute_sql_connection_error(self, mock_requests):
        """Test handling of connection error"""
        import requests
        mock_requests.post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        tool = TextToSQLTool(backend_url="http://test-backend:8000")
        result = tool._execute_sql("db-123", "SELECT * FROM table")
        
        assert result["success"] is False
        assert "connect" in result["error"].lower()
    
    @patch('src.news_reporter.tools_sql.text_to_sql_tool.requests')
    def test_execute_sql_http_error(self, mock_requests):
        """Test handling of HTTP error"""
        import requests
        mock_response = Mock()
        mock_response.json.return_value = {"detail": "Database not found"}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_requests.post.return_value = mock_response
        
        tool = TextToSQLTool(backend_url="http://test-backend:8000")
        result = tool._execute_sql("db-123", "SELECT * FROM table")
        
        assert result["success"] is False
        assert "error" in result["error"].lower()


class TestQueryDatabaseFunction:
    """Test cases for query_database Foundry tool function"""
    
    @patch('src.news_reporter.tools_sql.text_to_sql_tool.TextToSQLTool')
    def test_query_database_function_success(self, mock_tool_class):
        """Test query_database function returns JSON string"""
        # Mock tool instance
        mock_tool = Mock()
        mock_tool.query_database.return_value = {
            "success": True,
            "generated_sql": 'SELECT "name" FROM "Employee"',
            "explanation": "Test query",
            "confidence": 0.9,
            "results": {"row_count": 2},
            "error": None
        }
        mock_tool_class.return_value = mock_tool
        
        result = query_database("list the names", "db-123")
        
        # Should return JSON string
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["generated_sql"] == 'SELECT "name" FROM "Employee"'
    
    @patch('src.news_reporter.tools_sql.text_to_sql_tool.TextToSQLTool')
    def test_query_database_function_error(self, mock_tool_class):
        """Test query_database function handles errors"""
        # Mock tool instance with error
        mock_tool = Mock()
        mock_tool.query_database.return_value = {
            "success": False,
            "generated_sql": None,
            "explanation": None,
            "confidence": 0.0,
            "results": None,
            "error": "Test error"
        }
        mock_tool_class.return_value = mock_tool
        
        result = query_database("invalid query", "db-123")
        
        # Should return JSON string with error
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "error" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])














