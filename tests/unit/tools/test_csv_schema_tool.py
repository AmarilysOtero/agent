"""Unit tests for CSV/Excel Schema tool"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

try:
    from src.news_reporter.tools.csv_schema_tool import get_file_schema
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))
    from src.news_reporter.tools.csv_schema_tool import get_file_schema


class TestGetFileSchema:
    """Test cases for get_file_schema Foundry tool function"""
    
    @patch('src.news_reporter.tools.csv_schema_tool.CSVQueryTool')
    def test_get_file_schema_success_csv(self, mock_csv_tool_class):
        """Test successful schema retrieval for CSV file"""
        # Mock CSVQueryTool instance
        mock_tool = Mock()
        mock_tool.get_column_info.return_value = {
            "columns": {
                "id": {"dtype": "int64", "sample_values": [1, 2, 3]},
                "name": {"dtype": "object", "sample_values": ["John", "Jane"]},
                "email": {"dtype": "object", "sample_values": ["john@example.com", "jane@example.com"]}
            },
            "file_path": "test.csv"
        }
        mock_csv_tool_class.return_value = mock_tool
        
        result = get_file_schema("test.csv")
        
        # Should return JSON string
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "columns" in parsed
        assert "id" in parsed["columns"]
        assert "name" in parsed["columns"]
        assert "email" in parsed["columns"]
        assert parsed["file_path"] == "test.csv"
    
    @patch('src.news_reporter.tools.csv_schema_tool.CSVQueryTool')
    def test_get_file_schema_success_excel(self, mock_csv_tool_class):
        """Test successful schema retrieval for Excel file"""
        # Mock CSVQueryTool instance
        mock_tool = Mock()
        mock_tool.get_column_info.return_value = {
            "columns": {
                "product_id": {"dtype": "int64", "sample_values": [1001, 1002]},
                "product_name": {"dtype": "object", "sample_values": ["Widget A", "Widget B"]},
                "quantity": {"dtype": "int64", "sample_values": [50, 75]},
                "price": {"dtype": "float64", "sample_values": [9.99, 19.99]}
            },
            "file_path": "inventory.xlsx"
        }
        mock_csv_tool_class.return_value = mock_tool
        
        result = get_file_schema("inventory.xlsx")
        
        # Should return JSON string
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "columns" in parsed
        assert "product_id" in parsed["columns"]
        assert "product_name" in parsed["columns"]
        assert parsed["file_path"] == "inventory.xlsx"
    
    @patch('src.news_reporter.tools.csv_schema_tool.CSVQueryTool')
    def test_get_file_schema_file_not_found(self, mock_csv_tool_class):
        """Test handling of file not found error"""
        # Mock CSVQueryTool instance with error
        mock_tool = Mock()
        mock_tool.get_column_info.side_effect = FileNotFoundError("File not found: missing.csv")
        mock_csv_tool_class.return_value = mock_tool
        
        result = get_file_schema("missing.csv")
        
        # Should return JSON string with error
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "columns" in parsed
        assert parsed["columns"] == {}
        assert "error" in parsed
        assert "File not found" in parsed["error"]
    
    @patch('src.news_reporter.tools.csv_schema_tool.CSVQueryTool')
    def test_get_file_schema_generic_error(self, mock_csv_tool_class):
        """Test handling of generic errors"""
        # Mock CSVQueryTool instance with generic error
        mock_tool = Mock()
        mock_tool.get_column_info.side_effect = Exception("Unexpected error occurred")
        mock_csv_tool_class.return_value = mock_tool
        
        result = get_file_schema("corrupted.csv")
        
        # Should return JSON string with error
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "columns" in parsed
        assert "error" in parsed
        assert "Unexpected error" in parsed["error"]
    
    @patch('src.news_reporter.tools.csv_schema_tool.CSVQueryTool')
    def test_get_file_schema_empty_file(self, mock_csv_tool_class):
        """Test handling of empty file"""
        # Mock CSVQueryTool instance with empty result
        mock_tool = Mock()
        mock_tool.get_column_info.return_value = {
            "columns": {},
            "file_path": "empty.csv"
        }
        mock_csv_tool_class.return_value = mock_tool
        
        result = get_file_schema("empty.csv")
        
        # Should return JSON string with empty columns
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "columns" in parsed
        assert parsed["columns"] == {}
        assert parsed["file_path"] == "empty.csv"
    
    @patch('src.news_reporter.tools.csv_schema_tool.CSVQueryTool')
    def test_get_file_schema_json_format(self, mock_csv_tool_class):
        """Test that result is valid JSON"""
        # Mock CSVQueryTool instance
        mock_tool = Mock()
        mock_tool.get_column_info.return_value = {
            "columns": {
                "id": {"dtype": "int64", "sample_values": [1, 2]}
            },
            "file_path": "test.csv"
        }
        mock_csv_tool_class.return_value = mock_tool
        
        result = get_file_schema("test.csv")
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "columns" in parsed
        assert "file_path" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])









