"""Integration tests for CSV/Excel Schema tool

These tests require:
- Neo4j backend API running
- Test CSV/Excel files available
"""

import pytest
import os
import json
from pathlib import Path

try:
    from src.news_reporter.tools.csv_schema_tool import get_file_schema
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root))
    from src.news_reporter.tools.csv_schema_tool import get_file_schema


# Skip integration tests if environment variable is not set
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true",
    reason="Integration tests skipped (set SKIP_INTEGRATION_TESTS=false to run)"
)


class TestCSVSchemaIntegration:
    """Integration tests for CSV/Excel schema tool"""
    
    @pytest.fixture
    def test_csv_file(self, tmp_path):
        """Create a test CSV file"""
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("id,name,email\n1,John Doe,john@example.com\n2,Jane Smith,jane@example.com\n")
        return str(csv_file)
    
    def test_get_file_schema_csv_integration(self, test_csv_file):
        """Test schema retrieval for CSV file"""
        result_str = get_file_schema(test_csv_file)
        
        # Verify result is JSON string
        assert isinstance(result_str, str)
        
        # Parse and verify structure
        result = json.loads(result_str)
        assert isinstance(result, dict)
        assert "columns" in result
        assert "file_path" in result
        
        # Verify columns were extracted
        if "error" not in result or result["error"] is None:
            assert len(result["columns"]) > 0
            # Should have id, name, email columns
            assert "id" in result["columns"] or "name" in result["columns"]
    
    def test_get_file_schema_file_not_found(self):
        """Test error handling for non-existent file"""
        result_str = get_file_schema("nonexistent_file_12345.csv")
        
        # Should return JSON with error
        assert isinstance(result_str, str)
        result = json.loads(result_str)
        assert "error" in result
        assert result["error"] is not None


class TestFoundryToolIntegration:
    """Test Foundry tool function integration"""
    
    @pytest.fixture
    def test_csv_file(self, tmp_path):
        """Create a test CSV file"""
        csv_file = tmp_path / "test_inventory.csv"
        csv_file.write_text("product_id,product_name,quantity,price\n1001,Widget A,50,9.99\n1002,Widget B,75,19.99\n")
        return str(csv_file)
    
    def test_foundry_tool_format(self, test_csv_file):
        """Test that result format is compatible with Foundry"""
        result_str = get_file_schema(test_csv_file)
        
        # Should be valid JSON
        result = json.loads(result_str)
        
        # Should have required fields for Foundry
        assert isinstance(result, dict)
        assert "columns" in result
        assert "file_path" in result
        
        # Columns should be a dictionary
        assert isinstance(result["columns"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])









