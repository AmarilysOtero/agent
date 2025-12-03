"""Example usage of CSV Query Tool

This demonstrates how to use the CSV query tool to get exact numerical answers
for questions like "How many 4Runner TRD Pro are there?"

The tool is generic and works with any CSV file structure.
"""

from csv_query import (
    CSVQueryTool, 
    get_total_inventory_for_model, 
    csv_query,
    sum_numeric_columns
)


def example_get_total_inventory():
    """Example: Get total inventory count for a model (inventory-specific)"""
    
    file_path = r"C:\Alexis\DXC\AI\hackathon\Syntethic Data\Inventory\inventory 2025.csv"
    model = "4Runner TRD Pro"
    
    # Method 1: Use convenience function (for inventory CSVs with 'Model' column)
    result = get_total_inventory_for_model(file_path, model)
    
    print(f"Total {model} inventory: {result['total']}")
    print(f"Columns summed: {result.get('columns_summed', 0)}")
    if 'breakdown' in result:
        print(f"Sample breakdown: {list(result['breakdown'].items())[:3]}")
    
    return result


def example_generic_sum_numeric():
    """Example: Generic sum of numeric columns (works with any CSV)"""
    
    file_path = r"C:\Alexis\DXC\AI\hackathon\Syntethic Data\Inventory\inventory 2025.csv"
    
    # Generic approach: sum all numeric columns with a filter
    result = sum_numeric_columns(
        file_path=file_path,
        filters={'Model': '4Runner TRD Pro'},
        exclude_columns=['Factory Location', 'Model']  # Exclude non-numeric columns
    )
    
    print(f"Total (generic): {result['total']}")
    print(f"Columns summed: {result.get('columns_summed', 0)}")
    
    return result


def example_custom_csv():
    """Example: Using with a different CSV structure (sales data)"""
    
    # This would work with any CSV file
    file_path = "sales_data.csv"  # Example path
    
    # Sum numeric columns with multiple filters
    result = sum_numeric_columns(
        file_path=file_path,
        filters={'Product': 'Widget A', 'Region': 'North'},
        exclude_columns=['Product', 'Region', 'Date']  # Exclude non-numeric columns
    )
    
    print(f"Total sales: {result['total']}")
    return result


def example_query_with_filters():
    """Example: Query CSV with filters and aggregations"""
    
    file_path = r"C:\Alexis\DXC\AI\hackathon\Syntethic Data\Inventory\inventory 2025.csv"
    
    # Query: Get sum of specific date columns for 4Runner TRD Pro
    result = csv_query(
        file_path=file_path,
        filters={'Model': '4Runner TRD Pro'},
        aggregations={
            '2025-02-01 to 2025-02-07': ['sum'],
            '2025-02-08 to 2025-02-14': ['sum'],
            '2025-04-01 to 2025-04-07': ['sum'],
            '2025-04-08 to 2025-04-14': ['sum'],
        }
    )
    
    print(f"Query results: {len(result.get('data', []))} rows")
    if result.get('data'):
        print(f"First row: {result['data'][0]}")
    
    return result


def example_get_column_info():
    """Example: Get information about CSV columns"""
    
    file_path = r"C:\Alexis\DXC\AI\hackathon\Syntethic Data\Inventory\inventory 2025.csv"
    
    tool = CSVQueryTool()
    col_info = tool.get_column_info(file_path)
    
    print(f"Total columns: {col_info.get('total_columns', 0)}")
    print(f"Total rows: {col_info.get('total_rows', 0)}")
    print(f"\nColumn names (first 10):")
    for i, col_name in enumerate(list(col_info.get('columns', {}).keys())[:10]):
        print(f"  {i+1}. {col_name}")
    
    return col_info


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Get Total Inventory for Model (Inventory-specific)")
    print("=" * 60)
    example_get_total_inventory()
    
    print("\n" + "=" * 60)
    print("Example 2: Generic Sum of Numeric Columns")
    print("=" * 60)
    example_generic_sum_numeric()
    
    print("\n" + "=" * 60)
    print("Example 3: Query with Specific Aggregations")
    print("=" * 60)
    example_query_with_filters()
    
    print("\n" + "=" * 60)
    print("Example 4: Get Column Information")
    print("=" * 60)
    example_get_column_info()
    
    print("\n" + "=" * 60)
    print("Example 5: Custom CSV Structure (Sales Data)")
    print("=" * 60)
    print("(Commented out - uncomment and provide file path to test)")
    # example_custom_csv()

