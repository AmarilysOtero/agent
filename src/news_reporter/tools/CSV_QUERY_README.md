# CSV Query Tool

The CSV Query Tool provides exact numerical calculations on CSV files by calling the Neo4j backend's pandas query service. It complements RAG chunks which provide semantic search and context.

## Problem Solved

**Before:** When asking "How many 4Runner TRD Pro are there?", the system would:
- Count the number of RAG chunks (e.g., 12 rows)
- Give incorrect answers because it's counting chunks, not summing inventory values

**After:** The CSV Query Tool:
- Filters rows where Model = "4Runner TRD Pro"
- Sums ALL date columns (all inventory counts across all time periods)
- Returns the exact total inventory count

## Usage

### Generic Example: Sum Numeric Columns (Works with Any CSV)

```python
from src.news_reporter.tools.csv_query import sum_numeric_columns

# Works with any CSV file structure
result = sum_numeric_columns(
    file_path="any_file.csv",
    filters={'Product': 'Widget A', 'Region': 'North'},
    exclude_columns=['Product', 'Region', 'Date']  # Exclude non-numeric columns
)

print(f"Total: {result['total']}")
print(f"Breakdown: {result['breakdown']}")
```

### Inventory-Specific Example: Get Total Inventory

```python
from src.news_reporter.tools.csv_query import get_total_inventory_for_model

file_path = r"C:\path\to\inventory.csv"
model = "4Runner TRD Pro"

result = get_total_inventory_for_model(file_path, model)

print(f"Total inventory: {result['total']}")
print(f"Breakdown: {result['breakdown']}")
```

### Advanced Query with Filters and Aggregations

```python
from src.news_reporter.tools.csv_query import csv_query

result = csv_query(
    file_path=file_path,
    filters={'Model': '4Runner TRD Pro'},
    aggregations={
        '2025-04-01 to 2025-04-07': ['sum'],
        '2025-04-08 to 2025-04-14': ['sum'],
    }
)

print(f"Results: {result['data']}")
```

### Integration with RAG Search (Generic)

```python
from src.news_reporter.tools.neo4j_graphrag import graphrag_search
from src.news_reporter.tools.csv_query import (
    query_requires_exact_numbers,
    extract_csv_path_from_rag_results,
    extract_filter_value_from_query,
    sum_numeric_columns
)

query = "How many 4Runner TRD Pro are there?"

# Step 1: Check if query needs exact numbers
if query_requires_exact_numbers(query):
    # Step 2: Get RAG results to find CSV file path
    rag_results = graphrag_search(query, top_k=5)
    
    # Step 3: Extract CSV file path from RAG results
    csv_path = extract_csv_path_from_rag_results(rag_results)
    
    if csv_path:
        # Step 4: Extract filter value from query or RAG results
        # Generic: works with any column name
        filter_value = extract_filter_value_from_query(query, 'Model', rag_results)
        # Or for other columns: extract_filter_value_from_query(query, 'Product', rag_results)
        
        if filter_value:
            # Step 5: Get exact numerical answer (generic approach)
            exact_result = sum_numeric_columns(
                file_path=csv_path,
                filters={'Model': filter_value},
                exclude_columns=['Model', 'Factory Location']  # Exclude non-numeric
            )
            total = exact_result['total']
            
            # Step 6: Use RAG results for context/explanation
            context = format_rag_chunks(rag_results)
            
            answer = f"Total {filter_value}: {total}\n\n{context}"
        else:
            answer = "Could not extract filter value from query"
    else:
        answer = "No CSV file found in search results"
else:
    # Use regular RAG search
    rag_results = graphrag_search(query)
    answer = format_rag_results(rag_results)
```

## API Endpoints

The tool calls these Neo4j backend endpoints:

- `POST /api/v1/csv/query` - Execute pandas queries
- `POST /api/v1/csv/columns` - Get column information
- `POST /api/v1/csv/cache/clear` - Clear CSV cache

## Helper Functions

### `query_requires_exact_numbers(query: str) -> bool`
Detects if a query requires exact numerical calculation (e.g., "how many", "total", "sum")

### `extract_csv_path_from_rag_results(rag_results: List[Dict]) -> Optional[str]`
Extracts CSV file path from RAG search results

### `extract_filter_value_from_query(query: str, filter_column: str, rag_results: Optional[List[Dict]]) -> Optional[str]`
**Generic function** - Extracts filter value for any column from query or RAG results

### `extract_model_from_query(query: str, rag_results: Optional[List[Dict]]) -> Optional[str]`
Convenience wrapper for `extract_filter_value_from_query(query, 'Model', rag_results)`

### `detect_numeric_columns(file_path: str, exclude_columns: Optional[List[str]]) -> List[str]`
Auto-detects numeric columns in a CSV file (checks dtypes and sample values)

### `detect_date_columns(file_path: str, exclude_columns: Optional[List[str]]) -> List[str]`
Auto-detects date/time columns in a CSV file (looks for date patterns)

## Configuration

The tool uses the `NEO4J_API_URL` environment variable from `.env`:

```env
NEO4J_API_URL=http://localhost:8000
```

## Example Output

```python
{
    'total': 12345,  # Sum of all date columns
    'breakdown': {
        '2025-01-01 to 2025-01-07_sum': 100,
        '2025-01-08 to 2025-01-14_sum': 150,
        # ... all date columns
    },
    'metadata': {
        'row_count': 1,
        'columns': [...],
        'file_path': '...'
    },
    'date_columns_summed': 60
}
```

## See Also

- `csv_query_example.py` - Complete usage examples
- Neo4j Backend: `services/csv_query_service.py` - Backend implementation
- Neo4j Backend: `api/v1/csv/query.py` - API endpoints

