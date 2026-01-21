"""CSV Query Tool for exact numerical queries using pandas

Provides exact numerical calculations on CSV files by calling the Neo4j backend's
pandas query service. Complements RAG chunks which provide semantic search and context.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import requests
import logging
import time
import os
import numpy as np

try:
    from ..config import Settings
    from .embeddings import EmbeddingsProvider
except ImportError:
    import sys, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.config import Settings
    from src.news_reporter.tools.embeddings import EmbeddingsProvider

logger = logging.getLogger(__name__)

# Global embeddings provider (lazy initialization)
_embeddings_provider: Optional[EmbeddingsProvider] = None

def _get_embeddings_provider() -> Optional[EmbeddingsProvider]:
    """Lazy initialization of embeddings provider."""
    global _embeddings_provider
    if _embeddings_provider is None:
        try:
            _embeddings_provider = EmbeddingsProvider()
        except Exception as e:
            logger.warning(f"Failed to initialize EmbeddingsProvider: {e}. Semantic detection will be disabled.")
            return None
    return _embeddings_provider


class CSVQueryTool:
    """Tool for querying CSV files with exact numerical calculations using pandas"""
    
    def __init__(self, neo4j_api_url: str = None):
        """
        Args:
            neo4j_api_url: Base URL for Neo4j backend API (e.g., "http://localhost:8000")
        """
        settings = Settings.load()
        self.neo4j_api_url = neo4j_api_url or settings.neo4j_api_url
        if not self.neo4j_api_url:
            raise ValueError(
                "NEO4J_API_URL must be set in .env or passed to constructor. "
                "Example: http://localhost:8000"
            )
        
        # If running in Docker and URL uses localhost, replace with host.docker.internal
        if os.getenv("DOCKER_ENV") and "localhost" in self.neo4j_api_url:
            self.neo4j_api_url = self.neo4j_api_url.replace("localhost", "host.docker.internal")
            logger.info(f"Running in Docker - updated Neo4j URL to use host.docker.internal")
        
        # Remove trailing slash if present
        self.neo4j_api_url = self.neo4j_api_url.rstrip("/")
        logger.info(f"CSV Query Tool initialized with URL: {self.neo4j_api_url}")
    
    def query_csv(
        self,
        file_path: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        aggregations: Optional[Dict[str, List[str]]] = None,
        group_by: Optional[List[str]] = None,
        order_by: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query CSV file for exact numerical results
        
        Args:
            file_path: Path to CSV file
            filters: Column filters (e.g., {'Model': '4Runner TRD Pro'})
            columns: Columns to return
            aggregations: Aggregations (e.g., {'2025-04-01 to 2025-04-07': ['sum', 'mean']})
            group_by: Columns to group by
            order_by: Sort columns (e.g., ['column ASC', 'column DESC'])
            limit: Maximum rows to return
            
        Returns:
            Dict with 'data' (list of records) and 'metadata' (row count, etc.)
        """
        try:
            url = f"{self.neo4j_api_url}/api/v1/csv/query"
            payload = {
                "file_path": file_path,
            }
            if filters:
                payload["filters"] = filters
            if columns:
                payload["columns"] = columns
            if aggregations:
                payload["aggregations"] = aggregations
            if group_by:
                payload["group_by"] = group_by
            if order_by:
                payload["order_by"] = order_by
            if limit:
                payload["limit"] = limit
            
            logger.info(
                f"Querying CSV: {file_path} "
                f"(filters={filters}, aggregations={aggregations}, group_by={group_by})"
            )
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60.0)
            elapsed = time.time() - start_time
            logger.info(f"CSV query request completed in {elapsed:.2f}s")
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CSV query API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return {'data': [], 'metadata': {'error': str(e), 'file_path': file_path}}
        except Exception as e:
            logger.error(f"CSV query error: {e}", exc_info=True)
            return {'data': [], 'metadata': {'error': str(e), 'file_path': file_path}}
    
    def get_column_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about CSV columns (types, sample values, etc.)
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dict with column information
        """
        try:
            url = f"{self.neo4j_api_url}/api/v1/csv/columns"
            logger.info(f"Getting column info for CSV: {file_path}")
            response = requests.post(
                url,
                json={"file_path": file_path},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CSV column info API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return {'columns': {}, 'error': str(e), 'file_path': file_path}
        except Exception as e:
            logger.error(f"CSV column info error: {e}", exc_info=True)
            return {'columns': {}, 'error': str(e), 'file_path': file_path}
    
    def get_distinct_values(
        self,
        file_path: str,
        column: str,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get distinct/unique values from a column
        
        Useful for queries like "list all models" or "name all products"
        
        Args:
            file_path: Path to CSV file
            column: Column name to get distinct values from
            filters: Optional filters to apply before getting distinct values
            order_by: Optional sorting
            
        Returns:
            Dict with 'values' (list), 'count', and 'metadata'
        """
        try:
            url = f"{self.neo4j_api_url}/api/v1/csv/distinct"
            payload = {
                "file_path": file_path,
                "column": column
            }
            if filters:
                payload["filters"] = filters
            if order_by:
                payload["order_by"] = order_by
            
            logger.info(f"Getting distinct values for CSV: {file_path}, column: {column}")
            response = requests.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CSV distinct values API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return {'values': [], 'count': 0, 'error': str(e), 'file_path': file_path}
        except Exception as e:
            logger.error(f"CSV distinct values error: {e}", exc_info=True)
            return {'values': [], 'count': 0, 'error': str(e), 'file_path': file_path}
    
    def detect_numeric_columns(
        self,
        file_path: str,
        exclude_columns: Optional[List[str]] = None
    ) -> List[str]:
        """
        Auto-detect numeric columns in a CSV file
        
        Args:
            file_path: Path to CSV file
            exclude_columns: Optional list of column names to exclude from detection
            
        Returns:
            List of column names that appear to be numeric
        """
        try:
            col_info = self.get_column_info(file_path)
            if 'error' in col_info:
                logger.warning(f"Could not get column info: {col_info.get('error')}")
                return []
            
            exclude_set = set(exclude_columns or [])
            numeric_columns = []
            
            for col_name, col_data in col_info.get('columns', {}).items():
                if col_name in exclude_set:
                    continue
                
                dtype = col_data.get('dtype', '')
                # Check if pandas detected it as numeric
                if 'int' in dtype or 'float' in dtype:
                    numeric_columns.append(col_name)
                # Also check sample values to detect numeric strings
                elif dtype == 'object':
                    sample_values = col_data.get('sample_values', [])
                    if sample_values:
                        # Check if samples are numeric
                        numeric_samples = 0
                        for val in sample_values[:5]:  # Check first 5 samples
                            try:
                                float(str(val).replace(',', '').replace('$', '').strip())
                                numeric_samples += 1
                            except (ValueError, AttributeError):
                                pass
                        # If most samples are numeric, consider it numeric
                        if numeric_samples >= len(sample_values) * 0.7:
                            numeric_columns.append(col_name)
            
            logger.info(f"Auto-detected {len(numeric_columns)} numeric columns")
            return numeric_columns
            
        except Exception as e:
            logger.error(f"Error detecting numeric columns: {e}", exc_info=True)
            return []
    
    def detect_date_columns(
        self,
        file_path: str,
        exclude_columns: Optional[List[str]] = None
    ) -> List[str]:
        """
        Auto-detect date/time columns in a CSV file
        
        Looks for columns with date-like patterns (years, date ranges, etc.)
        
        Args:
            file_path: Path to CSV file
            exclude_columns: Optional list of column names to exclude from detection
            
        Returns:
            List of column names that appear to be date-related
        """
        try:
            col_info = self.get_column_info(file_path)
            if 'error' in col_info:
                logger.warning(f"Could not get column info: {col_info.get('error')}")
                return []
            
            exclude_set = set(exclude_columns or [])
            date_columns = []
            
            import re
            # Patterns that suggest date columns
            date_patterns = [
                r'\d{4}',  # Years (2025, 2024, etc.)
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Date formats (MM/DD/YYYY)
                r'to',  # Date ranges (2025-01-01 to 2025-01-07)
                r'week|month|quarter|year',  # Time period keywords
                r'q\d',  # Quarters (Q1, Q2, etc.)
            ]
            
            for col_name, col_data in col_info.get('columns', {}).items():
                if col_name in exclude_set:
                    continue
                
                col_lower = col_name.lower()
                # Check if column name matches date patterns
                matches_pattern = any(re.search(pattern, col_lower) for pattern in date_patterns)
                
                if matches_pattern:
                    date_columns.append(col_name)
            
            logger.info(f"Auto-detected {len(date_columns)} date columns")
            return date_columns
            
        except Exception as e:
            logger.error(f"Error detecting date columns: {e}", exc_info=True)
            return []
    
    def sum_numeric_columns(
        self,
        file_path: str,
        filters: Optional[Dict[str, Any]] = None,
        numeric_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Sum all numeric columns in a CSV file (generic version)
        
        This is a generic function that works with any CSV file structure.
        It auto-detects numeric columns and sums them.
        
        Args:
            file_path: Path to CSV file
            filters: Optional column filters (e.g., {'Model': '4Runner TRD Pro'})
            numeric_columns: Optional list of numeric columns to sum (auto-detected if None)
            exclude_columns: Optional list of columns to exclude from auto-detection
            
        Returns:
            Dict with:
            - total: Total sum of all numeric columns
            - breakdown: Dict with individual column sums
            - metadata: Query metadata
        """
        # Auto-detect numeric columns if not provided
        if not numeric_columns:
            numeric_columns = self.detect_numeric_columns(file_path, exclude_columns)
            
            if not numeric_columns:
                return {
                    'total': 0,
                    'error': "Could not auto-detect numeric columns. Please specify numeric_columns parameter.",
                    'metadata': {'file_path': file_path}
                }
        
        # Build aggregation dict: sum all numeric columns
        aggregations = {col: ['sum'] for col in numeric_columns if col}
        
        # Query with filter and aggregations
        result = self.query_csv(
            file_path=file_path,
            filters=filters,
            aggregations=aggregations
        )
        
        if 'error' in result.get('metadata', {}):
            return {
                'total': 0,
                'error': result['metadata']['error'],
                'metadata': result.get('metadata', {})
            }
        
        if result.get('data') and len(result['data']) > 0:
            # Extract all the sum values from the result
            breakdown = result['data'][0]
            
            # Sum all the numeric column sums
            total = sum(
                float(value) if value is not None and not isinstance(value, str) else 0.0
                for key, value in breakdown.items()
                if key.endswith('_sum') and isinstance(value, (int, float))
            )
            
            return {
                'total': int(total) if total.is_integer() else total,
                'breakdown': breakdown,
                'metadata': result.get('metadata', {}),
                'columns_summed': len([k for k in breakdown.keys() if k.endswith('_sum')])
            }
        else:
            return {
                'total': 0,
                'error': 'No data found for the specified filter',
                'metadata': result.get('metadata', {}),
                'breakdown': {}
            }
    
    def get_total_inventory(
        self,
        file_path: str,
        filter_column: str,
        filter_value: str,
        numeric_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get total by summing numeric columns with a filter (generic version)
        
        This is a generic function that works with any CSV file structure.
        It filters by any column and sums any numeric columns.
        
        Args:
            file_path: Path to CSV file
            filter_column: Column name to filter on (e.g., 'Model', 'Product', 'Category')
            filter_value: Value to filter by (e.g., '4Runner TRD Pro', 'Widget A')
            numeric_columns: Optional list of numeric columns to sum (auto-detected if None)
            exclude_columns: Optional list of columns to exclude from auto-detection
            
        Returns:
            Dict with:
            - total: Total sum of all numeric columns
            - breakdown: Dict with individual column sums
            - metadata: Query metadata
        """
        return self.sum_numeric_columns(
            file_path=file_path,
            filters={filter_column: filter_value},
            numeric_columns=numeric_columns,
            exclude_columns=exclude_columns
        )


def csv_query(
    file_path: str,
    filters: Optional[Dict[str, Any]] = None,
    columns: Optional[List[str]] = None,
    aggregations: Optional[Dict[str, List[str]]] = None,
    group_by: Optional[List[str]] = None,
    order_by: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function for CSV queries (matches Azure Search API style)
    
    Args:
        file_path: Path to CSV file
        filters: Column filters (e.g., {'Model': '4Runner TRD Pro'})
        columns: Columns to return
        aggregations: Aggregations (e.g., {'2025-04-01 to 2025-04-07': ['sum']})
        group_by: Columns to group by
        order_by: Sort columns
        limit: Max rows
        
    Returns:
        Dict with 'data' and 'metadata'
    """
    tool = CSVQueryTool()
    return tool.query_csv(
        file_path=file_path,
        filters=filters,
        columns=columns,
        aggregations=aggregations,
        group_by=group_by,
        order_by=order_by,
        limit=limit
    )


def get_total_inventory_for_model(
    file_path: str,
    model: str,
    numeric_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get total inventory count for a model (convenience function)
    
    This is a convenience wrapper for inventory-style CSVs with a 'Model' column.
    For more generic use, see sum_numeric_columns().
    
    Args:
        file_path: Path to CSV file
        model: Model name (e.g., "4Runner TRD Pro")
        numeric_columns: Optional list of numeric columns to sum (auto-detected if None)
        exclude_columns: Optional list of columns to exclude from auto-detection
        
    Returns:
        Dict with total count and breakdown
    """
    tool = CSVQueryTool()
    return tool.get_total_inventory(
        file_path=file_path,
        filter_column='Model',
        filter_value=model,
        numeric_columns=numeric_columns,
        exclude_columns=exclude_columns
    )


def sum_numeric_columns(
    file_path: str,
    filters: Optional[Dict[str, Any]] = None,
    numeric_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Sum all numeric columns in a CSV file (generic convenience function)
    
    This is a generic function that works with any CSV file structure.
    It auto-detects numeric columns and sums them.
    
    Args:
        file_path: Path to CSV file
        filters: Optional column filters (e.g., {'Model': '4Runner TRD Pro', 'Region': 'North'})
        numeric_columns: Optional list of numeric columns to sum (auto-detected if None)
        exclude_columns: Optional list of columns to exclude from auto-detection
        
    Returns:
        Dict with total sum and breakdown
    """
    tool = CSVQueryTool()
    return tool.sum_numeric_columns(
        file_path=file_path,
        filters=filters,
        numeric_columns=numeric_columns,
        exclude_columns=exclude_columns
    )


def extract_csv_path_from_rag_results(rag_results: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract CSV file path from RAG search results
    
    Looks for file_path in results that points to a CSV file.
    
    Args:
        rag_results: List of RAG result dicts (from graphrag_search)
        
    Returns:
        CSV file path if found, None otherwise
    """
    for result in rag_results:
        file_path = result.get('file_path', '')
        if file_path and file_path.lower().endswith('.csv'):
            return file_path
        file_name = result.get('file_name', '')
        if file_name and file_name.lower().endswith('.csv'):
            # Try to reconstruct path from directory_path and file_name
            directory_path = result.get('directory_path', '')
            if directory_path:
                import os
                return os.path.join(directory_path, file_name)
    return None


def _semantic_exact_number_detection(query: str) -> Dict[str, Any]:
    """
    Semantic similarity-based detection for exact numerical queries (Option 2).
    Uses embeddings to match query against intent descriptions.
    
    Args:
        query: User query text
        
    Returns:
        Dict with 'requires' (bool), 'confidence' (float 0.0-1.0), and 'method' (str)
    """
    provider = _get_embeddings_provider()
    if provider is None:
        return {
            'requires': False,
            'confidence': 0.0,
            'method': 'semantic_disabled'
        }
    
    # Intent descriptions for exact numerical queries
    intent_descriptions = [
        "How many items are there?",
        "What is the total count?",
        "Calculate the sum of values",
        "What is the exact number?",
        "Count all matching records",
        "What is the total quantity?",
        "Sum all values",
        "How much is the total?",
        "What is the precise count?",
        "Calculate total inventory",
        "Get the exact amount",
        "What is the aggregate total?",
        "Count the number of items",
        "What is the sum total?",
        "How many total units?",
    ]
    
    try:
        # Generate embeddings
        texts_to_embed = [query] + intent_descriptions
        embeddings = provider.embed(texts_to_embed)
        
        if not embeddings or len(embeddings) < 2:
            return {
                'requires': False,
                'confidence': 0.0,
                'method': 'semantic_error'
            }
        
        query_embedding = np.array(embeddings[0])
        intent_embeddings = np.array(embeddings[1:])
        
        # Calculate cosine similarity
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        intent_norms = intent_embeddings / (np.linalg.norm(intent_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(intent_norms, query_norm)
        
        # Get max similarity and average similarity
        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))
        
        # Use weighted combination: max (0.7) + avg (0.3)
        confidence = (max_similarity * 0.7) + (avg_similarity * 0.3)
        
        # Threshold for detection (tuned for semantic similarity)
        requires = confidence >= 0.65  # Higher threshold for semantic
        
        logger.debug(
            f"_semantic_exact_number_detection('{query[:50]}...') = "
            f"requires={requires}, confidence={confidence:.3f}, "
            f"max_sim={max_similarity:.3f}, avg_sim={avg_similarity:.3f}"
        )
        
        return {
            'requires': requires,
            'confidence': confidence,
            'method': 'semantic',
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity
        }
        
    except Exception as e:
        logger.warning(f"Semantic detection failed: {e}. Falling back to heuristics.")
        return {
            'requires': False,
            'confidence': 0.0,
            'method': 'semantic_error'
        }


def _semantic_list_detection(query: str) -> Dict[str, Any]:
    """
    Semantic similarity-based detection for list queries (Option 2).
    Uses embeddings to match query against intent descriptions.
    
    Args:
        query: User query text
        
    Returns:
        Dict with 'requires' (bool), 'confidence' (float 0.0-1.0), and 'method' (str)
    """
    provider = _get_embeddings_provider()
    if provider is None:
        return {
            'requires': False,
            'confidence': 0.0,
            'method': 'semantic_disabled'
        }
    
    # Intent descriptions for list queries
    intent_descriptions = [
        "List all items",
        "What are all the options?",
        "Show me all available values",
        "Name all items",
        "What are all the different types?",
        "List every option",
        "Show all distinct values",
        "What are all the categories?",
        "Enumerate all items",
        "Display all available options",
        "What are all the possible values?",
        "List all unique items",
        "Show me all the choices",
        "What are all the variants?",
        "Get all distinct entries",
    ]
    
    try:
        # Generate embeddings
        texts_to_embed = [query] + intent_descriptions
        embeddings = provider.embed(texts_to_embed)
        
        if not embeddings or len(embeddings) < 2:
            return {
                'requires': False,
                'confidence': 0.0,
                'method': 'semantic_error'
            }
        
        query_embedding = np.array(embeddings[0])
        intent_embeddings = np.array(embeddings[1:])
        
        # Calculate cosine similarity
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        intent_norms = intent_embeddings / (np.linalg.norm(intent_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(intent_norms, query_norm)
        
        # Get max similarity and average similarity
        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))
        
        # Use weighted combination: max (0.7) + avg (0.3)
        confidence = (max_similarity * 0.7) + (avg_similarity * 0.3)
        
        # Threshold for detection (tuned for semantic similarity)
        requires = confidence >= 0.65  # Higher threshold for semantic
        
        logger.debug(
            f"_semantic_list_detection('{query[:50]}...') = "
            f"requires={requires}, confidence={confidence:.3f}, "
            f"max_sim={max_similarity:.3f}, avg_sim={avg_similarity:.3f}"
        )
        
        return {
            'requires': requires,
            'confidence': confidence,
            'method': 'semantic',
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity
        }
        
    except Exception as e:
        logger.warning(f"Semantic detection failed: {e}. Falling back to heuristics.")
        return {
            'requires': False,
            'confidence': 0.0,
            'method': 'semantic_error'
        }


def _heuristic_exact_number_detection(query: str) -> Dict[str, Any]:
    """
    Heuristic-based detection for exact numerical queries.
    Uses structural patterns instead of hardcoded keywords.
    
    Args:
        query: User query text
        
    Returns:
        Dict with 'requires' (bool) and 'confidence' (float 0.0-1.0)
    """
    import re
    
    query_lower = query.lower().strip()
    confidence = 0.0
    signals = []
    
    # Pattern 1: Question words + quantity indicators (high confidence)
    quantity_indicators = ['many', 'much', 'total', 'sum', 'count', 'number', 'quantity']
    question_words = ['how', 'what', 'which']
    
    has_question = any(qw in query_lower for qw in question_words)
    has_quantity = any(qi in query_lower for qi in quantity_indicators)
    
    if has_question and has_quantity:
        confidence += 0.5
        signals.append("question+quantity")
    
    # Pattern 2: Imperative calculation verbs (high confidence)
    calc_verbs = ['calculate', 'compute', 'sum', 'count', 'aggregate', 'add', 'total']
    if any(verb in query_lower for verb in calc_verbs):
        confidence += 0.4
        signals.append("calculation_verb")
    
    # Pattern 3: "exact" or "precise" modifiers (very high confidence)
    if 'exact' in query_lower or 'precise' in query_lower:
        confidence += 0.6
        signals.append("exactness_modifier")
    
    # Pattern 4: Mathematical operations mentioned (medium confidence)
    math_ops = ['add', 'sum', 'total', 'aggregate', 'summarize', 'tally']
    if any(op in query_lower for op in math_ops):
        confidence += 0.3
        signals.append("math_operation")
    
    # Pattern 5: "what is the" + quantity pattern (high confidence)
    if re.search(r'what\s+is\s+the\s+(total|count|sum|number)', query_lower):
        confidence += 0.5
        signals.append("what_is_quantity")
    
    # Pattern 6: Numeric result expectation (low confidence, but helps)
    if re.search(r'\d+', query_lower) and (has_question or has_quantity):
        confidence += 0.1
        signals.append("numeric_expectation")
    
    # Cap confidence at 1.0
    confidence = min(confidence, 1.0)
    
    requires = confidence >= 0.4  # Threshold for detection
    
    logger.debug(
        f"_heuristic_exact_number_detection('{query[:50]}...') = "
        f"requires={requires}, confidence={confidence:.2f}, signals={signals}"
    )
    
    return {
        'requires': requires,
        'confidence': confidence,
        'signals': signals
    }


def query_requires_exact_numbers(query: str) -> bool:
    """
    Detect if a query requires exact numerical calculation.
    
    Uses hybrid approach: Option 2 (semantic similarity) + Option 3 (heuristics).
    - Primary: Semantic similarity (if available)
    - Fallback: Heuristic pattern matching
    
    Args:
        query: User query text
        
    Returns:
        True if query likely requires exact numerical calculation
    """
    # Try semantic detection first (Option 2)
    semantic_result = _semantic_exact_number_detection(query)
    
    # If semantic detection is available and confident, use it
    if semantic_result['method'] == 'semantic' and semantic_result['confidence'] >= 0.65:
        logger.debug(f"Using semantic detection for exact numbers: confidence={semantic_result['confidence']:.3f}")
        return semantic_result['requires']
    
    # Fallback to heuristic detection (Option 3)
    heuristic_result = _heuristic_exact_number_detection(query)
    
    # If semantic was attempted but low confidence, combine with heuristic
    if semantic_result['method'] == 'semantic' and semantic_result['confidence'] > 0.0:
        # Weighted combination: semantic (0.6) + heuristic (0.4)
        combined_confidence = (semantic_result['confidence'] * 0.6) + (heuristic_result['confidence'] * 0.4)
        requires = combined_confidence >= 0.5
        logger.debug(
            f"Using combined detection (semantic+heuristic) for exact numbers: "
            f"semantic={semantic_result['confidence']:.3f}, "
            f"heuristic={heuristic_result['confidence']:.3f}, "
            f"combined={combined_confidence:.3f}, requires={requires}"
        )
        return requires
    
    # Pure heuristic fallback
    logger.debug(f"Using heuristic detection for exact numbers: confidence={heuristic_result['confidence']:.3f}")
    return heuristic_result['requires']


def _heuristic_list_detection(query: str) -> Dict[str, Any]:
    """
    Heuristic-based detection for list queries.
    Uses structural patterns instead of hardcoded keywords.
    
    Args:
        query: User query text
        
    Returns:
        Dict with 'requires' (bool) and 'confidence' (float 0.0-1.0)
    """
    import re
    
    query_lower = query.lower().strip()
    confidence = 0.0
    signals = []
    
    # Pattern 1: "all" + plural noun pattern (high confidence)
    # Matches: "all models", "all products", "all items"
    if re.search(r'\ball\s+\w+s\b', query_lower):
        confidence += 0.5
        signals.append("all_plural")
    
    # Pattern 2: List/enumerate verbs + "all" or "every" (high confidence)
    list_verbs = ['list', 'name', 'show', 'enumerate', 'display', 'present']
    if any(verb in query_lower for verb in list_verbs):
        if 'all' in query_lower or 'every' in query_lower:
            confidence += 0.6
            signals.append("list_verb_all")
        else:
            confidence += 0.2  # Lower confidence if just verb without "all"
            signals.append("list_verb")
    
    # Pattern 3: "what are all" pattern (very high confidence)
    if re.search(r'what\s+are\s+all', query_lower):
        confidence += 0.7
        signals.append("what_are_all")
    
    # Pattern 4: Plural question pattern (medium confidence)
    # Matches: "what are the models", "what are the products"
    if re.search(r'what\s+(are|is)\s+the\s+\w+s', query_lower):
        confidence += 0.4
        signals.append("plural_question")
    
    # Pattern 5: "name all" or "list all" explicit patterns (very high confidence)
    if re.search(r'(name|list|show|enumerate)\s+all', query_lower):
        confidence += 0.7
        signals.append("explicit_list_all")
    
    # Pattern 6: "every" + noun pattern (medium confidence)
    if re.search(r'\bevery\s+\w+', query_lower):
        confidence += 0.3
        signals.append("every_pattern")
    
    # Pattern 7: "complete list" or "full list" (high confidence)
    if re.search(r'(complete|full|entire)\s+list', query_lower):
        confidence += 0.5
        signals.append("complete_list")
    
    # Cap confidence at 1.0
    confidence = min(confidence, 1.0)
    
    requires = confidence >= 0.4  # Threshold for detection
    
    logger.debug(
        f"_heuristic_list_detection('{query[:50]}...') = "
        f"requires={requires}, confidence={confidence:.2f}, signals={signals}"
    )
    
    return {
        'requires': requires,
        'confidence': confidence,
        'signals': signals
    }


def query_requires_list(query: str) -> bool:
    """
    Detect if a query requires a list of distinct values.
    
    Uses hybrid approach: Option 2 (semantic similarity) + Option 3 (heuristics).
    - Primary: Semantic similarity (if available)
    - Fallback: Heuristic pattern matching
    
    Args:
        query: User query text
        
    Returns:
        True if query likely requires a list of distinct values
    """
    # Try semantic detection first (Option 2)
    semantic_result = _semantic_list_detection(query)
    
    # If semantic detection is available and confident, use it
    if semantic_result['method'] == 'semantic' and semantic_result['confidence'] >= 0.65:
        logger.debug(f"Using semantic detection for list: confidence={semantic_result['confidence']:.3f}")
        return semantic_result['requires']
    
    # Fallback to heuristic detection (Option 3)
    heuristic_result = _heuristic_list_detection(query)
    
    # If semantic was attempted but low confidence, combine with heuristic
    if semantic_result['method'] == 'semantic' and semantic_result['confidence'] > 0.0:
        # Weighted combination: semantic (0.6) + heuristic (0.4)
        combined_confidence = (semantic_result['confidence'] * 0.6) + (heuristic_result['confidence'] * 0.4)
        requires = combined_confidence >= 0.5
        logger.debug(
            f"Using combined detection (semantic+heuristic) for list: "
            f"semantic={semantic_result['confidence']:.3f}, "
            f"heuristic={heuristic_result['confidence']:.3f}, "
            f"combined={combined_confidence:.3f}, requires={requires}"
        )
        return requires
    
    # Pure heuristic fallback
    logger.debug(f"Using heuristic detection for list: confidence={heuristic_result['confidence']:.3f}")
    return heuristic_result['requires']
    """
    Detect if a query requires listing all values from a column.
    
    Uses heuristic pattern matching (Option 3) - no hardcoded keywords.
    Can be extended with semantic similarity (Option 2) or LLM (Option 1) later.
    
    Args:
        query: User query text
        
    Returns:
        True if query likely requires listing all values
    """
    result = _heuristic_list_detection(query)
    return result['requires']


def is_csv_specific_query(query: str) -> bool:
    """
    Detect if query explicitly mentions CSV/inventory/file
    
    Args:
        query: User query text
        
    Returns:
        True if query is CSV-specific (mentions CSV, inventory file, etc.)
    """
    query_lower = query.lower()
    csv_indicators = [
        'csv', 'inventory', 'inventory file', 'inventory csv',
        'in the csv', 'from the csv', 'in the file', 'from the file',
        'inventory data', 'csv file', 'csv data', 'in the inventory'
    ]
    return any(indicator in query_lower for indicator in csv_indicators)


def is_date_like(value: str) -> bool:
    """Check if a value looks like a date/date range (to avoid using dates as filter values)"""
    import re
    value_lower = value.lower().strip()
    
    # Common month names (reject these)
    month_names = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    if value_lower in month_names:
        return True
    
    # Patterns that indicate dates
    date_patterns = [
        r'\d{4}',  # Contains year (2025, 2024, etc.)
        r'\d{1,2}[/-]\d{1,2}',  # Date format (MM/DD or DD-MM)
        r'to\s+\d',  # Date range ("to 2025" or "to 01-07")
        r'planned',  # "Planned" suffix
        r'week|month|quarter|year',  # Time period keywords
    ]
    
    # Check if value matches date patterns
    for pattern in date_patterns:
        if re.search(pattern, value_lower):
            return True
    
    # Check if it's mostly numbers (likely a date)
    if re.match(r'^\d+[\s-]+\d+', value):
        return True
    
    return False


def is_valid_model_name(value: str) -> bool:
    """Check if a value looks like a valid model/product name"""
    import re
    value = value.strip()
    
    # Too short
    if len(value) < 3:
        return False
    
    # Reject date-like values
    if is_date_like(value):
        return False
    
    # Reject if it's just numbers
    if re.match(r'^\d+$', value):
        return False
    
    # Reject common non-model words
    reject_words = [
        'columns', 'rows', 'data', 'table', 'csv', 'file',
        'factory', 'location', 'planned', 'total', 'sum',
        'count', 'average', 'mean', 'max', 'min'
    ]
    if value.lower() in reject_words:
        return False
    
    # Should have at least one letter (not just numbers/symbols)
    if not re.search(r'[a-zA-Z]', value):
        return False
    
    return True


def extract_filter_value_from_query(
    query: str,
    filter_column: str,
    rag_results: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Extract filter value from query or RAG results (generic version)
    
    Tries to find values for a specific column from the query or RAG results.
    Works with any column name, not just "Model".
    Excludes date-like values to avoid matching date columns.
    
    Args:
        query: User query text
        filter_column: Column name to extract value for (e.g., 'Model', 'Product', 'Category')
        rag_results: Optional RAG results to extract value from
        
    Returns:
        Filter value if found, None otherwise
    """
    import re
    
    # Check RAG results first for column mentions
    if rag_results:
        for result in rag_results:
            text = result.get('text', '')
            
            # Look for table format: | Model | 4Runner TRD Pro |
            # This is the most reliable pattern for CSV tables
            # Pattern: | Model | value | (where value is in the data row, not header)
            # We need to find the header row first, then get the value from the data row
            
            # Find all table rows
            table_rows = re.findall(r'\|[^|\n]+\|', text)
            if len(table_rows) >= 2:  # At least header + one data row
                # Find header row with our column
                header_row = None
                header_col_index = -1
                for row in table_rows[:5]:  # Check first few rows (usually header is early)
                    cells = [cell.strip() for cell in row.split('|')[1:-1]]  # Split and remove empty first/last
                    try:
                        col_index = [c.lower() for c in cells].index(filter_column.lower())
                        header_row = cells
                        header_col_index = col_index
                        logger.debug(f"Found {filter_column} column at index {col_index} in header")
                        break
                    except ValueError:
                        continue
                
                # If we found the header, get values from data rows
                if header_col_index >= 0:
                    for row in table_rows[1:]:  # Skip header row
                        cells = [cell.strip() for cell in row.split('|')[1:-1]]
                        if len(cells) > header_col_index:
                            value = cells[header_col_index].strip()
                            # Normalize whitespace: replace multiple spaces with single space
                            value = re.sub(r'\s+', ' ', value).strip()
                            # Validate the value
                            if is_valid_model_name(value):
                                logger.debug(f"Extracted {filter_column} value from table row: '{value}'")
                                return value
            
            # Fallback: simpler table pattern (less reliable)
            table_pattern = rf'\|\s*{re.escape(filter_column)}\s*\|[^|]*\|\s*([^\n|]+?)\s*\|'
            match = re.search(table_pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                value = re.sub(r'\s+', ' ', value).strip()
                # Validate the value
                if is_valid_model_name(value):
                    logger.debug(f"Extracted {filter_column} value from table (fallback): '{value}'")
                    return value
            
            # Look for other patterns (but prioritize non-date values)
            patterns = [
                rf'{re.escape(filter_column)}[:\s]+\|?\s*([^\n|]+?)(?:\s*\||\s*$)',
                rf'{re.escape(filter_column)}[:\s]+([^\n|]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    value = re.sub(r'\s+', ' ', value).strip()
                    # Validate the value
                    if is_valid_model_name(value):
                        logger.debug(f"Extracted {filter_column} value from RAG: '{value}'")
                        return value
    
    # Look in query for the model/product name directly
    # Pattern 1: Quoted strings
    quoted = re.search(r'"([^"]+)"', query)
    if quoted:
        value = quoted.group(1).strip()
        # Normalize whitespace: replace multiple spaces with single space
        value = re.sub(r'\s+', ' ', value).strip()
        if is_valid_model_name(value):
            logger.debug(f"Extracted {filter_column} value from quoted string: '{value}'")
            return value
    
    # Pattern 2: After "how many" or "how much" - extract the product name
    # "How many 4Runner TRD Pro are?" -> "4Runner TRD Pro"
    # This pattern handles product names that may start with numbers (like "4Runner")
    how_many_pattern = r'how\s+(?:many|much)\s+([A-Z0-9][A-Za-z0-9\s]+?)(?:\s+are|\s+is|\s+there|\s+in|\s*$|\?)'
    match = re.search(how_many_pattern, query, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        # Normalize whitespace: replace multiple spaces with single space
        value = re.sub(r'\s+', ' ', value).strip()
        if is_valid_model_name(value):
            logger.debug(f"Extracted {filter_column} value after 'how many/much': '{value}'")
            return value
    
    # Pattern 3: After "for" or "of" (e.g., "how many X for Model Y")
    after_for = re.search(r'(?:for|of)\s+([A-Z][^?.,!]+?)(?:\s+are|\s+is|\s+there|\s+in|$)', query, re.IGNORECASE)
    if after_for:
        value = after_for.group(1).strip()
        # Normalize whitespace: replace multiple spaces with single space
        value = re.sub(r'\s+', ' ', value).strip()
        if is_valid_model_name(value):
            logger.debug(f"Extracted {filter_column} value after 'for/of': '{value}'")
            return value
    
    # Pattern 4: Product names that may start with numbers (e.g., "4Runner TRD Pro")
    # This pattern handles: "4Runner", "4Runner TRD", "4Runner TRD Pro"
    number_product_pattern = r'\b(\d+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    match = re.search(number_product_pattern, query)
    if match:
        value = match.group(1).strip()
        # Normalize whitespace: replace multiple spaces with single space
        value = re.sub(r'\s+', ' ', value).strip()
        # Try to extend to get full name (e.g., "4Runner TRD Pro" instead of just "4Runner")
        # Look for additional capitalized words after the number+word pattern
        extended_pattern = rf'{re.escape(value)}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        extended_match = re.search(extended_pattern, query)
        if extended_match:
            extended_value = f"{value} {extended_match.group(1)}"
            # Normalize whitespace: replace multiple spaces with single space
            extended_value = re.sub(r'\s+', ' ', extended_value).strip()
            if is_valid_model_name(extended_value):
                logger.debug(f"Extracted {filter_column} value from extended number+product pattern: '{extended_value}'")
                return extended_value
        if is_valid_model_name(value):
            logger.debug(f"Extracted {filter_column} value from number+product pattern: '{value}'")
            return value
    
    # Pattern 5: Capitalized multi-word phrases (e.g., "TRD Pro" if no number prefix)
    # Look for sequences of capitalized words, prioritizing longer matches
    capitalized_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b'
    matches = re.findall(capitalized_pattern, query)
    # Sort by length (longest first) to get longer matches first
    matches.sort(key=len, reverse=True)
    for match in matches:
        # Normalize whitespace: replace multiple spaces with single space
        match = re.sub(r'\s+', ' ', match).strip()
        # Skip common words that aren't product names
        skip_words = {'How', 'Many', 'What', 'Total', 'Sum', 'Count', 'There', 'Are', 'Is', 'Toyota'}
        words = match.split()
        if (len(words) >= 2 and 
            not any(word in skip_words for word in words) and
            is_valid_model_name(match)):
            logger.debug(f"Extracted {filter_column} value from capitalized phrase: '{match}'")
            return match
    
    logger.debug(f"Could not extract {filter_column} value from query: '{query[:100]}...'")
    return None


def extract_model_from_query(query: str, rag_results: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """
    Extract model name from query or RAG results (convenience function)
    
    This is a convenience wrapper for extract_filter_value_from_query().
    For more generic use, call extract_filter_value_from_query() directly.
    
    Args:
        query: User query text
        rag_results: Optional RAG results to extract model from
        
    Returns:
        Model name if found, None otherwise
    """
    return extract_filter_value_from_query(query, 'Model', rag_results)


def get_distinct_values(
    file_path: str,
    column: str,
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get distinct/unique values from a CSV column (convenience function)
    
    Args:
        file_path: Path to CSV file
        column: Column name to get distinct values from
        filters: Optional filters to apply
        order_by: Optional sorting
        
    Returns:
        Dict with 'values' (list), 'count', and 'metadata'
    """
    tool = CSVQueryTool()
    return tool.get_distinct_values(file_path, column, filters, order_by)

