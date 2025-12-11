# src/news_reporter/database.py
"""SQLite database connection and execution helper"""
from __future__ import annotations
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Database:
    """SQLite database connection manager"""
    
    def __init__(self, db_path: str = "chat_sessions.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_dir()
        logger.info(f"Database initialized at: {self.db_path}")
    
    def _ensure_db_dir(self):
        """Ensure the directory for the database file exists"""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a new database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def execute_sql(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL statement and return results
        
        Args:
            sql: SQL statement to execute
            params: Optional parameters for parameterized queries
            
        Returns:
            List of result rows as dictionaries
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # Commit if it's a write operation
            if sql.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP')):
                conn.commit()
                logger.debug(f"Executed write SQL: {sql[:100]}...")
                return []
            
            # Fetch results for SELECT queries
            results = [dict(row) for row in cursor.fetchall()]
            logger.debug(f"Executed read SQL: {sql[:100]}... (returned {len(results)} rows)")
            return results
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            logger.error(f"Failed SQL: {sql}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def execute_many(self, sql: str, params_list: List[tuple]) -> None:
        """
        Execute a SQL statement multiple times with different parameters
        
        Args:
            sql: SQL statement to execute
            params_list: List of parameter tuples
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            conn.commit()
            logger.debug(f"Executed batch SQL: {sql[:100]}... ({len(params_list)} operations)")
        except Exception as e:
            logger.error(f"Batch SQL execution error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        results = self.execute_sql(sql, (table_name,))
        return len(results) > 0
