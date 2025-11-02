"""Neo4j database connection management"""

import logging
from neo4j import GraphDatabase
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Manages Neo4j database connection"""
    
    def __init__(self):
        """Initialize connection with settings"""
        settings = get_settings()
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
        self.encrypted = settings.neo4j_encrypted
        self.driver: Optional[GraphDatabase.driver] = None
        
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                encrypted=self.encrypted
            )
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j database connection")
    
    def get_session(self):
        """Get a new database session"""
        if not self.driver:
            self.connect()
        return self.driver.session()


# Global connection instance
_neo4j_connection: Optional[Neo4jConnection] = None


def get_neo4j_connection() -> Neo4jConnection:
    """Get Neo4j connection instance (singleton pattern)"""
    global _neo4j_connection
    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection()
    return _neo4j_connection


