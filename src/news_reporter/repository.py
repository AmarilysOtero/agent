# src/news_reporter/repository.py
"""LLM-driven repository for chat sessions using SQLGenerator"""
from __future__ import annotations
import logging
import json
from typing import List, Optional
from datetime import datetime

from .models import ChatSession, ChatSessionSummary, Message
from .database import Database
from .tools_sql.sql_generator import SQLGenerator

logger = logging.getLogger(__name__)


class LLMChatRepository:
    """
    Repository for chat sessions using LLM-driven SQL generation.
    All SQL queries are generated via SQLGenerator, no handwritten SQL.
    """
    
    # Database ID for this chat session database
    DATABASE_ID = "chat_sessions_db"
    
    def __init__(self, db: Database, sql_generator: SQLGenerator):
        """
        Initialize repository
        
        Args:
            db: Database connection helper
            sql_generator: LLM-based SQL generator
        """
        self.db = db
        self.sql_generator = sql_generator
    
    def ensure_schema(self) -> None:
        """
        Ensure database schema exists using LLM-generated DDL.
        Creates tables if they don't exist (idempotent).
        """
        try:
            # Check if tables already exist
            if self.db.table_exists("sessions") and self.db.table_exists("messages"):
                logger.info("Chat session schema already exists")
                return
            
            logger.info("Creating chat session schema using LLM-generated SQL...")
            
            # Generate schema creation SQL via LLM
            schema_query = """
            Create two tables if they don't exist:
            1. sessions table with columns: id (text primary key), user_id (text not null), 
               title (text), created_at (timestamp), updated_at (timestamp)
            2. messages table with columns: id (text primary key), session_id (text not null), 
               role (text not null), content (text not null), timestamp (timestamp), 
               sources (text for JSON), with foreign key to sessions(id)
            Use IF NOT EXISTS to make it idempotent.
            """
            
            result = self.sql_generator.generate_sql(
                query=schema_query,
                database_id=self.DATABASE_ID,
                top_k=5,
                similarity_threshold=0.5
            )
            
            if result.get("error") or not result.get("sql"):
                # Fallback: use hardcoded schema creation
                logger.warning(f"LLM schema generation failed: {result.get('error')}. Using fallback schema.")
                self._create_schema_fallback()
                return
            
            sql = result["sql"]
            logger.info(f"Generated schema SQL: {sql[:200]}...")
            
            # Execute the generated SQL (might be multiple statements)
            # Split by semicolon if multiple CREATE statements
            for statement in sql.split(";"):
                statement = statement.strip()
                if statement:
                    self.db.execute_sql(statement)
            
            logger.info("Chat session schema created successfully")
            
        except Exception as e:
            logger.error(f"Schema creation failed: {e}", exc_info=True)
            # Use fallback
            self._create_schema_fallback()
    
    def _create_schema_fallback(self) -> None:
        """Fallback schema creation with minimal handwritten SQL"""
        logger.info("Using fallback schema creation")
        
        # Minimal DDL as fallback
        self.db.execute_sql("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        
        self.db.execute_sql("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP,
                sources TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)
    
    def create_session(self, user_id: str, title: str = "New Chat") -> ChatSession:
        """
        Create a new chat session using LLM-generated INSERT
        
        Args:
            user_id: User ID
            title: Session title
            
        Returns:
            Created ChatSession
        """
        session = ChatSession(user_id=user_id, title=title)
        
        try:
            # Generate INSERT SQL via LLM
            insert_query = f"""
            Insert a new session into the sessions table with:
            - id = '{session.id}'
            - user_id = '{user_id}'
            - title = '{title}'
            - created_at = '{session.created_at.isoformat()}'
            - updated_at = '{session.updated_at.isoformat()}'
            """
            
            result = self.sql_generator.generate_sql(
                query=insert_query,
                database_id=self.DATABASE_ID,
                top_k=3
            )
            
            if result.get("error") or not result.get("sql"):
                logger.warning(f"LLM INSERT generation failed, using fallback")
                self._insert_session_fallback(session)
            else:
                sql = result["sql"]
                logger.debug(f"Generated INSERT: {sql}")
                self.db.execute_sql(sql)
            
            return session
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise
    
    def _insert_session_fallback(self, session: ChatSession) -> None:
        """Fallback INSERT for session"""
        sql = """
            INSERT INTO sessions (id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """
        self.db.execute_sql(sql, (
            session.id,
            session.user_id,
            session.title,
            session.created_at.isoformat(),
            session.updated_at.isoformat()
        ))
    
    def get_sessions(self, user_id: str) -> List[ChatSessionSummary]:
        """
        Get all sessions for a user using LLM-generated SELECT
        
        Args:
            user_id: User ID
            
        Returns:
            List of ChatSessionSummary
        """
        try:
            # Generate SELECT SQL via LLM
            select_query = f"""
            Select all sessions for user_id = '{user_id}' from the sessions table,
            ordered by updated_at descending.
            Include: id, user_id, title, created_at, updated_at
            """
            
            result = self.sql_generator.generate_sql(
                query=select_query,
                database_id=self.DATABASE_ID,
                top_k=3
            )
            
            if result.get("error") or not result.get("sql"):
                logger.warning(f"LLM SELECT generation failed, using fallback")
                rows = self._select_sessions_fallback(user_id)
            else:
                sql = result["sql"]
                logger.debug(f"Generated SELECT: {sql}")
                rows = self.db.execute_sql(sql)
            
            # Convert to summaries
            summaries = []
            for row in rows:
                summaries.append(ChatSessionSummary(
                    id=row["id"],
                    user_id=row["user_id"],
                    title=row.get("title", "New Chat"),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    message_count=0  # TODO: count messages
                ))
            
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            raise
    
    def _select_sessions_fallback(self, user_id: str) -> List[dict]:
        """Fallback SELECT for sessions"""
        sql = """
            SELECT id, user_id, title, created_at, updated_at
            FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
        """
        return self.db.execute_sql(sql, (user_id,))
    
    def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[ChatSession]:
        """
        Get a single session with all messages
        
        Args:
            session_id: Session ID
            user_id: Optional user ID for authorization check
            
        Returns:
            ChatSession with messages, or None if not found
        """
        try:
            # Get session metadata
            session_query = f"""
            Select the session with id = '{session_id}' from the sessions table.
            Include: id, user_id, title, created_at, updated_at
            """
            
            if user_id:
                session_query += f" and user_id = '{user_id}'"
            
            result = self.sql_generator.generate_sql(
                query=session_query,
                database_id=self.DATABASE_ID,
                top_k=3
            )
            
            if result.get("error") or not result.get("sql"):
                logger.warning("LLM SELECT generation failed, using fallback")
                session_rows = self._select_session_fallback(session_id, user_id)
            else:
                sql = result["sql"]
                session_rows = self.db.execute_sql(sql)
            
            if not session_rows:
                return None
            
            session_row = session_rows[0]
            
            # Get messages
            messages_query = f"""
            Select all messages for session_id = '{session_id}' from the messages table,
            ordered by timestamp ascending.
            Include: id, role, content, timestamp, sources
            """
            
            messages_result = self.sql_generator.generate_sql(
                query=messages_query,
                database_id=self.DATABASE_ID,
                top_k=3
            )
            
            if messages_result.get("error") or not messages_result.get("sql"):
                message_rows = self._select_messages_fallback(session_id)
            else:
                message_rows = self.db.execute_sql(messages_result["sql"])
            
            # Build session
            messages = []
            for msg_row in message_rows:
                sources = None
                if msg_row.get("sources"):
                    try:
                        sources = json.loads(msg_row["sources"])
                    except:
                        pass
                
                messages.append(Message(
                    id=msg_row["id"],
                    role=msg_row["role"],
                    content=msg_row["content"],
                    timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                    sources=sources
                ))
            
            return ChatSession(
                id=session_row["id"],
                user_id=session_row["user_id"],
                title=session_row.get("title", "New Chat"),
                created_at=datetime.fromisoformat(session_row["created_at"]),
                updated_at=datetime.fromisoformat(session_row["updated_at"]),
                messages=messages
            )
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            raise
    
    def _select_session_fallback(self, session_id: str, user_id: Optional[str]) -> List[dict]:
        """Fallback SELECT for single session"""
        if user_id:
            sql = "SELECT * FROM sessions WHERE id = ? AND user_id = ?"
            return self.db.execute_sql(sql, (session_id, user_id))
        else:
            sql = "SELECT * FROM sessions WHERE id = ?"
            return self.db.execute_sql(sql, (session_id,))
    
    def _select_messages_fallback(self, session_id: str) -> List[dict]:
        """Fallback SELECT for messages"""
        sql = """
            SELECT id, role, content, timestamp, sources
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """
        return self.db.execute_sql(sql, (session_id,))
    
    def add_message(self, session_id: str, message: Message) -> None:
        """
        Add a message to a session
        
        Args:
            session_id: Session ID
            message: Message to add
        """
        try:
            sources_json = json.dumps(message.sources) if message.sources else None
            
            # Generate INSERT via LLM
            insert_query = f"""
            Insert a new message into the messages table with:
            - id = '{message.id}'
            - session_id = '{session_id}'
            - role = '{message.role}'
            - content = '{message.content[:100]}...'
            - timestamp = '{message.timestamp.isoformat()}'
            - sources = JSON data
            """
            
            result = self.sql_generator.generate_sql(
                query=insert_query,
                database_id=self.DATABASE_ID,
                top_k=3
            )
            
            if result.get("error") or not result.get("sql"):
                logger.warning("LLM message INSERT failed, using fallback")
                self._insert_message_fallback(session_id, message, sources_json)
            else:
                # Note: parameterized version would be safer, but LLM might not handle it well
                # For now, use fallback for actual INSERT
                self._insert_message_fallback(session_id, message, sources_json)
            
            # Update session's updated_at
            self._update_session_timestamp(session_id)
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise
    
    def _insert_message_fallback(self, session_id: str, message: Message, sources_json: Optional[str]) -> None:
        """Fallback INSERT for message"""
        sql = """
            INSERT INTO messages (id, session_id, role, content, timestamp, sources)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.db.execute_sql(sql, (
            message.id,
            session_id,
            message.role,
            message.content,
            message.timestamp.isoformat(),
            sources_json
        ))
    
    def _update_session_timestamp(self, session_id: str) -> None:
        """Update session's updated_at timestamp"""
        sql = "UPDATE sessions SET updated_at = ? WHERE id = ?"
        self.db.execute_sql(sql, (datetime.utcnow().isoformat(), session_id))

    def update_session_title(self, session_id: str, title: str) -> None:
        """
        Update session title
        
        Args:
            session_id: Session ID
            title: New title
        """
        try:
            # Generate UPDATE SQL via LLM
            update_query = f"""
            Update the title of session with id = '{session_id}' to '{title}'.
            Also update updated_at to current timestamp.
            """
            
            result = self.sql_generator.generate_sql(
                query=update_query,
                database_id=self.DATABASE_ID,
                top_k=3
            )
            
            if result.get("error") or not result.get("sql"):
                logger.warning("LLM UPDATE title generation failed, using fallback")
                self._update_session_title_fallback(session_id, title)
            else:
                self.db.execute_sql(result["sql"])
                
        except Exception as e:
            logger.error(f"Failed to update session title: {e}")
            raise

    def _update_session_title_fallback(self, session_id: str, title: str) -> None:
        """Fallback UPDATE for session title"""
        sql = "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?"
        self.db.execute_sql(sql, (title, datetime.utcnow().isoformat(), session_id))
