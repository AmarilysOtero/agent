"""
PR5 Integration Tests: Auth Dependency

Tests for the canonical auth dependency (get_current_user) including
cookie authentication, session validation, and error handling.
"""
import pytest
from fastapi import HTTPException
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta

from src.news_reporter.dependencies.auth import get_current_user, UserPrincipal, get_auth_collections


class TestAuthDependency:
    """Tests for canonical get_current_user dependency"""
    
    def test_no_cookie_returns_401(self):
        """Test that missing cookie returns 401 Unauthorized"""
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(sid=None)
        
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Not authenticated"
    
    def test_invalid_session_id_returns_401(self):
        """Test that invalid session ID returns 401"""
        mock_sessions = MagicMock()
        mock_users = MagicMock()
        
        # Session not found
        mock_sessions.find_one.return_value = None
        
        with patch('src.news_reporter.dependencies.auth.get_auth_collections', return_value=(mock_sessions, mock_users)):
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(sid="invalid_session_id")
            
            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Invalid session"
    
    def test_expired_session_returns_401_and_deletes(self):
        """Test that expired session returns 401 and attempts deletion"""
        mock_sessions = MagicMock()
        mock_users = MagicMock()
        
        # Expired session
        expired_time = datetime.now(timezone.utc) - timedelta(days=1)
        mock_sessions.find_one.return_value = {
            "_id": "test_session",
            "userId": "test_user_id",
            "expiresAt": expired_time
        }
        
        with patch('src.news_reporter.dependencies.auth.get_auth_collections', return_value=(mock_sessions, mock_users)):
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(sid="test_session")
            
            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Session expired"
            
            # Verify session deletion was attempted
            mock_sessions.delete_one.assert_called_once_with({"_id": "test_session"})
    
    def test_user_not_found_returns_401(self):
        """Test that missing user returns 401"""
        from bson import ObjectId
        
        mock_sessions = MagicMock()
        mock_users = MagicMock()
        
        # Valid session but user not found
        user_id = ObjectId()  # Use valid ObjectId
        future_time = datetime.now(timezone.utc) + timedelta(days=1)
        mock_sessions.find_one.return_value = {
            "_id": "test_session",
            "userId": str(user_id),
            "expiresAt": future_time
        }
        mock_users.find_one.return_value = None
        
        with patch('src.news_reporter.dependencies.auth.get_auth_collections', return_value=(mock_sessions, mock_users)):
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(sid="test_session")
            
            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "User not found"
    
    def test_valid_session_returns_user_principal(self):
        """Test that valid session returns UserPrincipal with id and email"""
        from bson import ObjectId
        
        mock_sessions = MagicMock()
        mock_users = MagicMock()
        
        user_id = ObjectId()
        future_time = datetime.now(timezone.utc) + timedelta(days=1)
        
        mock_sessions.find_one.return_value = {
            "_id": "test_session",
            "userId": str(user_id),
            "expiresAt": future_time
        }
        
        mock_users.find_one.return_value = {
            "_id": user_id,
            "email": "test@example.com"
        }
        
        with patch('src.news_reporter.dependencies.auth.get_auth_collections', return_value=(mock_sessions, mock_users)):
            result = get_current_user(sid="test_session")
            
            assert isinstance(result, UserPrincipal)
            assert result.id == str(user_id)
            assert result.email == "test@example.com"
    
    def test_database_unavailable_returns_503(self):
        """Test that database connection failure returns 503"""
        with patch('src.news_reporter.dependencies.auth.get_auth_collections') as mock_get_collections:
            mock_get_collections.side_effect = HTTPException(
                status_code=503,
                detail="Authentication service unavailable"
            )
            
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(sid="test_session")
            
            assert exc_info.value.status_code == 503
            assert "unavailable" in exc_info.value.detail.lower()
    
    def test_lazy_initialization_no_import_time_connection(self):
        """Test that importing the module doesn't connect to MongoDB"""
        # This test verifies that the module-level variables are None
        # and get_auth_collections handles lazy init
        
        # Reset module-level cache
        import src.news_reporter.dependencies.auth as auth_module
        auth_module._auth_client = None
        auth_module._sessions_collection = None
        auth_module._users_collection = None
        
        # Verify they're None (no import-time connection)
        assert auth_module._auth_client is None
        assert auth_module._sessions_collection is None
        assert auth_module._users_collection is None
