# routers/auth.py
from fastapi import APIRouter, HTTPException, Response, Cookie, Depends
from typing import Optional
from datetime import datetime, timedelta, timezone
import bcrypt
import secrets
from pymongo import MongoClient
from bson import ObjectId
import os

from ..models.auth import UserRegister, UserLogin, UserResponse
from ..config import Settings
from ..dependencies.auth import UserPrincipal, get_current_user, get_auth_collections

router = APIRouter(prefix="/api/auth", tags=["auth"])

# MongoDB connection
try:
    settings = Settings.load()
    MONGO_AUTH_URL = settings.auth_api_url
    if not MONGO_AUTH_URL:
        raise RuntimeError("MONGO_AUTH_URL is not set in environment or .env")
    
    client = MongoClient(MONGO_AUTH_URL)
    auth_db = client.get_database()  # Uses database from URI
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    # We don't raise here to avoid crashing the whole app on import, 
    # but endpoints will fail if called.
    auth_db = None
    print(f"Failed to connect to MongoDB: {e}")

# Collections
if auth_db is not None:
    users_collection = auth_db["users"]
    sessions_collection = auth_db["sessions"]
else:
    users_collection = None
    sessions_collection = None

# Session TTL (7 days)
SESSION_EXPIRY_DAYS = 7


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister):
    """
    Register a new user.
    
    Returns 400 if email already exists.
    Returns 503 if database unavailable.
    """
    # Get collections with lazy initialization
    sessions_collection, users_collection = get_auth_collections()
    
    # Check if user exists
    existing = users_collection.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate password length
    if len(user_data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    # Create user document
    user_doc = {
        "email": user_data.email,
        "passwordHash": hash_password(user_data.password),
        "createdAt": datetime.now(timezone.utc)
    }
    
    result = users_collection.insert_one(user_doc)
    
    return UserResponse(
        id=str(result.inserted_id),
        email=user_data.email,
        createdAt=user_doc["createdAt"]
    )


@router.post("/login")
async def login(credentials: UserLogin, response: Response):
    """
    Login user and create session.
    
    Returns 401 if credentials invalid.
    Returns 503 if database unavailable.
    Always returns JSON (never HTML).
    """
    # Get collections with lazy initialization
    sessions_collection, users_collection = get_auth_collections()
    
    # Find user by email
    user = users_collection.find_one({"email": credentials.email})
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not verify_password(credentials.password, user["passwordHash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create session
    session_id = secrets.token_urlsafe(32)
    session_doc = {
        "_id": session_id,
        "userId": str(user["_id"]),
        "email": user["email"],
        "createdAt": datetime.now(timezone.utc),
        "expiresAt": datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS)
    }
    
    sessions_collection.insert_one(session_doc)
    
    # Set HttpOnly cookie
    response.set_cookie(
        key="sid",
        value=session_id,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60,
        path="/"
    )
    
    return {
        "message": "Login successful",
        "user": {
            "id": str(user["_id"]),
            "email": user["email"]
        }
    }


@router.post("/logout")
async def logout(response: Response, sid: Optional[str] = Cookie(None)):
    """
    Log out user by clearing session cookie.
    
    Returns 503 if database unavailable (but still clears cookie).
    """
    if sid:
        try:
            sessions_collection, _ = get_auth_collections()
            sessions_collection.delete_one({"_id": sid})
        except HTTPException:
            # DB unavailable, but we can still clear the cookie
            pass
    
    # Clear cookie
    response.delete_cookie(key="sid", path="/")
    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_me(user: UserPrincipal = Depends(get_current_user)):
    """
    Get current authenticated user info.
    
    Returns 401 if not authenticated (handled by get_current_user dependency).
    Returns 200 with user info if authenticated.
    """
    return {
        "id": user.id,
        "email": user.email,
        "authenticated": True
    }
