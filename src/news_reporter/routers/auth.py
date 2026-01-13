# routers/auth.py
from fastapi import APIRouter, HTTPException, Response, Cookie, Depends
from typing import Optional
from datetime import datetime, timedelta
import bcrypt
import secrets
from pymongo import MongoClient
from bson import ObjectId
import os
from datetime import timezone  # Consistency patch: for timezone-aware timestamps

from ..models.auth import UserRegister, UserLogin, UserResponse
from ..config import Settings
from ..dependencies.auth import UserPrincipal, get_current_user  # Canonical auth dependencies

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


def create_session(user_id: str) -> dict:
    """Create a new session for the user."""
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS)  # Consistency patch: timezone-aware
    
    session = {
        "_id": session_id,
        "userId": user_id,
        "createdAt": datetime.now(timezone.utc),  # Consistency patch: timezone-aware
        "expiresAt": expires_at
    }
    
    sessions_collection.insert_one(session)
    return session


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister):
    """Register a new user."""
    print("\n\n")
    print("\nregister")
    if auth_db is None:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    # Check if user already exists
    existing_user = users_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate password length
    if len(user_data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    # Hash password and create user
    hashed_pw = hash_password(user_data.password)
    new_user = {
        "email": user_data.email,
        "passwordHash": hashed_pw,
        "createdAt": datetime.now(timezone.utc)  # Consistency patch: timezone-aware
    }
    
    result = users_collection.insert_one(new_user)
    
    return UserResponse(
        id=str(result.inserted_id),
        email=user_data.email,
        created_at=new_user["createdAt"]
    )


@router.post("/login")
async def login(credentials: UserLogin, response: Response):
    """Login and create a session."""
    print("\n\n")
    print("\nlogin")
    # Find user
    user = users_collection.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not verify_password(credentials.password, user["passwordHash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create session
    session = create_session(str(user["_id"]))
    
    # Set HttpOnly cookie
    response.set_cookie(
        key="sid",
        value=session["_id"],
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in production with HTTPS
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
    """Logout and delete session."""
    if sid:
        sessions_collection.delete_one({"_id": sid})
    
    response.delete_cookie(key="sid", path="/")
    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_current_user_info(user: UserPrincipal = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    PR5 Fix 2: Works with UserPrincipal auth contract.
    """
    return {
        "id": user.id,
        "email": user.email
    }
