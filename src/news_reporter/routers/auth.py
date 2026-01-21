from fastapi import APIRouter, HTTPException, Response, Cookie, Depends
from typing import Optional
from datetime import datetime, timedelta
import secrets
import os

# Optional bcrypt import
try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    bcrypt = None
    _BCRYPT_AVAILABLE = False

# Optional MongoDB imports
try:
    from pymongo import MongoClient
    from bson import ObjectId
    _MONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    ObjectId = None
    _MONGO_AVAILABLE = False

from ..models.auth import UserRegister, UserLogin, UserResponse
from ..config import Settings

router = APIRouter(prefix="/api/auth", tags=["auth"])

# MongoDB connection
auth_db = None
if _MONGO_AVAILABLE:
    try:
        settings = Settings.load()
        MONGO_AUTH_URL = settings.auth_api_url
        if not MONGO_AUTH_URL:
            print("WARNING: MONGO_AUTH_URL is not set in environment or .env")
            auth_db = None
        else:
            # Parse connection string to extract components for better error messages
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(MONGO_AUTH_URL)
            db_name = parsed.path.lstrip('/').split('?')[0] if parsed.path else 'auth_db'
            query_params = parse_qs(parsed.query)
            auth_source = query_params.get('authSource', [db_name])[0]
            
            print(f"[AUTH] Connecting to MongoDB:")
            print(f"  Host: {parsed.hostname}:{parsed.port}")
            print(f"  Database: {db_name}")
            print(f"  AuthSource: {auth_source}")
            print(f"  Username: {parsed.username}")
            
            # Use explicit parameters to avoid URL parsing issues with special characters
            # Extract password from URL (unquote to handle URL encoding)
            from urllib.parse import unquote
            password = unquote(parsed.password) if parsed.password else ""
            print(f"[AUTH] Using explicit parameters (password length: {len(password)})")
            
            # Use the parsed hostname from connection string (mongo in Docker, 127.0.0.1 locally)
            mongo_host = parsed.hostname or "127.0.0.1"
            mongo_port = parsed.port or 27017
            
            print(f"[AUTH] Connecting to {mongo_host}:{mongo_port}...")
            client = MongoClient(
                host=mongo_host,
                port=mongo_port,
                username=parsed.username,
                password=password,
                authSource=auth_source,
                authMechanism="SCRAM-SHA-256",
                serverSelectionTimeoutMS=5000
            )
            auth_db = client[db_name]
            auth_db.command('ping')
            # Connection already tested above
            print(f"[AUTH] Successfully connected to MongoDB database: {auth_db.name}")
    except Exception as e:
        print(f"[AUTH] Failed to connect to MongoDB: {e}")
        print(f"[AUTH] Connection string: {MONGO_AUTH_URL.split('@')[0]}@***")
        print(f"[AUTH] Troubleshooting:")
        print(f"  1. Verify MongoDB is running (run from any directory):")
        print(f"     docker ps | findstr rag-mongo")
        print(f"  2. Check password matches MONGO_APP_PASS (run from RAG_Infra directory):")
        print(f"     cd c:\\Alexis\\Projects\\RAG_Infra")
        print(f"     # Check .env file for MONGO_APP_PASS value")
        print(f"  3. Verify user exists (run from RAG_Infra directory):")
        print(f"     cd c:\\Alexis\\Projects\\RAG_Infra")
        print(f"     docker exec rag-mongo mongosh -u root -p rootpassword --authenticationDatabase admin")
        print(f"  4. Test connection (run from RAG_Infra directory):")
        print(f"     cd c:\\Alexis\\Projects\\RAG_Infra")
        print(f"     docker exec rag-mongo mongosh \"{MONGO_AUTH_URL.split('@')[0]}@127.0.0.1:27017/{MONGO_AUTH_URL.split('/')[-1].split('?')[0]}?authSource={auth_source}\"")
        # We don't raise here to avoid crashing the whole app on import, 
        # but endpoints will fail if called.
        auth_db = None
else:
    print("pymongo not available - MongoDB features disabled")

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
    if not _BCRYPT_AVAILABLE:
        raise HTTPException(status_code=503, detail="bcrypt is not available. Please install bcrypt: pip install bcrypt")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    if not _BCRYPT_AVAILABLE:
        raise HTTPException(status_code=503, detail="bcrypt is not available. Please install bcrypt: pip install bcrypt")
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def create_session(user_id: str) -> dict:
    """Create a new session for the user."""
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=SESSION_EXPIRY_DAYS)
    
    session = {
        "_id": session_id,
        "userId": user_id,
        "createdAt": datetime.utcnow(),
        "expiresAt": expires_at
    }
    
    sessions_collection.insert_one(session)
    return session


def get_current_user(sid: Optional[str] = Cookie(None)) -> dict:
    """Dependency to get the current user from session cookie."""
    print(f"[AUTH] get_current_user called with sid: {sid}")
    
    if not sid:
        print("[AUTH] No session cookie found - returning 401")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if sessions_collection is None:
        print("[AUTH] Sessions collection unavailable")
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    session = sessions_collection.find_one({"_id": sid})
    if not session:
        print(f"[AUTH] Session {sid} not found in database")
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check if session is expired
    if session["expiresAt"] < datetime.utcnow():
        sessions_collection.delete_one({"_id": sid})
        print(f"[AUTH] Session {sid} expired")
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Check if MongoDB is connected
    if users_collection is None:
        print("[AUTH] MongoDB connection unavailable")
        raise HTTPException(
            status_code=503,
            detail="Authentication service is unavailable. MongoDB connection failed."
        )
    
    # Get user
    user = users_collection.find_one({"_id": ObjectId(session["userId"])})
    if not user:
        print(f"[AUTH] User {session['userId']} not found")
        raise HTTPException(status_code=401, detail="User not found")
    
    print(f"[AUTH] User authenticated: {user['email']}")
    return user


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister):
    """Register a new user."""
    print("\n\n")
    print("\nregister")
    
    # Check if MongoDB is connected
    if users_collection is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is unavailable. MongoDB connection failed."
        )

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
        "createdAt": datetime.utcnow()
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
    
    # Check if MongoDB is connected
    if users_collection is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is unavailable. MongoDB connection failed."
        )
    
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
    
    # Clear cookie
    response.delete_cookie(key="sid", path="/")
    
    return {"message": "Logout successful"}


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user)):
    """Get current user info."""
    return UserResponse(
        id=str(user["_id"]),
        email=user["email"],
        created_at=user["createdAt"]
    )
