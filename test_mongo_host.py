#!/usr/bin/env python
"""Test MongoDB connection from host"""
import sys
import os

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pymongo import MongoClient
    
    # Test the exact connection string from config
    from src.news_reporter.config import Settings
    settings = Settings.load()
    conn_str = settings.auth_api_url
    
    if not conn_str:
        print("❌ MONGO_AUTH_URL is not set")
        sys.exit(1)
    
    print(f"Testing connection from HOST machine:")
    print(f"  Connection string: {conn_str.split('@')[0]}@***")
    print()
    
    # First, test with root user to see if it's a general connection issue
    print("Testing root user connection from host...")
    try:
        root_client = MongoClient(
            'mongodb://root:rootpassword@127.0.0.1:27017/admin',
            serverSelectionTimeoutMS=5000
        )
        root_db = root_client.admin
        root_result = root_db.command('ping')
        print(f"[OK] Root user connection works from host!")
    except Exception as e:
        print(f"[ERROR] Root user also failed: {e}")
        print("This suggests a general network/connection issue, not user-specific.")
    print()
    
    # Try connection with explicit parameters
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(conn_str)
    
    # Try both methods: connection string and explicit parameters
    print("Attempting connection method 1: Connection string...")
    try:
        client = MongoClient(conn_str, serverSelectionTimeoutMS=5000)
        db = client.get_database()
        result = db.command('ping')
        print(f"[OK] Method 1 (connection string) works!")
    except Exception as e1:
        print(f"[ERROR] Method 1 failed: {e1}")
        print()
        print("Attempting connection method 2: Explicit parameters...")
        # Try with explicit parameters
        auth_source = parse_qs(parsed.query).get('authSource', ['auth_db'])[0]
        db_name = parsed.path.lstrip('/').split('?')[0] if parsed.path else 'auth_db'
        
        client = MongoClient(
            host=parsed.hostname or '127.0.0.1',
            port=parsed.port or 27017,
            username=parsed.username,
            password=parsed.password,
            authSource=auth_source,
            serverSelectionTimeoutMS=5000
        )
        db = client[db_name]
        result = db.command('ping')
        print(f"[OK] Method 2 (explicit parameters) works!")
    
    # Test ping
    result = db.command('ping')
    print(f"[OK] SUCCESS! Connection works from host.")
    print(f"   Database: {db.name}")
    print(f"   Ping result: {result}")
    sys.exit(0)
    
except ImportError as e:
    print(f"[ERROR] pymongo not installed: {e}")
    print("   Install with: pip install pymongo")
    print("   Or activate venv: .venv\\Scripts\\activate")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    print()
    print("Troubleshooting:")
    print("1. Verify MongoDB port is accessible from host:")
    print("   Test: telnet 127.0.0.1 27017")
    print("2. Check if MongoDB is bound to 0.0.0.0 (should be via Docker port mapping)")
    print("3. Try connecting with mongosh from host (if installed):")
    print(f"   mongosh \"{conn_str}\"")
    sys.exit(1)





