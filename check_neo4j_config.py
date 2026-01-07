"""Check if Neo4j is configured in Agent .env file"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

env_file = Path(__file__).parent / ".env"

print("=" * 80)
print("Neo4j Configuration Check")
print("=" * 80)

if not env_file.exists():
    print("❌ .env file not found at:", env_file)
    print("\nPlease create .env file from .env.example")
    exit(1)

# Read .env file
neo4j_config = {}
with open(env_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            if 'NEO4J' in key.upper() or 'USE_NEO4J' in key.upper():
                neo4j_config[key] = value

print(f"\n📄 Checking: {env_file}")
print("\n" + "-" * 80)

if neo4j_config:
    print("✅ Neo4j configuration found:")
    for key, value in neo4j_config.items():
        # Mask sensitive values
        if 'PASSWORD' in key.upper() or 'SECRET' in key.upper() or 'KEY' in key.upper():
            display_value = "***" if value else "(not set)"
        else:
            display_value = value if value else "(not set)"
        print(f"   {key}={display_value}")
    
    # Check if Neo4j is enabled
    use_neo4j = neo4j_config.get('USE_NEO4J_SEARCH', '').lower() in ('1', 'true', 'yes')
    neo4j_url = neo4j_config.get('NEO4J_API_URL', '')
    
    print("\n" + "-" * 80)
    if use_neo4j or neo4j_url:
        print("✅ Neo4j is CONFIGURED and will be used")
        if neo4j_url:
            print(f"   API URL: {neo4j_url}")
        if use_neo4j:
            print("   USE_NEO4J_SEARCH: enabled")
        print("\n📋 To use Neo4j in Docker:")
        print("   1. Ensure NEO4J_API_URL is set (will be overridden to http://neo4j-backend:8000 in Docker)")
        print("   2. Start services: docker-compose -f docker-compose.dev.yml up -d")
    else:
        print("⚠️  Neo4j configuration found but not actively enabled")
        print("   Set USE_NEO4J_SEARCH=true or NEO4J_API_URL to enable")
else:
    print("❌ No Neo4j configuration found in .env")
    print("\n📋 To enable Neo4j, add to .env:")
    print("   NEO4J_API_URL=http://localhost:8000  # For local development")
    print("   # In Docker, this will be automatically set to http://neo4j-backend:8000")
    print("   USE_NEO4J_SEARCH=true  # Optional: enable Neo4j search")

print("\n" + "=" * 80)
