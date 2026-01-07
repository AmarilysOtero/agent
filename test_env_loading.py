from pathlib import Path
import os
from dotenv import load_dotenv

# Test the same path resolution as foundry_runner.py
p = Path(__file__).resolve().parents[0] / ".env"
print(f"Looking for .env at: {p}")
print(f"Exists: {p.exists()}")

if p.exists():
    load_dotenv(dotenv_path=p, override=True)
    print(f"AZURE_AI_PROJECT_ENDPOINT: {os.getenv('AZURE_AI_PROJECT_ENDPOINT', 'NOT SET')[:50]}...")
    print(f"AZURE_CLIENT_ID: {'SET' if os.getenv('AZURE_CLIENT_ID') else 'NOT SET'}")
    print(f"AZURE_CLIENT_SECRET: {'SET' if os.getenv('AZURE_CLIENT_SECRET') else 'NOT SET'}")
    print(f"AZURE_TENANT_ID: {'SET' if os.getenv('AZURE_TENANT_ID') else 'NOT SET'}")
else:
    print("ERROR: .env file not found!")



