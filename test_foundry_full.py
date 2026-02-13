import os
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

load_dotenv()

print("--- ENVIRONMENT CHECK ---")
for key in [
    'AZURE_AI_PROJECT_ENDPOINT',
    'AZURE_CLIENT_ID',
    'AZURE_CLIENT_SECRET',
    'AZURE_TENANT_ID',
    'MONGO_AUTH_URL',
    'MONGO_AGENT_URL',
    'MONGO_APP_PASS',
    'AZURE_OPENAI_ENDPOINT',
    'AZURE_OPENAI_API_KEY',
    'AZURE_OPENAI_CHAT_DEPLOYMENT',
]:
    print(f"{key}: {os.getenv(key)}")

print("\n--- FOUNDY CLIENT TEST ---")
try:
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    credential = DefaultAzureCredential()
    client = AIProjectClient(endpoint=endpoint, credential=credential)
    print("[SUCCESS] Connected to Foundry endpoint.")
    print("client dir:", dir(client))
    for resource in ['agents', 'threads', 'assistants', 'messages', 'runs']:
        if hasattr(client, resource):
            print(f"{resource} dir:", dir(getattr(client, resource)))
except Exception as e:
    print(f"[FAIL] Could not connect to Foundry: {e}")

print("\n--- MONGODB TEST ---")
try:
    from pymongo import MongoClient
    mongo_url = os.getenv('MONGO_AGENT_URL')
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=3000)
    db_name = mongo_url.split('/')[-1].split('?')[0]
    db = client[db_name]
    db.command('ping')
    print(f"[SUCCESS] Connected to MongoDB: {db_name}")
    print("Collections:", db.list_collection_names())
except Exception as e:
    print(f"[FAIL] Could not connect to MongoDB: {e}")
