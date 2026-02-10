import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# List of important Foundry and MongoDB environment variables to check
keys = [
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
]

print("--- Environment Variable Check ---")
for key in keys:
    value = os.getenv(key)
    if value:
        print(f"{key}: {value}")
    else:
        print(f"{key}: [NOT SET]")

print("\nIf any are [NOT SET], check your .env configuration.")
