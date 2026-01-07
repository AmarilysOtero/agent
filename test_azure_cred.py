from azure.identity import DefaultAzureCredential
import os

print("Testing DefaultAzureCredential...")
print("Available env vars:")
print(f"  AZURE_CLIENT_ID: {bool(os.getenv('AZURE_CLIENT_ID'))}")
print(f"  AZURE_CLIENT_SECRET: {bool(os.getenv('AZURE_CLIENT_SECRET'))}")
print(f"  AZURE_TENANT_ID: {bool(os.getenv('AZURE_TENANT_ID'))}")
print(f"  AZURE_AI_PROJECT_ENDPOINT: {bool(os.getenv('AZURE_AI_PROJECT_ENDPOINT'))}")

try:
    cred = DefaultAzureCredential()
    print("Created DefaultAzureCredential object")
    # Try to get a token to see if it works
    token = cred.get_token("https://cognitiveservices.azure.com/.default")
    print(f"Successfully got token: {token.token[:20]}...")
except Exception as e:
    print(f"Failed to get token: {e}")



