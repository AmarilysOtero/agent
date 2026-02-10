import os
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

load_dotenv()

endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
credential = DefaultAzureCredential()
client = AIProjectClient(endpoint=endpoint, credential=credential)

print("client dir:", dir(client))
for resource in ['agents', 'threads', 'assistants', 'messages', 'runs']:
    if hasattr(client, resource):
        obj = getattr(client, resource)
        print(f"{resource} dir:", dir(obj))
