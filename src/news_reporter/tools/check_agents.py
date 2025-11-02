# tools/check_agents.py
import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

conn = os.environ["AI_PROJECT_CONNECTION_STRING"]
client = AIProjectClient.from_connection_string(credential=DefaultAzureCredential(), conn_str=conn)

proj = client.projects.get_project()
print("Project:", proj.name)

agents = list(client.agents.list_agents())
print("Count:", len(agents))
for a in agents:
    print(a.id, a.name)
