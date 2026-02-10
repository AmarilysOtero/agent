import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

mongo_keys = [
    'MONGO_AUTH_URL',
    'MONGO_AGENT_URL',
    'MONGO_WORKFLOW_URL'
]

print("--- MongoDB Connection Tests ---")
for key in mongo_keys:
    url = os.getenv(key)
    if url:
        try:
            client = MongoClient(url, serverSelectionTimeoutMS=3000)
            db_name = url.split('/')[-1].split('?')[0]
            db = client[db_name]
            db.command('ping')
            print(f"[SUCCESS] {key}: Connected to {db_name}, collections: {db.list_collection_names()}")
        except Exception as e:
            print(f"[FAIL] {key}: {e}")
    else:
        print(f"[NOT SET] {key}")

# Neo4j connection test
try:
    from neo4j import GraphDatabase
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')
    if uri and user and password:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            print(f"[SUCCESS] Neo4j: {uri} - {result.single()['test']}")
    else:
        print("[NOT SET] Neo4j connection variables")
except Exception as e:
    print(f"[FAIL] Neo4j: {e}")
