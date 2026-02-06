# tools/provision_search_ingestion.py
from __future__ import annotations
import os, json, requests

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
SEARCH_KEY      = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME      = os.environ.get("AZURE_SEARCH_INDEX", "pdf_chunks_1536")

# Blob (data source)
STORAGE_CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]  # same value as your AZURE_BLOB_CONN_STR
BLOB_CONTAINER   = os.environ.get("BLOB_CONTAINER_RAW", "raw")

# Azure OpenAI (for the Embedding skill)
AOAI_ENDPOINT     = os.environ.get("AZURE_OPENAI_EMBED_ENDPOINT", os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
AOAI_API_KEY      = os.environ.get("AZURE_OPENAI_EMBED_API_KEY", os.environ.get("AZURE_OPENAI_API_KEY", ""))
AOAI_DEPLOYMENT   = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
AOAI_API_VERSION  = os.environ.get(
    "AZURE_OPENAI_EMBED_API_VERSION",
    os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
)

# Names in Search
DATASOURCE_NAME = f"ds_blob_{BLOB_CONTAINER}"
SKILLSET_NAME   = "ss_pdf_chunk_embed"
INDEXER_NAME    = f"ixr_{INDEX_NAME}"

headers = {"Content-Type": "application/json", "api-key": SEARCH_KEY}

def upsert(url: str, payload: dict):
    r = requests.put(url, headers=headers, data=json.dumps(payload))
    if not r.ok:
        raise SystemExit(f"{url} → {r.status_code}\n{r.text}")
    print(f"✅ {url.split('/')[-1]} upserted")

def post(url: str, payload: dict | None = None):
    r = requests.post(url, headers=headers, data=(json.dumps(payload) if payload else None))
    if r.status_code not in (200, 202):
        raise SystemExit(f"{url} → {r.status_code}\n{r.text}")
    print(f"▶ {url.split('/')[-1]} posted")

def create_data_source():
    url = f"{SEARCH_ENDPOINT}/datasources/{DATASOURCE_NAME}?api-version=2024-07-01"
    payload = {
        "name": DATASOURCE_NAME,
        "type": "azureblob",
        "credentials": {"connectionString": STORAGE_CONN_STR},
        "container": {"name": BLOB_CONTAINER},
        "dataChangeDetectionPolicy": {"@odata.type": "#Microsoft.Azure.Search.HighWaterMarkChangeDetectionPolicy"},
    }
    upsert(url, payload)

def create_skillset():
    # DocumentExtraction → Split (~1000 chars) → AzureOpenAIEmbedding
    url = f"{SEARCH_ENDPOINT}/skillsets/{SKILLSET_NAME}?api-version=2024-07-01"
    payload = {
        "name": SKILLSET_NAME,
        "description": "Crack PDFs, split into chunks, and embed with Azure OpenAI",
        "skills": [
            {
                "@odata.type": "#Microsoft.Skills.Util.DocumentExtractionSkill",
                "name": "#crack",
                "context": "/document",
                "description": "Extract text/metadata from PDF",
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
                "name": "#split",
                "context": "/document",
                "textSplitMode": "pages",
                "maximumPageLength": 1000,
                "inputs": [{"name": "text", "source": "/document/content"}],
                "outputs": [{"name": "textItems", "targetName": "chunks"}],
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
                "name": "#embed",
                "description": "Embed each chunk to 1536-d vector",
                "context": "/document/chunks/*",
                "inputs": [{"name": "text", "source": "/document/chunks/*"}],
                "outputs": [{"name": "embedding", "targetName": "vector"}],
                "azureOpenAIParameters": {
                    "resourceUri": AOAI_ENDPOINT,
                    "apiKey": AOAI_API_KEY,
                    "deploymentId": AOAI_DEPLOYMENT,
                    "apiVersion": AOAI_API_VERSION,
                },
            },
        ],
    }
    upsert(url, payload)

def create_index():
    # Vector field: 1536 dims for text-embedding-3-small
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2024-07-01"
    payload = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "id",       "type": "Edm.String", "key": True,  "filterable": True},
            {"name": "fileName", "type": "Edm.String", "searchable": True},
            {"name": "blobUri",  "type": "Edm.String", "filterable": True},
            {"name": "chunk",    "type": "Edm.String", "searchable": True},  # chunk text
            {
                "name": "vector", "type": "Collection(Edm.Single)", "searchable": True,
                "vectorSearchDimensions": 1536, "vectorSearchProfileName": "vprof"
            },
            {"name": "timestamp", "type": "Edm.DateTimeOffset", "filterable": True, "sortable": True}
        ],
        "vectorSearch": {
            "profiles":   [{"name": "vprof", "algorithmConfigurationName": "hnsw"}],
            "algorithms": [{"name": "hnsw",  "kind": "hnsw"}],
        },
    }
    upsert(url, payload)

def create_indexer():
    url = f"{SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}?api-version=2024-07-01"
    payload = {
        "name": INDEXER_NAME,
        "dataSourceName": DATASOURCE_NAME,
        "targetIndexName": INDEX_NAME,
        "skillsetName": SKILLSET_NAME,
        "parameters": {
            "configuration": {
                "parsingMode": "default",
                "dataToExtract": "contentAndMetadata",
                "failOnUnsupportedContentType": False,
                "failOnUnprocessableDocument": False,
            }
        },
        "fieldMappings": [
            {"sourceFieldName": "metadata_storage_name", "targetFieldName": "fileName"},
            {"sourceFieldName": "metadata_storage_path", "targetFieldName": "blobUri"},
        ],
        "outputFieldMappings": [
            {"sourceFieldName": "/document/chunks/*",            "targetFieldName": "chunk"},
            {"sourceFieldName": "/document/chunks/*/vector",     "targetFieldName": "vector"},
        ],
        "schedule": {"interval": "PT2H"}  # optional: index every 2 hours
    }
    upsert(url, payload)

def run_indexer_once():
    # Kick off one run now (you can also wait for schedule)
    url = f"{SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}/run?api-version=2024-07-01"
    post(url)

if __name__ == "__main__":
    create_data_source()
    create_skillset()
    create_index()
    create_indexer()
    run_indexer_once()
    print(f"\nDone. Drop PDFs into '{BLOB_CONTAINER}'. Search will crack, chunk, embed, and index into '{INDEX_NAME}'.")
