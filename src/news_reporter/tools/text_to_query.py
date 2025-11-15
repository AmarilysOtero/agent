import os
import sys
import pandas as pd
import psycopg2
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../agents")))
from src.news_reporter.config import Settings
from src.news_reporter.agents.agents import TextToQueryAgent, QueryResultInterpreter

# ==========================
# Cargar variables de entorno
# ==========================
load_dotenv()

# Azure OpenAI
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_DB")
api_key = os.getenv("AZURE_OPENAI_KEY_DB")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_DB")

client = AzureOpenAI(
    azure_endpoint=endpoint,  # type: ignore
    api_key=api_key,
    api_version="2024-02-15-preview"
)

# Supabase PostgreSQL (conexión directa)
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_PORT = os.getenv("SUPABASE_PORT")
SUPABASE_DB = os.getenv("SUPABASE_DB")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

# ==========================
# Conexión a PostgreSQL
# ==========================
def get_connection():
    return psycopg2.connect(
        host=SUPABASE_HOST,
        port=SUPABASE_PORT,
        database=SUPABASE_DB,
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
        sslmode="require"
    )

# Ejecutar query SQL arbitrario
# ==========================
def execute_sql_query(query):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)

        if cur.description is not None:  # SELECT
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return columns, rows
        else:  # INSERT, UPDATE, DELETE
            affected = cur.rowcount
            conn.commit()
            q_lower = query.lower()
            if q_lower.startswith("insert"):
                return f"Se agregaron {affected} auto(s) a la base de datos."
            elif q_lower.startswith("update"):
                return f"Se actualizaron {affected} auto(s) en la base de datos."
            elif q_lower.startswith("delete"):
                return f"Se eliminaron {affected} auto(s) de la base de datos."
            else:
                return f"Query ejecutado correctamente. Filas afectadas: {affected}"

    except Exception as e:
        return f"Error al ejecutar el query: {e}"
    finally:
        if conn:
            conn.close()

# ==========================
# Main
# ==========================
def query_main(cfg: Settings, prompt: str):
    agent = TextToQueryAgent(cfg.agent_id_text_to_query)
    interpreter = QueryResultInterpreter(cfg.agent_id_query_interpreter)

    # Generar SQL con el LLM
    query_spec = agent.run(prompt)  # puede ser str o dict

    # Detectar si es diccionario o string
    if isinstance(query_spec, dict):
        query = query_spec.get("query", "")
    elif isinstance(query_spec, str):
        query = query_spec
    else:
        return "❌ El agente no devolvió un query válido."

    # Limpiar query
    query = query.replace("`", "").strip()
    if query.endswith(";"):
        query = query[:-1]

    result = execute_sql_query(query)

    # Si el resultado es SELECT, pasar al interpreter
    if isinstance(result, tuple):
        columns, rows = result
        result_text = interpreter.run(prompt, columns, rows, limit=5)
    else:
        result_text = result

    return result_text
