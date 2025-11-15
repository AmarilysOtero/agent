import os
import sys
import json
import requests
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../agents")))
from src.news_reporter.config import Settings
from src.news_reporter.agents.agents import TextToAPIAgent, APIResultInterpreter

# ==========================
# Cargar variables de entorno
# ==========================
load_dotenv()

# Supabase REST API
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SCHEMA = os.getenv("SUPABASE_SCHEMA", "public")

# ==========================
# Ejecutar llamada a la API de Supabase
# ==========================
def execute_api_call(api_spec):
    url = f"{SUPABASE_URL}/rest/v1{api_spec['endpoint']}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    method = api_spec["method"].upper()
    params = api_spec.get("params", {})
    body = api_spec.get("body", {})

    try:
        response = requests.request(method, url, headers=headers, params=params, json=body)
        if response.status_code in [200, 201, 204]:
            try:
                return response.json()
            except:
                return {"message": f"✅ Llamada {method} ejecutada correctamente (sin contenido)."}
        else:
            return {"error": f"❌ Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Error al ejecutar la llamada API: {e}"}

# ==========================
# Main
# ==========================
def api_main(cfg: Settings, prompt: str) -> str:
    agent = TextToAPIAgent(cfg.agent_id_text_to_api)
    interpreter = APIResultInterpreter(cfg.agent_id_api_interpreter)

    api_spec = agent.run(SUPABASE_SCHEMA, prompt)
    json_result = execute_api_call(api_spec)
    result_text = interpreter.run(prompt, json_result)
    
    return result_text