import os
import sys
from openai import AzureOpenAI
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../agents")))
from src.news_reporter.config import Settings
from src.news_reporter.agents.agents import ResultInterpreter

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

# ==========================
# Main
# ==========================
def interpreter_main(cfg: Settings,  prompt: str, result1: str, result2: str | dict, result3: str):
    interpreter = ResultInterpreter(cfg.agent_id_interpreter)

    # Generar SQL con el LLM
    query_spec = interpreter.run(prompt, result1, result2, result3)  # puede ser str o dict
    return query_spec
