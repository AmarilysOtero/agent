from __future__ import annotations
import json
import logging
from pydantic import BaseModel, Field, ValidationError
from src.news_reporter.foundry_runner import run_foundry_agent, run_foundry_agent_json
from src.news_reporter.tools.azure_search import hybrid_search
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
load_dotenv()

# Azure OpenAI
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_DB")
api_key = os.getenv("AZURE_OPENAI_KEY_DB")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_DB")

client = AzureOpenAI(
    azure_endpoint=endpoint, # type: ignore
    api_key=api_key,
    api_version="2024-02-15-preview"
)

# ---------- TRIAGE (Foundry) ----------

class IntentResult(BaseModel):
    intents: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    targets: list[str] = Field(default_factory=list)

class TriageAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, goal: str) -> IntentResult:
        content = f"Classify and return JSON only. User goal: {goal}"
        print("TriageAgent: using Foundry agent:", self._id)  # keep print
        raw = run_foundry_agent(self._id, content).strip()
        print("Triage raw:", raw)  # keep print
        try:
            data = json.loads(raw)
            return IntentResult(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Triage parse error: %s", e)
            return IntentResult(intents=["unknown"], confidence=0.0, rationale="parse_error")

# ---------- AI SEARCH (Foundry) ----------

class AiSearchAgent:
    """Search agent using Azure AI Search"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
        print("AiSearchAgent: using Foundry agent:", self._id)  # keep print
        results = hybrid_search(
            search_text=query,
            top_k=8,
            select=["file_name", "content", "url", "last_modified"],
            semantic=False
        )

        if not results:
            return "No results found in Azure AI Search."

        findings = []
        for res in results:
            content = (res.get("content") or "").replace("\n", " ")
            findings.append(f"- {res.get('file_name')}: {content[:300]}...")

        # print("AiSearchAgent list of sources/content\n\n" + "\n".join(findings))
        return "\n".join(findings)


# ---------- NEO4J GRAPHRAG SEARCH (Foundry) ----------

class Neo4jGraphRAGAgent:
    """Search agent using Neo4j GraphRAG retrieval"""
    
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, query: str) -> str:
        print("Neo4jGraphRAGAgent: using Foundry agent:", self._id)  # keep print
        from ..tools.neo4j_graphrag import graphrag_search
        
        results = graphrag_search(
            query=query,
            top_k=8,
            similarity_threshold=0.7
        )

        if not results:
            return "No results found in Neo4j GraphRAG."

        findings = []
        for res in results:
            text = res.get("text", "").replace("\n", " ")
            file_name = res.get("file_name", "Unknown")
            directory = res.get("directory_name", "")
            
            # Include GraphRAG metadata for explainability
            metadata = ""
            if "hybrid_score" in res:
                metadata = f" [score: {res['hybrid_score']:.2f}]"
            if "metadata" in res and res["metadata"].get("hop_count", 0) > 0:
                hops = res["metadata"]["hop_count"]
                metadata += f" [hops: {hops}]"
            
            # Format source info
            if directory:
                source_info = f"{directory}/{file_name}"
            else:
                source_info = file_name
            
            findings.append(f"- {source_info}: {text[:300]}...{metadata}")

        return "\n".join(findings)


# ---------- REPORTER (Foundry) ----------

class NewsReporterAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, topic: str, latest_news: str) -> str:
        content = (
            f"Topic: {topic}\n"
            f"Latest info:\n{latest_news}\n"
            # "Write a 60-90s news broadcast script."
            "Write a description about the information in the tone of a news reporter." 
        )
        print("NewsReporterAgent: using Foundry agent:", self._id)  # keep print
        return run_foundry_agent(self._id, content)

# ---------- REVIEWER (Foundry, strict JSON) ----------

class ReviewAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, topic: str, candidate_script: str) -> dict:
        """
        Foundry system prompt already defines the JSON schema. We still remind at user layer.
        Returns a dict with keys: decision, reason, suggested_changes, revised_script.
        Return ONLY STRICT JSON (no markdown, no prose) as per your schema.
        """
        prompt = (
            f"Topic: {topic}\n\n"
            f"Candidate script:\n{candidate_script}\n\n"
            # "Evaluate factual accuracy, clarity, neutral tone, explicit dates, and 60-90s length. "
            "Evaluate factual accuracy, relevance, and tone of a news reporter. " 
            "Return ONLY STRICT JSON (no markdown, no prose) as per your schema."
        )
        print("ReviewAgent: using Foundry agent:", self._id)  # keep print
        try:
            data = run_foundry_agent_json(
                self._id,
                prompt,
                system_hint="You are a reviewer that returns STRICT JSON only."
            )
            if not isinstance(data, dict) or "decision" not in data:
                raise ValueError("Invalid JSON shape from reviewer")
            decision = (data.get("decision") or "revise").lower()
            return {
                "decision": decision if decision in {"accept", "revise"} else "revise",
                "reason": data.get("reason", ""),
                "suggested_changes": data.get("suggested_changes", ""),
                "revised_script": data.get("revised_script", candidate_script),
            }
        except Exception as e:
            logger.error("Review parse error: %s", e)
            # Fail-safe: accept last script to avoid infinite loops
            return {
                "decision": "accept",
                "reason": "parse_error",
                "suggested_changes": "",
                "revised_script": candidate_script,
            }
            
class TextToQueryAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id
        
    def get_schema_message(self, csv_path: str):
        df = pd.read_csv(csv_path)
        message = "La base de datos tiene las siguientes tablas y columnas:\n"
        for table in df['table'].unique():
            cols = df[df['table'] == table]
            col_list = ", ".join([f"{row['column']} ({row['type']})" for _, row in cols.iterrows()])
            message += f"- {table}: {col_list}\n"
        return message

    def run(self, prompt: str) -> dict:
        schema_message = self.get_schema_message("schema.csv")
        system_message = f"""
            Eres un experto en SQL (PostgreSQL) que traduce preguntas en lenguaje natural a SQL válidos para la base de datos de Supabase.
            Este es el schema de la tabla {schema_message}.
            Usa comillas dobles (") para los nombres de las tablas.
            Devuelve SOLO el query SQL sin explicación, sin ```sql``` ni nada adicional.
            """
        response = client.chat.completions.create(
            model=deployment,  # type: ignore
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()  # type: ignore

class TextToAPIAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id
        
    def get_schema_message(self, csv_path: str):
        df = pd.read_csv(csv_path)
        message = "La base de datos tiene las siguientes tablas y columnas:\n"
        for table in df['table'].unique():
            cols = df[df['table'] == table]
            col_list = ", ".join([f"{row['column']} ({row['type']})" for _, row in cols.iterrows()])
            message += f"- {table}: {col_list}\n"
        return message

    def run(self, supabase_schema: str, prompt: str) -> dict:
        schema_message = self.get_schema_message("schema.csv")
        system_message = f"""
            Eres un experto en integraciones REST con Supabase.
            Convierte las solicitudes del usuario en llamadas API a la base de datos Supabase (PostgREST).
            Este es el schema de la tabla {schema_message}. Escribe la primera letra de la tabla en mayuscula.
            Devuelve **únicamente un JSON válido** con este formato exacto:

            {{
            "method": "GET" | "POST" | "PUT" | "DELETE",
            "endpoint": "/<tabla_o_ruta>",
            "params": {{ "columna": "valor" }},
            "body": {{ "columna": "valor" }}
            }}

            Reglas:
            - Usa {supabase_schema} como schema base.
            - Si el usuario pide ver o listar, usa GET.
            - Si el usuario pide agregar, usa POST.
            - Si el usuario pide actualizar, usa PUT.
            - Si el usuario pide borrar, usa DELETE.
            - No incluyas explicaciones ni comentarios.
            - Los filtros van en "params" usando la sintaxis de Supabase para hacer comparaciones tanto de tipo texto, int o fecha.
            - Siempre usa sintaxis PostgREST en params:
                - Para strings: name="eq.<valor>"
                - Para números: price="gt.<valor>" o "eq.<valor>"
            - Para filtros con fechas:
    - Siempre usa formato ISO 8601 (YYYY-MM-DD).
    - Para igualdad: campo="eq.<fecha>".
    - Para mayor/menor: campo="gt.<fecha>", "lt.<fecha>", "gte.<fecha>", "lte.<fecha>".
        - Si el usuario dice "antes de", usa lt.<fecha>.
        - Si el usuario dice "después de", usa gt.<fecha>.
        - Si el usuario da una fecha con texto ("9 de julio de 1997"), conviértela a ISO 8601 ("1997-07-09").
    - Devuelve solo el JSON, sin bloques de texto.
    """
        response = client.chat.completions.create(
            model=deployment, # type: ignore
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )

        try:
            return json.loads(response.choices[0].message.content.strip()) # type: ignore
        except json.JSONDecodeError:
            raise ValueError("El modelo no devolvió JSON válido.")
        
class APIResultInterpreter:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    def run(self, user_prompt: str, json_data: dict[str, str]) -> str:
        if "error" in json_data:
            return json_data["error"] # type: ignore

        # Si el resultado es una lista vacía
        if isinstance(json_data, list) and len(json_data) == 0:
            return "No se encontraron resultados." # type: ignore

        # Si el resultado es una lista de objetos (por ejemplo, registros)
        if isinstance(json_data, list) and len(json_data) > 0:
            # Convertimos los primeros 3 resultados a texto
            sample = json.dumps(json_data[:], ensure_ascii=False) # type: ignore
            instruction = f"""
                Eres un asistente que explica resultados de bases de datos en lenguaje natural.

                El usuario preguntó: "{user_prompt}"
                Aquí tienes una muestra del resultado en formato JSON:
                {sample}

                Describe brevemente el resultado de forma natural (en español), como si se lo contaras a un humano.
                Devuelve un string con la respuesta.
                """
            response = client.chat.completions.create(
                model=deployment, # type: ignore
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en explicar datos en lenguaje natural."},
                    {"role": "user", "content": instruction}
                ]
            )
            result = response.choices[0].message.content
            return result # type: ignore

        # Si es un diccionario con un mensaje
        if isinstance(json_data, dict) and "message" in json_data:
            return json_data["message"] # type: ignore

        # Si es un diccionario genérico
        return json.dumps(json_data, indent=4, ensure_ascii=False) # type: ignore

class QueryResultInterpreter:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    def run(self, user_prompt: str, columns: list, rows: list, limit: int) -> dict:
        if not rows:
            return "No se encontraron resultados." # type: ignore

        sample_rows = rows[:limit]
        sample_text = "\n".join(
            [", ".join([f"{col}: {val}" for col, val in zip(columns, row)]) for row in sample_rows]
        )

        more_text = ""
        if len(rows) > limit:
            more_text = f"\n...y {len(rows) - limit} registros más."

        instruction = f"""
            Eres un asistente que explica resultados de bases de datos en lenguaje natural.

            El usuario preguntó: "{user_prompt}"
            Estos son los primeros {limit} resultados:
            {sample_text}
            {more_text}

            Describe brevemente el resultado de forma natural (en español), como si se lo contaras a un humano.
            """

        response = client.chat.completions.create(
            model=deployment,  # type: ignore
            messages=[
                {"role": "system", "content": "Eres un asistente experto en explicar datos en lenguaje natural."},
                {"role": "user", "content": instruction}
            ]
        )
        return response.choices[0].message.content.strip()  # type: ignore

class ResultInterpreter:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    def run(self, user_prompt: str, result1: str, result2: str | dict, result3: str) -> str:     
        instruction = f"""
            Eres un asistente que analiza tres parrafos que tienen informacion sobre la misma consulta.
            La diferencia es que uno se obtuvo por medio de query, otro por medio de API call y otro con otro agente.
            Encargate de analizar los tres parrafos, unirlos y hacerlos uno.

            El usuario preguntó: "{user_prompt}"
            Estos son los parrafos {result1}, {result2} y {result3}
            Si uno de los parrafos no tiene resultado, tiene un error o esta vacio, omitelo y solo interpreta los que si tengan un resultado.
            
            Describe brevemente el resultado de forma natural (en español), como si se lo contaras a un humano.           
            """

        response = client.chat.completions.create(
            model=deployment,  # type: ignore
            messages=[
                {"role": "system", "content": "Eres un asistente experto en explicar datos en lenguaje natural."},
                {"role": "user", "content": instruction}
            ]
        )
        return response.choices[0].message.content.strip()  # type: ignore