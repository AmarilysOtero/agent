from __future__ import annotations
import asyncio
import logging
import os
import psycopg2

from .config import Settings
from .workflows.workflow_factory import run_sequential_goal
from .tools.text_to_api import api_main
from .tools.text_to_query import query_main
from .tools.interpreter import interpreter_main



logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


async def main() -> None:
   logging.basicConfig(level=logging.INFO)
   cfg = Settings.load()

   #prompt = "prepare a news script for John on latest news for the world?"
   #prompt = "What are Kevin skills you can find and his DXC Email address?"    
   #prompt = "tell me about Kevin ?"    
   #prompt = "what is the profit before income taxes in 2024?"  
   
   print("🌐 Welcome to your virtual assistant. How can I help you?")
   prompt = input("Describe your query (ex: 'show me all employees'): ")
   print("Prompt:", prompt)

   result = await run_sequential_goal(cfg, prompt)
   db_query_result = query_main(cfg, prompt)
   db_api_result = api_main(cfg, prompt)
   db_result = interpreter_main(cfg, prompt, result, db_query_result, db_api_result)
 
   #print(result)
   print()
   print(db_result)    

if __name__ == "__main__":
    asyncio.run(main())
