from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # or "azure"

def get_llm():
    if LLM_BACKEND == "azure":
        return ChatOpenAI(
            deployment_name="gpt-4-8k",
            openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.2
        )
    else:
        return ChatOllama(
            model="mistral",
            base_url="http://192.168.1.158:11434",
            temperature=0.2
        )
