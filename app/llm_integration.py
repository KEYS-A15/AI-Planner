import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate

load_dotenv()

def get_llm_chain(prompt: PromptTemplate, model: str = "phi3", base_url: str = "http://localhost", format: str = "json") -> Runnable:
    llm = OllamaLLM(
        base_url=base_url,
        model=model,
        format=format,
        temperature=float(os.environ.get("TEMPERATURE", 0.5)),
        top_k=int(os.environ.get("TOP_K", 2)),
        top_p=float(os.environ.get("TOP_P", 0.5))
    )

    return prompt | llm