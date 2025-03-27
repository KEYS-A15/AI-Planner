from app.llm_integration import get_llm_chain

def plan_task(task: str, model: str = "phi3", base_url: str = "http://localhost", format: str = "json") -> str:
    chain = get_llm_chain(model, base_url, format)
    response = chain.invoke({"task": task})
    return response