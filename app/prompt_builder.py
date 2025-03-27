import yaml
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def load_function_pool(path : str = "app/function_pool.yaml") -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
def function_pool_formatter(function_pool : dict[str, dict]) -> str:
    output = ""
    for name, meta in function_pool.items():
        params = meta.get("parameters", {})
        params_str = ", ".join([f"{k}: {v}" for k, v in params.items()]) if params else "None"
        desc = meta["description"].strip().replace("\n", " ")
        output += f"- {name}({params_str}) : {desc}"
    return output

def build_prompt_template(function_pool : dict[str, dict]) -> PromptTemplate:
    function_description = function_pool_formatter(function_pool)

    template = f"""
    You are an intelligent planning assistant. Your job is to convert high level natural language tasks into a sequence of low-level function calls using the available API below.

    Only use the following function:
    {function_description}

    Do not use any other functions apart from provided list, and never create new functions.

    Example:
    Task: Buy a Coffee and return
    Steps:
    1. move_forward(direction=0, speed=5)    #Move forward to the shop
    2. pay(amount=10)                        #Pay for the coffee
    3. pick_up(item="coffee")                #Pick up the coffee
    4. turn_back()                           #Turn back and return
    5. move_forward(direction=0, speed=5)    #Move forward to the starting point

    Only output the function call. The output format should be in JSON format follows:
        "1" "move_forward(direction=0, speed=5)",
        "2" "pay(amount=10)",
        "3" "pick_up(item='coffee')",
        "4" "turn_back()",
        "5" "move_forward(direction=0, speed=5)"
        ... and so on
    Do not explain your reasoning or include extra text.

    Now convert the following task into a JSON sequence of function calls:
    Task: {{task}}
    Steps (JSON):
    ```json
    ```
    """
    return PromptTemplate(input_variables=["task"], template=template)


if __name__ == "__main__":
    function_pool = load_function_pool()
    prompt = build_prompt_template(function_pool)

    llm = OllamaLLM(base_url="192.168.0.15", model="phi3", format="json", temperature=float(os.environ["TEMPERATURE"]), top_k=2, top_p=0.25)
    chain = prompt | llm

    response = chain.invoke({"task": "Buy a free coffee and go to the park to the right"})
    print(response, "\n\n")