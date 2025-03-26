import yaml
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough

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

    Only use the following function do not create any function of your own:
    {function_description}

    Example:
    Task: Buy a Coffee and return
    Steps:
    1. move_forward(direction=0, speed=5)    #Move forward to the shop
    2. pay(amount=10)                        #Pay for the coffee
    3. pick_up(item="coffee")                #Pick up the coffee
    4. turn_back()                           #Turn back and return
    5. move_forward(direction=0, speed=5)    #Move forward to the starting point

    Only output the function call list.
    Do not explain your reasoning or include extra text.

    Now convert the following task:
    Task: {{task}}
    Steps:
    """
    return PromptTemplate(input_variables=["task"], template=template)



if __name__ == "__main__":
    function_pool = load_function_pool()
    prompt = build_prompt_template(function_pool)

    llm = OllamaLLM(base_url="192.168.0.15", model="mistral")
    chain = prompt | llm

    response = chain.invoke({"task": "Get a free coffee and return"})
    print(response)