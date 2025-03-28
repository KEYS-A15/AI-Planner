import yaml
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from app.parser import FunctionPool

load_dotenv()

class PromptBuilder:
    def __init__(self, path: str = "app/function_pool.yaml"):
        self.function_pool = self.load_function_pool(path)
        self.function_description = self.function_pool_formatter(self.function_pool)

    def load_function_pool(self, path : str = "app/function_pool.yaml") -> dict:
        with open(path, "r") as file:
            return yaml.safe_load(file)
        
    def function_pool_formatter(self, function_pool : dict[str, dict]) -> str:
        output = ""
        for name, meta in function_pool.items():
            params = meta.get("parameters", {})
            params_str = ", ".join([f"{k}: {v}" for k, v in params.items()]) if params else "None"
            desc = meta["description"].strip().replace("\n", " ")
            output += f"- {name}({params_str}) : {desc}"
        return output

    def build_prompt_template(self) -> PromptTemplate:

        template = f"""
        You are an intelligent planning assistant. Your job is to convert high level natural language tasks into a sequence of low-level function calls using the available API below.

        Only use the following function:
        {self.function_description}

        Instructions:
        - Do not use any other functions apart from provided list, and never create new functions. 
        - Do not hallucinate or make up any functions such as move_right or move_left.
        - Any Directional movement should use move_forward function with direction and speed parameters.
        - To turn left or right do not turn back instead use move_forward directly

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