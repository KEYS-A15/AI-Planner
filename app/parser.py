import yaml
import re

class FunctionPool:
    def __init__(self, path: str = "app/function_pool.yaml"):
        self.function_pool = self.load_function_pool(path)

    def load_function_pool(self, path : str = "app/function_pool.yaml") -> dict:
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def output_function_extraction(self, llm_response: dict) -> list:
        function_calls = ", ".join(list(llm_response.values()))
        pattern = r"(\w+)\(.*?\)"

        return [f"{func}" for func in re.findall(pattern, function_calls)]

    def validator(self, llm_response) -> list:

        functions_allowed = list(self.function_pool.keys())
        used_functions = self.output_function_extraction(llm_response=llm_response)
        invalid_functions = [func for func in used_functions if func not in functions_allowed]
        if invalid_functions:
            return invalid_functions
        else:
            return None