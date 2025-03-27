import yaml

def validator(function_pool : dict[str, dict], llm_response : str) -> bool:
    functions = function_pool.keys()
    for function in functions:
        if function not in llm_response:
            return False
    return True