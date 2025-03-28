import json
import typer
import questionary
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.prompt_builder import PromptBuilder
from app.llm_integration import get_llm_chain
from app.parser import FunctionPool

cli = typer.Typer()

MODEL_OPTIONS = ["phi3", "phi4-mini", "mistral"]

@cli.command()
def run(
    task : str = typer.Option(..., prompt="Enter your task", help="Task to convert to function plan."),
    base_url : str = typer.Option("http://localhost:11434", help="Ollama base URL (default: http://localhost:11434)"),
    format: str = typer.Option("json", help="Expected model output format (default: json)")
):
    model = questionary.select(
        "Select a mini-LLM model: ",
        choices=MODEL_OPTIONS
    ).ask()
    
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_prompt_template()
    chain = get_llm_chain(prompt, model, base_url, format)

    response = chain.invoke({"task": task})
    print("Generated Response:\n", response, "\n\n")

    try:
        parsed = json.loads(response)
        validator = FunctionPool()
        invalid = validator.validator(parsed)
        if invalid:
            print(f"Invalid functions detected: {invalid}")
        else:
            print("All functions are valid.")
    except Exception as e:
        print(f"Error parsing response: {e}")

if __name__ == "__main__":
    cli()