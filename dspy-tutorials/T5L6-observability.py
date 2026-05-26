"""

https://dspy.ai/tutorials/observability/


uv add fastapi uvicorn

uv uvicorn T5L5-deployment:app --reload


"""

import random
import string
import os,json,requests
from pathlib import Path
import dspy
from dotenv import load_dotenv
from pydantic import BaseModel
import orjson
from dspy.utils import download
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List

load_dotenv(Path.home() / ".env")


##
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("T5L6-observability")
mlflow.dspy.autolog()

## models

lm_mini = dspy.LM(model="deepseek/deepseek-v4-flash",
     api_key=os.environ["DEEPSEEK_API_KEY"],
      max_tokens=64000,        # cap each step                                                 
      temperature=1)   
lm_max = dspy.LM(model="deepseek/deepseek-v4-pro", 
    api_key=os.environ["DEEPSEEK_API_KEY"],
      max_tokens=64000,        # cap each step                                                 
      temperature=1)   

dspy.configure(lm=lm_mini ) 

##
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def retrieve(query: str):
    """Retrieve top 3 relevant Wikipedia snippets via Tavily."""
    response = tavily.search(
        query=query,
        max_results=3,
        include_domains=["en.wikipedia.org"],
    )
    return [r["content"] for r in response["results"]]


agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)


prediction = agent(question="Which baseball team does Shohei Ohtani play for in June 2025?")
print(prediction.answer)


##
# Print out 5 LLM calls
dspy.inspect_history(n=5)



## building a custom solution 
from dspy.utils.callback import BaseCallback


# 1. Define a custom callback class that extends BaseCallback class
class AgentLoggingCallback(BaseCallback):
    # 2. Implement on_module_end handler to run a custom logging code.
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")
    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())

# 3. Set the callback to DSPy setting so it will be applied to program execution
dspy.configure(callbacks=[AgentLoggingCallback()])




