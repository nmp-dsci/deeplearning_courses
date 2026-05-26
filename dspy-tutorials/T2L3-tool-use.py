
"""
https://dspy.ai/tutorials/tool_use/




"""



import random
import string
import os
from pathlib import Path
import dspy
from dotenv import load_dotenv
from pydantic import BaseModel
import orjson
from dspy.utils import download
from tavily import TavilyClient
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List

load_dotenv(Path.home() / ".env")



##
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("T2L3-tool_use")
mlflow.dspy.autolog()

## models


lm_mini = dspy.LM(model="deepseek/deepseek-v4-flash",
     api_key=os.environ["DEEPSEEK_API_KEY"],
      max_tokens=8000,        # cap each step                                                 
      temperature=0.7)        # break repetition loops  )
lm_max = dspy.LM(model="deepseek/deepseek-v4-pro", 
    api_key=os.environ["DEEPSEEK_API_KEY"],
      max_tokens=8000,        # cap each step                                                 
      temperature=0.7)        # break repetition loops  )

dspy.configure(lm=lm_mini) 

## 
from dspy.utils import download

download("https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/main/data/ToolHop.json")

data = orjson.loads(open("ToolHop.json").read())
random.Random(0).shuffle(data)

### setup ToolHop task with set of tools 

import re
import inspect

examples = []
fns2code = {}

def finish(answer: str):
    """Conclude the trajectory and return the final answer."""
    return answer

for datapoint in data:
    func_dict = {}
    for func_code in datapoint["functions"]:
        cleaned_code = func_code.rsplit("\n\n# Example usage", 1)[0]
        fn_name = re.search(r"^\s*def\s+([a-zA-Z0-9_]+)\s*\(", cleaned_code)
        fn_name = fn_name.group(1) if fn_name else None
        #
        if not fn_name:
            continue
        #
        local_vars = {}
        exec(cleaned_code, {}, local_vars)
        fn_obj = local_vars.get(fn_name)
        #
        if callable(fn_obj):
            func_dict[fn_name] = fn_obj
            assert fn_obj not in fns2code, f"Duplicate function found: {fn_name}"
            fns2code[fn_obj] = (fn_name, cleaned_code)
            #
    func_dict["finish"] = finish
    #
    example = dspy.Example(question=datapoint["question"], answer=datapoint["answer"], functions=func_dict)
    examples.append(example.with_inputs("question", "functions"))


examples[0].functions


trainset, devset, testset = examples[:100], examples[100:400], examples[400:]


## evaluate 
### additional data: (1) stricter definnition of ground truth and 5 steps only 

from func_timeout import func_set_timeout

def wrap_function_with_timeout(fn):
    @func_set_timeout(10)
    def wrapper(*args, **kwargs):
        try:
            return {"return_value": fn(*args, **kwargs), "errors": None}
        except Exception as e:
            return {"return_value": None, "errors": str(e)}
    #
    return wrapper

def fn_metadata(func):
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or "No docstring."
    return dict(function_name=func.__name__, arguments=str(signature), docstring=docstring)

def metric(example, pred, trace=None):
    gold = str(example.answer).rstrip(".0").replace(",", "").lower()
    pred = str(pred.answer).rstrip(".0").replace(",", "").lower()
    return pred == gold  # stricter than the original paper's metric!

evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=0, max_errors=999)

### Define Agent 


class Agent(dspy.Module):
    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        instructions = "For the final answer, produce short (not full sentence) answers in which you format dates as YYYY-MM-DD, names as Firstname Lastname, and numbers without leading 0s."
        signature = dspy.Signature('question, trajectory, functions -> next_selected_fn, args: dict[str, Any]', instructions)
        self.react = dspy.ChainOfThought(signature)
    #
    def forward(self, question, functions):
        tools = {fn_name: fn_metadata(fn) for fn_name, fn in functions.items()}
        trajectory = []
        # 
        for _ in range(self.max_steps):
            pred = self.react(question=question, trajectory=trajectory, functions=tools)
            selected_fn = pred.next_selected_fn.strip('"').strip("'")
            fn_output = wrap_function_with_timeout(functions[selected_fn])(**pred.args)
            trajectory.append(dict(reasoning=pred.reasoning, selected_fn=selected_fn, args=pred.args, **fn_output))
            #
            if selected_fn == "finish":
                break
        #
        return dspy.Prediction(answer=fn_output.get("return_value", ''), trajectory=trajectory)

## 
agent = Agent()
evaluate(agent)

##run optimizer
simba = dspy.SIMBA(metric=metric, max_steps=1, max_demos=1,bsize=5)
optimized_agent = simba.compile(agent, trainset=trainset[:5], seed=6793115)

evaluate(optimized_agent)
