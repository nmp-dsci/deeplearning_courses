
"""
https://dspy.ai/tutorials/math/

uv add "git+https://github.com/hendrycks/math.git"


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
mlflow.set_experiment("T2L1-math-reasoning")
mlflow.dspy.autolog()

## models


lm_mini = dspy.LM(model="deepseek/deepseek-v4-flash", api_key=os.environ["DEEPSEEK_API_KEY"])
lm_max = dspy.LM(model="deepseek/deepseek-v4-pro", api_key=os.environ["DEEPSEEK_API_KEY"])

dspy.configure(lm=lm_mini) 


## load data 
from dspy.datasets import MATH

dataset = MATH(subset='algebra')
print(len(dataset.train), len(dataset.dev))


example = dataset.train[0]
print("Question:", example.question)
print("Answer:", example.answer)

## define module 


module = dspy.ChainOfThought("question -> answer")
module(question=example.question)

### setup evaluator 

THREADS = 24
kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

## evaluate module
evaluate(module)


## run optimation with evaluation 
kwargs = dict(num_threads=THREADS, teacher_settings=dict(lm=lm_max), prompt_model=lm_mini)
optimizer = dspy.MIPROv2(metric=dataset.metric, auto="medium", **kwargs)

kwargs = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_module = optimizer.compile(module, trainset=dataset.train, **kwargs)







