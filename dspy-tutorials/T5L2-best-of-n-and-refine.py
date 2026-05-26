"""
https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/


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
mlflow.set_experiment("T5L2-best-of-n-and-refine")
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

dspy.configure(lm=lm_mini) 

## define program

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # Brussels


## error handling 

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)

best_of_3(question="What is the capital of Belgium?")
# raises an error after the first failure


## Refine 


def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = refine(question="What is the capital of Belgium?")
print(result.answer)  # Brussels

##

import dspy

class FactualityJudge(dspy.Signature):
    """Determine if a statement is factually accurate."""
    statement: str = dspy.InputField()
    is_factual: bool = dspy.OutputField()

factuality_judge = dspy.ChainOfThought(FactualityJudge)

def factuality_reward(args, pred: dspy.Prediction) -> float:
    statement = pred.answer    
    result = factuality_judge(statement)    
    return 1.0 if result.is_factual else 0.0

refined_qa = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=factuality_reward,
    threshold=1.0
)

result = refined_qa(question="Tell me about Belgium's capital city.")
print(result.answer)


