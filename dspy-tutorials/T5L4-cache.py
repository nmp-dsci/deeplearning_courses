"""
https://dspy.ai/tutorials/cache/

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

dspy.configure(lm=lm_mini, track_usage=True) 

##

import dspy
import os
import time

predict = dspy.Predict("question->answer")


start = time.time()
result1 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result1.get_lm_usage()}")

start = time.time()
result2 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result2.get_lm_usage()}")









