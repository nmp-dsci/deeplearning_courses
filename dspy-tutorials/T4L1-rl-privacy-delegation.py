"""

https://dspy.ai/tutorials/gepa_trusted_monitor/

uv pip install control-arena==8.0.0 inspect-ai==0.3.137 plotly kaleido nbformat


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
from tavily import TavilyClient
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List

load_dotenv(Path.home() / ".env")

TRAIN_LIMIT = 100
VAL_LIMIT = 50
EVAL_LIMIT = 50


##
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("T3L3-gepa-facility")
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
