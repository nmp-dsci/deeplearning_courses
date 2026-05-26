"""

https://dspy.ai/tutorials/mcp/


uv add "dspy[mcp]"

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
