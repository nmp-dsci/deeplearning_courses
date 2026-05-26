
"""
https://dspy.ai/tutorials/classification_finetuning/

uv pip install "sglang[all]>=0.4.4.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

DONT HAVE GPU so skipped

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

