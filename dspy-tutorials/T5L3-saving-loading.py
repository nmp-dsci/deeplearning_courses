"""
https://dspy.ai/tutorials/saving/

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

##


import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric


gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)

compiled_dspy_program.save("./dspy_program/program.json", save_program=False)

compiled_dspy_program.save("./dspy_program/program.pkl", save_program=False)

## load json 

loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/program.json")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)


## load pickle 

loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/program.pkl", allow_pickle=True)

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)









