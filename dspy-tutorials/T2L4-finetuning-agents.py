"""

https://dspy.ai/tutorials/games/



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


lm_mini = dspy.LM(model="deepseek/deepseek-v4-flash", api_key=os.environ["DEEPSEEK_API_KEY"])
lm_max = dspy.LM(model="deepseek/deepseek-v4-pro", api_key=os.environ["DEEPSEEK_API_KEY"])

dspy.configure(lm=lm_mini) 


## lms
import dspy

gpt4o_mini = dspy.LM('gpt-4o-mini-2024-07-18')
gpt4o = dspy.LM('openai/gpt-4o')
dspy.configure(experimental=True)


## data
from dspy.datasets.alfworld import AlfWorld

alfworld = AlfWorld()
trainset, devset = alfworld.trainset[:200], alfworld.devset[-200:]
len(trainset), len(devset)

## Example 
example = trainset[0]

with alfworld.POOL.session() as env:
    task, info = env.init(**example.inputs())

print(task)

## Defining the agent 

class Agent(dspy.Module):
    def __init__(self, max_iters=50, verbose=False):
        self.max_iters = max_iters
        self.verbose = verbose
        self.react = dspy.Predict("task, trajectory, possible_actions: list[str] -> action")

    def forward(self, idx):
        with alfworld.POOL.session() as env:
            trajectory = []
            task, info = env.init(idx)
            if self.verbose:
                print(f"Task: {task}")

            for _ in range(self.max_iters):
                trajectory_ = "\n".join(trajectory)
                possible_actions = info["admissible_commands"][0] + ["think: ${...thoughts...}"]
                prediction = self.react(task=task, trajectory=trajectory_, possible_actions=possible_actions)
                trajectory.append(f"> {prediction.action}")

                if prediction.action.startswith("think:"):
                    trajectory.append("OK.")
                    continue

                obs, reward, done, info = env.step(prediction.action)
                obs, reward, done = obs[0], reward[0], done[0]
                trajectory.append(obs)

                if self.verbose:
                    print("\n".join(trajectory[-2:]))

                if done:
                    break

        assert reward == int(info["won"][0]), (reward, info["won"][0])
        return dspy.Prediction(trajectory=trajectory, success=reward)











