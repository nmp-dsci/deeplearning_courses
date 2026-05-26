"""

https://dspy.ai/tutorials/streaming/


 
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
mlflow.set_experiment("T5L8-streaming")
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


predict = dspy.Predict("question->answer")

# Enable streaming for the 'answer' field
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)

import asyncio

async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")
    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())




# Streaming Multiple Fields

class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("answer->simplified_answer")
    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer


predict = MyModule()
stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="answer"),
    dspy.streaming.StreamListener(signature_field_name="simplified_answer"),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)

async def read_output_stream():
    output = stream_predict(question="why did a chicken cross the kitchen?")
    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value

program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)


## Streaming the same pield multiple times 

def fetch_user_info(user_name: str):
    """Get user information like name, birthday, etc."""
    return {
        "name": user_name,
        "birthday": "2009-05-16",
    }


def get_sports_news(year: int):
    """Get sports news for a given year."""
    if year == 2009:
        return "Usane Bolt broke the world record in the 100m race."
    return None


react = dspy.ReAct("question->answer", tools=[fetch_user_info, get_sports_news])

stream_listeners = [
    # dspy.ReAct has a built-in output field called "next_thought".
    dspy.streaming.StreamListener(signature_field_name="next_thought", allow_reuse=True),
]
stream_react = dspy.streamify(react, stream_listeners=stream_listeners)


async def read_output_stream():
    output = stream_react(question="What sports news happened in the year Adam was born?")
    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value


print(asyncio.run(read_output_stream()))




