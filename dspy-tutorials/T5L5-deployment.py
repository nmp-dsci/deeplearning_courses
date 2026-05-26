"""

https://dspy.ai/tutorials/deployment/


uv add fastapi uvicorn

uv uvicorn T5L5-deployment:app --reload


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

dspy.configure(lm=lm_mini ) 
dspy_program = dspy.ChainOfThought("question -> answer")

##


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dspy

app = FastAPI(
    title="DSPy Program API",
    description="A simple API serving a DSPy Chain of Thought program",
    version="1.0.0"
)

# Define request model for better documentation and validation
class Question(BaseModel):
    text: str

# Configure your language model and 'asyncify' your DSPy program.
dspy.configure(lm=lm_mini, async_max_workers=4) # default is 8
dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


## Streaming 
# streaming_dspy_program = dspy.streamify(dspy_program)

# @app.post("/predict/stream")
# async def stream(question: Question):
#     async def generate():
#         async for value in streaming_dspy_program(question=question.text):
#             if isinstance(value, dspy.Prediction):
#                 data = {"prediction": value.labels().toDict()}
#             elif isinstance(value, litellm.ModelResponse):
#                 data = {"chunk": value.json()}
#             yield f"data: {orjson.dumps(data).decode()}\n\n"
#         yield "data: [DONE]\n\n"
#     return StreamingResponse(generate(), media_type="text/event-stream")

# Since you're often going to want to stream the result of a DSPy program as server-sent events,
# we've included a helper function for that, which is equivalent to the code above.

# from dspy.utils.streaming import streaming_response

# @app.post("/predict/stream")
# async def stream(question: Question):
#     stream = streaming_dspy_program(question=question.text)
#     return StreamingResponse(streaming_response(stream), media_type="text/event-stream")




