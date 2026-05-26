"""
https://dspy.ai/tutorials/program_of_thought/


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
mlflow.set_experiment("T1L10-program-of-thought")
mlflow.dspy.autolog()


## 1. using local SAndbox

sandbox = dspy.LocalSandbox()
expr = "value = 2*5 + 4\nvalue"
answer = sandbox.execute(expr)
answer

with dspy.PythonInterpreter() as interp:                                
    answer = interp.execute("value = 2*5 + 4\nvalue")           
    print(answer)  # 14     


## demonstrating ProgramOfThought

lm = dspy.LM(model="deepseek/deepseek-v4-flash", api_key=os.environ["DEEPSEEK_API_KEY"])
dspy.configure(lm=lm)


# simple modeule
class BasicGenerateAnswer(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

pot = dspy.ProgramOfThought(BasicGenerateAnswer)
problem = "2*5 + 4"
pot(question=problem).answer

dspy.inspect_history()


## Chain of thought  

problem = "Compute 12! / sum of prime numbers between 1 and 30."

cot = dspy.ChainOfThought(BasicGenerateAnswer)


## Answer vis COT : wrong  
cot(question=problem).answer

## Answer via POT: Correct 
pot(question=problem).answer

dspy.inspect_history()


#########
## 3. Computation of Contextual Reasoning 

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def search_wikipedia(query: str):
    response = tavily.search(
        query=query,
        max_results=3,
        include_domains=["en.wikipedia.org"],
    )
    return [r["content"] for r in response["results"]]


search_result = search_wikipedia(query="What is the square of the total sum of the atomic number of the metal ")



## 

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer the non-numerical components of a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

from dspy.dsp.utils import deduplicate

class MultiHopSearchWithPoT(dspy.Module):
    def __init__(self, num_hops):
        self.num_hops = num_hops
        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.generate_answer = dspy.ProgramOfThought(GenerateAnswer, max_iters=3)
    #
    def forward(self, question):
        context = []
        for _ in range(self.num_hops):
            query = self.generate_query(context=context, question=question).query
            context = deduplicate(context + search_wikipedia(query))
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

multi_hop_pot = MultiHopSearchWithPoT(num_hops=2)
question = (
    "What is the square of the total sum of the atomic number of the metal "
    "that makes up the gift from France to the United States in the late "
    "19th century and the sum of the number of digits in the first 10 prime numbers?"
)
multi_hop_pot(question=question).answer

dspy.inspect_history()



