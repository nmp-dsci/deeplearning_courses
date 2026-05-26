
import random
import string
import os
from pathlib import Path
import dspy
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path.home() / ".env")

##
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("T1L3-customizing-models")
mlflow.dspy.autolog()


## Define Multi-Stage program 

class QueryGenerator(dspy.Signature):
    """Generate a query based on question to fetch relevant context"""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

def search_wikipedia(query: str) -> list[str]:
    """Query ColBERT endpoint, which is a knowledge source based on wikipedia data"""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=1)
    return [x["text"] for x in results]


class RAG(dspy.Module):
    def __init__(self):
        self.query_generator = dspy.Predict(QueryGenerator)
        self.answer_generator = dspy.ChainOfThought("question,context->answer")
    #
    def forward(self, question, **kwargs):
        query = self.query_generator(question=question).query
        context = search_wikipedia(query)[0]
        return self.answer_generator(question=question, context=context).answer



### Run agetn

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
rag = RAG()



print(rag(question="Is Lebron James the basketball GOAT?"))








