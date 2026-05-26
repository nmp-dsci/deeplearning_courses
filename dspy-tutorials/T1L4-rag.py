
import random
import string
import os
from pathlib import Path
import dspy
from dotenv import load_dotenv
from pydantic import BaseModel
import orjson
from dspy.utils import download


load_dotenv(Path.home() / ".env")

##
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("T1L4-rag")
mlflow.dspy.autolog()

## 
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)


## Simple 
qa = dspy.Predict('question: str -> response: str')
response = qa(question="what are high memory and low memory on linux?")

print(response.response)

dspy.inspect_history(n=1)

## Chain of thought 
cot = dspy.ChainOfThought('question -> response')
cot(question="should curly braces appear on their own line?")


##############################
## RAG QA 

# Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")

with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [orjson.loads(line) for line in f]



data = [dspy.Example(**d).with_inputs('question') for d in data]

# Let's pick an `example` here from the data.
example = data[2]
example

## split into test / train 
import random

random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]

len(trainset), len(devset), len(testset)

####Evaluation in DSPy 
from dspy.evaluate import SemanticF1

# Instantiate the metric.
metric = SemanticF1(decompositional=True)

# Produce a prediction from our `cot` module, using the `example` above as input.
pred = cot(**example.inputs())

# Compute the metric score for the prediction.
score = metric(example, pred)

print(f"Question: \t {example.question}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {pred.response}\n")
print(f"Semantic F1 Score: {score.score:.2f}")

dspy.inspect_history(n=1)


#########################
# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=devset[:10], metric=metric, num_threads=5,
                         display_progress=True, display_table=2)

# Evaluate the Chain-of-Thought program.
evaluate(cot) # 38/ 100 


###########################
## RAG

download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")


max_characters = 6000  # for truncating >99th percentile of documents
topk_docs_to_retrieve = 5  # number of documents to retrieve per search query

with open("ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [orjson.loads(line)['text'][:max_characters] for line in f]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")

corpus = corpus[:20000]

embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)




class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')
    #
    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)

context = search("what are high memory and low memory on linux?").passages


rag = RAG()
rag(question="what are high memory and low memory on linux?")


dspy.inspect_history(n=1)

evaluate(RAG()) ## score: 52


### Run Prompt Optimisation 

tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)  # use fewer threads if your rate limit is small

optimized_rag = tp.compile(RAG(), trainset=trainset[:20],
                           max_bootstrapped_demos=2, max_labeled_demos=2)

## spot check: baseline RAG vs optimised RAG 

baseline = rag(question="cmd+tab does not work on hidden or minimized windows")
print(baseline.response)

pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
print(pred.response)

## Get Costs 
cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])  # in USD, as calculated by LiteLLM for certain providers


## Saving and loading 

optimized_rag.save("optimized_rag.json")

loaded_rag = RAG()
loaded_rag.load("optimized_rag.json")

loaded_rag(question="cmd+tab does not work on hidden or minimized windows")




