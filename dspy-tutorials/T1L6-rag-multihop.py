"""
https://dspy.ai/tutorials/multihop_search/


uv add bm25s PyStemmer "jax[cpu]" 

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


load_dotenv(Path.home() / ".env")

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


os.environ["GROQ_API_KEY"]



##
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("T1L6-rag-multihop")
mlflow.dspy.autolog()


lm = dspy.LM('groq/llama-3.1-8b-instant', max_tokens=3000)
gpt4o = dspy.LM('openai/gpt-4o', max_tokens=3000)


dspy.configure(lm=lm)


### Download data  & load  WIKIPEDIA Extract 
from dspy.utils import download

# download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
# !tar -xzvf wiki.abstracts.2017.tar.gz

import orjson
corpus = []

with open("wiki.abstracts.2017.jsonl") as f:
    for line in f:
        line = orjson.loads(line)
        corpus.append(f"{line['title']} | {' '.join(line['text'])}")

len(corpus)


###  index WIKI  
import bm25s
import Stemmer

stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)

###  eval dataset: HoVer dataset.
import random
from dspy.datasets import DataLoader

kwargs = dict(fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
hover = DataLoader().from_huggingface(dataset_name="vincentkoc/hover-parquet", split="train", trust_remote_code=True, **kwargs)

hpqa_ids = set()
hover = [
    dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:200], hover[200:500], hover[650:]

## visualise an example 

example = trainset[0]

print("Claim:", example.claim)
print("Pages that must be retrieved:", example.titles)

## Define function for search

def search(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return run


## Multi-hop program in DSPy, 

class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')
    #
    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []
        #
        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)
        #
        return dspy.Prediction(notes=notes, titles=list(set(titles)))


## Evaluation Metric
def top5_recall(example, pred, trace=None):
    gold_titles = example.titles
    recall = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)
    #
    # If we're "bootstrapping" for optimization, return True if and only if the recall is perfect.
    if trace is not None:
        return recall >= 1.0
    
    # If we're just doing inference, just measure the recall.
    return recall

evaluate = dspy.Evaluate(devset=devset[:5], metric=top5_recall, num_threads=16, display_progress=True, display_table=5)

evaluate(Hop())

## now lets optimize the two prompts inside the Hop() progrma

models = dict(prompt_model=gpt4o, teacher_settings=dict(lm=gpt4o))
tp = dspy.MIPROv2(metric=top5_recall, auto="medium", num_threads=16, **models)

kwargs = dict(minibatch_size=40, minibatch_full_eval_steps=4)
optimized = tp.compile(Hop(), trainset=trainset, max_bootstrapped_demos=4, max_labeled_demos=4, **kwargs)

evaluate(optimized)


optimized(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles

dspy.inspect_history(n=2)

optimized.save("optimized_hop.json")

loaded_program = Hop()
loaded_program.load("optimized_hop.json")

loaded_program(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles
