
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

##
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("T1L5-rag-agents")
mlflow.dspy.autolog()


## 
llama3b = dspy.LM('groq/llama-3.1-8b-instant', temperature=0.7)
sonnet = dspy.LM("anthropic/claude-sonnet-4-6", temperature=0.7)

dspy.configure(lm=sonnet)

## load dataset

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
trainset, devset, testset = hover[:100], hover[100:200], hover[650:]


example = trainset[0]

print("Claim:", example.claim)
print("Pages that must be retrieved:", example.titles)

## tool function 

DOCS = {}

def search(query: str, k: int) -> list[str]:
    res = tavily.search(
        query=query,
        max_results=k,
        search_depth="basic",
        include_domains=["en.wikipedia.org"],
    )["results"]
    results = []
    for r in res:
        title = r.get("title") or r["url"].rsplit("/", 1)[-1].replace("_", " ")
        text = r.get("content", "") or ""
        DOCS[title] = text
        results.append(f"{title} | {text}")
    return results


def search_wikipedia(query: str) -> list[str]:
    """Returns top-5 results and then the titles of the top-5 to top-30 results."""
    topK = search(query, 30)
    titles, topK = [f"`{x.split(' | ')[0]}`" for x in topK[5:30]], topK[:5]
    return topK + [f"Other retrieved pages have titles: {', '.join(titles)}."]


def lookup_wikipedia(title: str) -> str:
    """Returns the text of the Wikipedia page, if it exists."""
    if title in DOCS:
        return DOCS[title]
    # 
    results = [x for x in search(title, 10) if x.startswith(title + " | ")]
    if not results:
        return f"No Wikipedia page found for title: {title}"
    return results[0]


### 


instructions = "Find all Wikipedia titles relevant to verifying (or refuting) the claim."
signature = dspy.Signature("claim -> titles: list[str]", instructions)
react = dspy.ReAct(signature, tools=[search_wikipedia, lookup_wikipedia], max_iters=20)

## test run 
react(claim="David Gregory was born in 1625.").titles[:3]

## EVAL run 
def top5_recall(example, pred, trace=None):
    gold_titles = example.titles
    recall = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)
    # If we're "bootstrapping" for optimization, return True if and only if the recall is perfect.
    if trace is not None:
        return recall >= 1.0
    # If we're just doing inference, just measure the recall.
    return recall

evaluate = dspy.Evaluate(devset=devset[:20], metric=top5_recall, num_threads=16, display_progress=True, display_table=5)

evaluate(react)


### Optimize model 

kwargs = dict(teacher_settings=dict(lm=sonnet), prompt_model=sonnet, max_errors=999)

tp = dspy.MIPROv2(metric=top5_recall, auto="medium", num_threads=16, **kwargs)

optimized_react = tp.compile(react, trainset=trainset[:10], max_bootstrapped_demos=3, max_labeled_demos=0)

evaluate(optimized_react)


### save model 
optimized_react.save("optimized_react.json")

loaded_react = dspy.ReAct("claim -> titles: list[str]", tools=[search_wikipedia, lookup_wikipedia], max_iters=20)
loaded_react.load("optimized_react.json")

loaded_react(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles