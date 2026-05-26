"""

https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/


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
from tavily import TavilyClient
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List

load_dotenv(Path.home() / ".env")



##
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("T3L2-gepa-facility")
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

# Load dataset


def init_dataset():
    # Load from the url
    url = "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json"
    dataset = json.loads(requests.get(url).text)
    dspy_dataset = [
        dspy.Example({
            "message": d['fields']['input'],
            "answer": d['answer'],
        }).with_inputs("message")
        for d in dataset
    ]
    random.Random(0).shuffle(dspy_dataset)
    train_set = dspy_dataset[:int(len(dspy_dataset) * 0.33)]
    val_set = dspy_dataset[int(len(dspy_dataset) * 0.33):int(len(dspy_dataset) * 0.66)]
    test_set = dspy_dataset[int(len(dspy_dataset) * 0.66):]
    #
    return train_set, val_set, test_set

train_set, val_set, test_set = init_dataset()

len(train_set), len(val_set), len(test_set)

## view example
print("Input Message:")
print(train_set[0]['message'])

print("\n\nGold Answer:")
for k, v in json.loads(train_set[0]['answer']).items():
    print(f"{k}: {v}")

## Define program 



from typing import List, Literal


class FacilitySupportAnalyzerUrgency(dspy.Signature):
    """
    Read the provided message and determine the urgency.
    """
    message: str = dspy.InputField()
    urgency: Literal['low', 'medium', 'high'] = dspy.OutputField()

class FacilitySupportAnalyzerSentiment(dspy.Signature):
    """
    Read the provided message and determine the sentiment.
    """
    message: str = dspy.InputField()
    sentiment: Literal['positive', 'neutral', 'negative'] = dspy.OutputField()

class FacilitySupportAnalyzerCategories(dspy.Signature):
    """
    Read the provided message and determine the set of categories applicable to the message.
    """
    message: str = dspy.InputField()
    categories: List[Literal["emergency_repair_services", "routine_maintenance_requests", "quality_and_safety_concerns", "specialized_cleaning_services", "general_inquiries", "sustainability_and_environmental_practices", "training_and_support_requests", "cleaning_services_scheduling", "customer_feedback_and_complaints", "facility_management_issues"]] = dspy.OutputField()

class FacilitySupportAnalyzerMM(dspy.Module):
    def __init__(self):
        self.urgency_module = dspy.ChainOfThought(FacilitySupportAnalyzerUrgency)
        self.sentiment_module = dspy.ChainOfThought(FacilitySupportAnalyzerSentiment)
        self.categories_module = dspy.ChainOfThought(FacilitySupportAnalyzerCategories)
    #
    def forward(self, message: str):
        urgency = self.urgency_module(message=message)
        sentiment = self.sentiment_module(message=message)
        categories = self.categories_module(message=message)
        #
        return dspy.Prediction(
            urgency=urgency.urgency,
            sentiment=sentiment.sentiment,
            categories=categories.categories
        )

program = FacilitySupportAnalyzerMM()

## Evaluation Metric 

def score_urgency(gold_urgency, pred_urgency):
    """
    Compute score for the urgency module.
    """
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    return score

def score_sentiment(gold_sentiment, pred_sentiment):
    """
    Compute score for the sentiment module.
    """
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    return score

def score_categories(gold_categories, pred_categories):
    """
    Compute score for the categories module.
    Uses the same match/mismatch logic as category accuracy in the score.
    """
    correct = 0
    for k, v in gold_categories.items():
        if v and k in pred_categories:
            correct += 1
        elif not v and k not in pred_categories:
            correct += 1
    score = correct / len(gold_categories)
    return score

def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Computes a score based on agreement between prediction and gold standard for categories, sentiment, and urgency.
    Returns the score (float).
    """
    # Parse gold standard from example
    gold = json.loads(example['answer'])
    # Compute scores for all modules
    score_urgency_val = score_urgency(gold['urgency'], pred.urgency)
    score_sentiment_val = score_sentiment(gold['sentiment'], pred.sentiment)
    score_categories_val = score_categories(gold['categories'], pred.categories)
    # Overall score: average of the three accuracies
    total = (score_urgency_val + score_sentiment_val + score_categories_val) / 3
    return total

## run evaluation 

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=32,
    display_table=True,
    display_progress=True
)

evaluate(program)

## Optimization with GEPA 

import json
import dspy

def feedback_urgency(gold_urgency, pred_urgency):
    """
    Generate feedback for the urgency module.
    """
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    if gold_urgency == pred_urgency:
        feedback = f"You correctly classified the urgency of the message as `{gold_urgency}`. This message is indeed of `{gold_urgency}` urgency."
    else:
        feedback = f"You incorrectly classified the urgency of the message as `{pred_urgency}`. The correct urgency is `{gold_urgency}`. Think about how you could have reasoned to get the correct urgency label."
    return feedback, score

def feedback_sentiment(gold_sentiment, pred_sentiment):
    """
    Generate feedback for the sentiment module.
    """
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    if gold_sentiment == pred_sentiment:
        feedback = f"You correctly classified the sentiment of the message as `{gold_sentiment}`. This message is indeed `{gold_sentiment}`."
    else:
        feedback = f"You incorrectly classified the sentiment of the message as `{pred_sentiment}`. The correct sentiment is `{gold_sentiment}`. Think about how you could have reasoned to get the correct sentiment label."
    return feedback, score

def feedback_categories(gold_categories, pred_categories):
    """
    Generate feedback for the categories module.
    Uses the same match/mismatch logic as category accuracy in the score.
    """
    correctly_included = [k for k, v in gold_categories.items() if v and k in pred_categories]
    incorrectly_included = [k for k, v in gold_categories.items() if not v and k in pred_categories]
    incorrectly_excluded = [k for k, v in gold_categories.items() if v and k not in pred_categories]
    correctly_excluded = [k for k, v in gold_categories.items() if not v and k not in pred_categories]  # For completeness in accuracy check
    # Recompute category accuracy (matches score logic)
    score = (len(correctly_included) + len(correctly_excluded)) / len(gold_categories)
    if score == 1.0:
        fb_text = f"The category classification is perfect. You correctly identified that the message falls under the following categories: `{repr(correctly_included)}`."
    else:
        fb_text = f"The category classification is not perfect. You correctly identified that the message falls under the following categories: `{repr(correctly_included)}`.\n"
        if incorrectly_included:
            fb_text += f"However, you incorrectly identified that the message falls under the following categories: `{repr(incorrectly_included)}`. The message DOES NOT fall under these categories.\n"
        if incorrectly_excluded:
            prefix = "Additionally, " if incorrectly_included else "However, "
            fb_text += f"{prefix}you didn't identify the following categories that the message actually falls under: `{repr(incorrectly_excluded)}`.\n"
        fb_text += "Think about how you could have reasoned to get the correct category labels."
    return fb_text, score

def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Computes a score based on agreement between prediction and gold standard for categories, sentiment, and urgency.
    Optionally provides feedback text for a specific predictor module, using the same comparison logic as the score.
    Returns a dspy.Prediction with score (float) and feedback (str).
    """
    # Parse gold standard from example
    gold = json.loads(example['answer'])
    # Compute feedback and scores for all modules
    fb_urgency, score_urgency = feedback_urgency(gold['urgency'], pred.urgency)
    fb_sentiment, score_sentiment = feedback_sentiment(gold['sentiment'], pred.sentiment)
    fb_categories, score_categories = feedback_categories(gold['categories'], pred.categories)
    # Overall score: average of the three accuracies
    total = (score_urgency + score_sentiment + score_categories) / 3
    if pred_name is None:
        return total
    elif pred_name == 'urgency_module.predict':
        feedback = fb_urgency
    elif pred_name == 'sentiment_module.predict':
        feedback = fb_sentiment
    elif pred_name == 'categories_module.predict':
        feedback = fb_categories
    #
    return dspy.Prediction(score=total, feedback=feedback)

## run GEPA optimizer

from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light", # <-- We will use a light budget for this tutorial. However, we typically recommend using auto="heavy" for optimized performance!
    num_threads=32,
    track_stats=True,
    use_merge=False,
    reflection_lm=lm_max
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)

### 
for name, pred in program.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")

### 
for name, pred in optimized_program.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")

evaluate(optimized_program)

###

def dag_to_dot(parent_program_for_candidate, dominator_program_ids, best_program_idx, full_eval_scores):
    dot_lines = [
        "digraph G {",
        "    node [style=filled, shape=circle, fontsize=50];"
    ]
    n = len(parent_program_for_candidate)
    # Set up nodes with colors and scores in labels
    for idx in range(n):
        score = full_eval_scores[idx]
        label = f"{idx}\\n({score:.2f})"
        if idx == best_program_idx:
            dot_lines.append(f'    {idx} [label="{label}", fillcolor=cyan, fontcolor=black];')
        elif idx in dominator_program_ids:
            dot_lines.append(f'    {idx} [label="{label}", fillcolor=orange, fontcolor=black];')
        else:
            dot_lines.append(f'    {idx} [label="{label}"];')
    # Set up edges
    for child, parents in enumerate(parent_program_for_candidate):
        for parent in parents:
            if parent is not None:
                dot_lines.append(f'    {parent} -> {child};')
    #
    dot_lines.append("}")
    return "\n".join(dot_lines)

from gepa.gepa_utils import find_dominator_programs
pareto_front_programs = find_dominator_programs(optimized_program.detailed_results.per_val_instance_best_candidates, optimized_program.detailed_results.val_aggregate_scores)

print(dag_to_dot(
    optimized_program.detailed_results.parents,
    pareto_front_programs,
    optimized_program.detailed_results.best_idx,
    optimized_program.detailed_results.val_aggregate_scores
))

