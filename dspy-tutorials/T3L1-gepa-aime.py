"""

https://dspy.ai/tutorials/gepa_aime/



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
mlflow.set_experiment("T3L1-gepa-aime")
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
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in train_split
    ]
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)
    #
    test_split = load_dataset("MathArena/aime_2025")['train']
    test_split = [
        dspy.Example({
            "problem": x['problem'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in test_split
    ]
    #
    train_set = train_split[:int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]
    test_set = test_split # * 5
    #
    return train_set, val_set, test_set


train_set, val_set, test_set = init_dataset()

len(train_set), len(val_set), len(test_set)

print("Problem:")
print(train_set[0]['problem'])
print("\n\nSolution:")
print(train_set[0]['solution'])
print("\n\nAnswer:")
print(train_set[0]['answer'])


## define program 

class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(GenerateResponse)


## Evaluation metric

def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)

## Evaluation un optimzed Chain of Though 

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=32,
    display_table=True,
    display_progress=True
)

evaluate(program)


## Optimzze the evalaution metric for dspy.GEPA 
def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    written_solution = example.get('solution', '')
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."
        return dspy.Prediction(score=0, feedback=feedback_text)
    #
    score = int(correct_answer == llm_answer)
    #
    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."
    #       
    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."
    #
    return dspy.Prediction(score=score, feedback=feedback_text)



## Evaluation with GEPA 

from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True,
    reflection_minibatch_size=3,
    reflection_lm=lm_max
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)

evaluate(optimized_program)


