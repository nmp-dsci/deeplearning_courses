"""

https://dspy.ai/tutorials/conversation_history/

"""


import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set — check ~/.env")

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(QA)
history = dspy.History(messages=[])

while True:
    question = input("Type your question, end conversation by typing 'finish': ")
    if question == "finish":
        break
    outputs = predict(question=question, history=history)
    print(f"\n{outputs.answer}\n")
    history.messages.append({"question": question, **outputs})

dspy.inspect_history()


### history in Few-shot Examples: attach examples
predict.demos.append(
    dspy.Example(
        question="What is the capital of France?",
        history=dspy.History(
            messages=[{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]
        ),
        answer="The capital of France is Paris.",
    )
)

predict(question="What is the capital of America?", history=dspy.History(messages=[]))
dspy.inspect_history()

