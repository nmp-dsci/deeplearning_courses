
import sys
import os

import anthropic


def run(question: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        system="You are a math assistant. Solve the given math problem and respond with only the final numeric answer. Do not include units, explanations, or working — just the number.",
        messages=[{"role": "user", "content": question}],
    )

    return response.content[0].text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llm_model.py '<question>'", file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1]
    result = run(question)
    print(result)
