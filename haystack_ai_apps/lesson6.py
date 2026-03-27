
from dotenv import load_dotenv
load_dotenv()

import pprint
import gradio as gr
from typing import List
from haystack import component, Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker



## Create RAG Pipeline as a Function 

template = """
Answer the questions based on the given context.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
Question: {{ question }}
Answer:
"""

rag_pipe = Pipeline()
rag_pipe.add_component("prompt_builder", PromptBuilder(template=template))
rag_pipe.add_component("llm", OpenAIGenerator())

rag_pipe.connect("prompt_builder", "llm")


def rag_pipeline_func(query: str):
    '''
    "Get information about where people live"
    The query to use in the search. Infer this from the user's message. It should be a question or a statement
    '''
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
        Document(content="My name is Marta and I live in Madrid."),
        Document(content="My name is Harry and I live in London."),
    ]
    result = rag_pipe.run({"prompt_builder": {"question": query, 
                                              "documents": documents}})
    return {"reply": result["llm"]["replies"][0]}


## Create a Weather Function
WEATHER_INFO = {
    "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
    "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
    "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    "Madrid": {"weather": "sunny", "temperature": 10, "unit": "celsius"},
    "London": {"weather": "cloudy", "temperature": 9, "unit": "celsius"},
}

def get_current_weather(location: str):
    '''
    Get the current weather
    "location": {"type": "string", "description": "The city"}
    '''
    if location in WEATHER_INFO:
        return WEATHER_INFO[location]
    else:
        return {"weather": "sunny", "temperature": 70, "unit": "fahrenheit"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_pipeline_func",
            "description": "Get information about where people live",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to use in the search. Infer this from the user's message. It should be a question or a statement",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city"}
                },
                "required": ["location"],
            },
        },
    },
]

## Create an OpenAIChatGenerator 
# messages can come from the following roles: system, assitant, user, function 

chat_generator = OpenAIChatGenerator(model="gpt-3.5-turbo", generation_kwargs={'tools': tools})
replies = chat_generator.run(messages=[ChatMessage.from_user("Where does Mark live?")])

print(replies['replies'][0])

## Calling the function 

from haystack.tools import create_tool_from_function

rag_tool = create_tool_from_function(rag_pipeline_func)
weather_tool = create_tool_from_function(get_current_weather)

tools = [rag_tool, weather_tool]

function_caller = ToolInvoker(tools=tools)

results = function_caller.run(messages=replies['replies'])
pprint.pprint(results["tool_messages"][0])

## Create a Chat Agent with Tools

from haystack.components.agents import Agent

SYSTEM_PROMPT = """If needed, break down the user's question into simpler questions and follow-up questions that you can use with your tools.
Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."""

chat_generator = OpenAIChatGenerator(model="gpt-3.5-turbo")
chat_agent = Agent(
    chat_generator=chat_generator,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

response = chat_agent.run(messages=[ChatMessage.from_user("What's the weather like in Madrid?")])
print(response['last_message'].text)


## Gradio Chat App

def chat(message, history):
    messages = []
    for item in history:
        if item["role"] == "user":
            messages.append(ChatMessage.from_user(item["content"]))
        elif item["role"] == "assistant":
            messages.append(ChatMessage.from_assistant(item["content"]))
    messages.append(ChatMessage.from_user(message))
    response = chat_agent.run(messages=messages)
    return response['last_message'].text


demo = gr.ChatInterface(
    fn=chat,
    examples=[
        "Can you tell me where Giorgio lives?",
        "What's the weather like in Madrid?",
        "Who lives in London?",
        "What's the weather like where Mark lives?",
    ],
    title="Ask me about weather or where people live!",
)
demo.launch(share=True)










