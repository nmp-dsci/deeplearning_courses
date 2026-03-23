
from dotenv import load_dotenv
_ = load_dotenv()


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
# from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver


from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)



####################################

from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""


def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


####################

class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
    #
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
    #
    def exists_action(self, state: AgentState):
        print(state)
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    #
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}



prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""


model = ChatOpenAI(model="gpt-3.5-turbo")


with SqliteSaver.from_conn_string(":memory:") as memory:
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)
    messages = [HumanMessage(content="Whats the weather in SF?")]
    thread = {"configurable": {"thread_id": "1"}}
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v)
    print('#'*30)
    abot.graph.get_state(thread)
    print('#'*30)
    abot.graph.get_state(thread).next


with SqliteSaver.from_conn_string(":memory:") as memory:
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)
    messages = [HumanMessage("Whats the weather in LA?")]
    thread = {"configurable": {"thread_id": "2"}}
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v)
    while abot.graph.get_state(thread).next:
        print("\n", abot.graph.get_state(thread),"\n")
        _input = input("proceed?")
        if _input != "y":
            print("aborting")
            break
        for event in abot.graph.stream(None, thread):
            for v in event.values():
                print(v)



####################
## Build a small graph 


class AgentState(TypedDict):
    lnode: str
    scratch: str
    count: Annotated[int, operator.add]



def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
    return {"lnode": "node_1",
            "count": 1,
           }

def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2",
            "count": 1,
           }

def should_continue(state):
    return state["count"] < 4

builder = StateGraph(AgentState)
builder.add_node("Node1", node1)
builder.add_node("Node2", node2)

builder.add_edge("Node1", "Node2")
builder.add_conditional_edges("Node2", 
                              should_continue, 
                              {True: "Node1", False: END})
builder.set_entry_point("Node1")

with SqliteSaver.from_conn_string(":memory:") as memory: 
    graph = builder.compile(checkpointer=memory)
    thread = {"configurable": {"thread_id": str(1)}}
    graph.invoke({"count":0, "scratch":"hi"},thread)
    graph.get_state(thread)
