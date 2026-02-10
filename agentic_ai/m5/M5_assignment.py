

# =========================
# Imports
# =========================

# --- Standard library 
from datetime import datetime
import re
import json
import ast


# --- Third-party ---
from IPython.display import Markdown, display
from aisuite import Client

# --- Local / project ---
import research_tools
import unittests

CLIENT = Client()



# - `research_tools.arxiv_search_tool(query)` → search academic papers from **arXiv**  
#   *Example:* `research_tools.arxiv_search_tool("neural networks for climate modeling")`
# - `research_tools.tavily_search_tool(query)` → perform web searches with the **Tavily API**  
#   *Example:* `research_tools.tavily_search_tool("latest trends in sunglasses fashion")`
# - `research_tools.wikipedia_search_tool(query)` → retrieve summaries from **Wikipedia**  
#   *Example:* `research_tools.wikipedia_search_tool("Ensemble Kalman Filter")`

# =========================
# EXERCISE 1: planner agent
# Ensure CLIENT.chat.completions.create is correctly configured.
# Pass the model and messages parameters correctly:
# Model: Use "openai:o4-mini" by default.
# Messages: Set with {"role": "user", "content": user_prompt}.
# Temperature: Fixed at 1 for creative outputs.


# GRADED FUNCTION: planner_agent

def planner_agent(topic: str, model: str = "openai:o4-mini") -> list[str]:
    """
    Generates a plan as a Python list of steps (strings) for a research workflow.

    Args:
        topic (str): Research topic to investigate.
        model (str): Language model to use.

    Returns:
        List[str]: A list of executable step strings.
    """

    
    # Build the user prompt
    user_prompt = f"""
    You are a planning agent responsible for organizing a research workflow with multiple intelligent agents.

    🧠 Available agents:
    - A research agent who can search the web, Wikipedia, and arXiv.
    - A writer agent who can draft research summaries.
    - An editor agent who can reflect and revise the drafts.

    🎯 Your job is to write a clear, step-by-step research plan **as a valid Python list**, where each step is a string.
    Each step should be atomic, executable, and must rely only on the capabilities of the above agents.

    🚫 DO NOT include irrelevant tasks like "create CSV", "set up a repo", "install packages", etc.
    ✅ DO include real research-related tasks (e.g., search, summarize, draft, revise).
    ✅ DO assume tool use is available.
    ✅ DO NOT include explanation text — return ONLY the Python list.
    ✅ The final step should be to generate a Markdown document containing the complete research report.

    Topic: "{topic}"
    """

    # Add the user prompt to the messages list
    messages = [{"role": "user", "content": user_prompt}]

    ### START CODE HERE ###

    # Call the LLM
    response = CLIENT.chat.completions.create( 
        # Pass in the model
        model=model,
        # Define the messages. Remember this is meant to be a user prompt!
        messages=messages,
        # Keep responses creative
        temperature=1, 
    )

    ### END CODE HERE ###

    # Extract message from response
    steps_str = response.choices[0].message.content.strip()

    # Parse steps
    steps = ast.literal_eval(steps_str)

    return steps

# Test your code!
unittests.test_planner_agent(planner_agent)

# =========================

# GRADED FUNCTION: research_agent

def research_agent(task: str, model: str = "openai:gpt-4o", return_messages: bool = False):
    """
    Executes a research task using tools via aisuite (no manual loop).
    Returns either the assistant text, or (text, messages) if return_messages=True.
    """
    print("==================================")  
    print("🔍 Research Agent")                 
    print("==================================")

    current_time = datetime.now().strftime('%Y-%m-%d')
    
    ### START CODE HERE ###

    # Create a customizable prompt by defining the role (e.g., "research assistant"),
    # listing tools (arxiv_tool, tavily_tool, wikipedia_tool) for various searches,
    # specifying the task with a placeholder, and including a current_time placeholder.
    prompt = f"""
You are a research assistant with access to powerful tools for gathering information and insights. 

Your task is to execute the following research task: "{task}"

If needed, today date is {datetime.now().strftime("%Y-%m-%d")}

You can call the following tools:
- tavily_search_tool: to discover external web trends.
- wikipedia_search_tool:: to retrieve summaries of concepts.
- arxiv_search_tool: to find academic papers on arXiv.


Once your analysis is complete, summarize:
- The top 2–3 trends you found.
- The product(s) from the catalog that fit these trends.
- A justification of why they are a good fit for the summer campaign.
"""
    
    # Create the messages dict to pass to the LLM. Remember this is a user prompt!
    messages = [{"role": 'user', "content": prompt}]

    # Save all of your available tools in the tools list. These can be found in the research_tools module.
    # You can identify each tool in your list like this: 
    # research_tools.<name_of_tool>, where <name_of_tool> is replaced with the function name of the tool.
    tools_ = [research_tools.arxiv_search_tool, research_tools.tavily_search_tool, research_tools.wikipedia_search_tool]
    
    # Call the model with tools enabled
    response = CLIENT.chat.completions.create(  
        # Set the model
        model=model,
        # Pass in the messages. You already defined this!
        messages=messages,
        # Pass in the tools list. You already defined this!
        tools=tools_,
        # Set the LLM to automatically choose the tools
        tool_choice="auto",
        # Set the max turns to 6
        max_turns=6
    )  
    
    ### END CODE HERE ###

    content = response.choices[0].message.content
    print("✅ Output:\n", content)

    
    return (content, messages) if return_messages else content  



# =========================
# exercise 3: writer agent
# GRADED FUNCTION: writer_agent
def writer_agent(task: str, model: str = "openai:gpt-4o") -> str: # @REPLACE def writer_agent(task: str, model: str = None) -> str:
    """
    Executes writing tasks, such as drafting, expanding, or summarizing text.
    """
    print("==================================")
    print("✍️ Writer Agent")
    print("==================================")

    ### START CODE HERE ###
    
    # Create the system prompt.
    # This should assign the LLM the role of a writing agent specialized in generating well-structured academic or technical content
    system_prompt = "You are a writing agent specialized in generating well-structured academic or technical content."

    # Define the system msg by using the system_prompt and assigning the role of system
    system_msg = {"role": "system", "content": system_prompt}

    # Define the user msg. In this case the user prompt should be the task passed to the function
    user_msg = {"role": "user", "content": task}

    # Add both system and user messages to the messages list
    messages = [system_msg, user_msg]
    
    ### END CODE HERE ###

    response = CLIENT.chat.completions.create(
        model=model, 
        messages=messages,
        temperature=1.0
    )

    return response.choices[0].message.content


# =========================

# GRADED FUNCTION: editor_agent
def editor_agent(task: str, model: str = "openai:gpt-4o") -> str:
    """
    Executes editorial tasks such as reflection, critique, or revision.
    """
    print("==================================")
    print("🧠 Editor Agent")
    print("==================================")
    
    ### START CODE HERE ###

    # Create the system prompt.
    # This should assign the LLM the role of an editor agent specialized in reflecting on, critiquing, or improving existing drafts.
    system_prompt = "You are an editor agent specialized in reflecting on, critiquing, or improving existing drafts."
    
    # Define the system msg by using the system_prompt and assigning the role of system
    system_msg = {"role": "system", "content": system_prompt}

    # Define the user msg. In this case the user prompt should be the task passed to the function
    user_msg = {"role": "user", "content": task}
    
    # Add both system and user messages to the messages list
    messages = [system_msg, user_msg]
    
    ### END CODE HERE ###
    
    response = CLIENT.chat.completions.create(
        model=model, 
        messages=messages,
        temperature=0.7 
    )
    
    return response.choices[0].message.content

# Test your code!
unittests.test_editor_agent(editor_agent)


# =========================


agent_registry = {
    "research_agent": research_agent,
    "editor_agent": editor_agent,
    "writer_agent": writer_agent,
}

def clean_json_block(raw: str) -> str:
    """
    Clean the contents of a JSON block that may come wrapped with Markdown backticks.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()

def executor_agent(topic, model: str = "openai:gpt-4o", limit_steps: bool = True):

    plan_steps = planner_agent(topic)
    max_steps = 4

    if limit_steps:
        plan_steps = plan_steps[:min(len(plan_steps), max_steps)]
    
    history = []

    print("==================================")
    print("🎯 Editor Agent")
    print("==================================")

    for i, step in enumerate(plan_steps):

        agent_decision_prompt = f"""
        You are an execution manager for a multi-agent research team.

        Given the following instruction, identify which agent should perform it and extract the clean task.

        Return only a valid JSON object with two keys:
        - "agent": one of ["research_agent", "editor_agent", "writer_agent"]
        - "task": a string with the instruction that the agent should follow

        Only respond with a valid JSON object. Do not include explanations or markdown formatting.

        Instruction: "{step}"
        """
        response = CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": agent_decision_prompt}],
            temperature=0,
        )

        raw_content = response.choices[0].message.content
        cleaned_json = clean_json_block(raw_content)
        agent_info = json.loads(cleaned_json)

        agent_name = agent_info["agent"]
        task = agent_info["task"]

        context = "\n".join([
            f"Step {j+1} executed by {a}:\n{r}" 
            for j, (s, a, r) in enumerate(history)
        ])
        enriched_task = f"""
        You are {agent_name}.

        Here is the context of what has been done so far:
        {context}

        Your next task is:
        {task}
        """

        print(f"\n🛠️ Executing with agent: `{agent_name}` on task: {task}")

        if agent_name in agent_registry:
            output = agent_registry[agent_name](enriched_task)
            history.append((step, agent_name, output))
        else:
            output = f"⚠️ Unknown agent: {agent_name}"
            history.append((step, agent_name, output))

        print(f"✅ Output:\n{output}")

    return history

# If you want to see the full workflow without limiting the number of steps. Set limit_steps to False
# Keep in mind this could take more than 10 minutes to complete
executor_history = executor_agent("The ensemble Kalman filter for time series forecasting", limit_steps=True)

md = executor_history[-1][-1].strip("`")  
display(Markdown(md))
