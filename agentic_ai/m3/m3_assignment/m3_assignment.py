# ================================
# Standard library imports
# ================================
import json

# ================================
# Third-party imports
# ================================
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import display, HTML

# ================================
# Local / project imports
# ================================
import research_tools

# ================================
# Environment setup
# ================================
load_dotenv()  # Load environment variables from .env file

# Instantiate OpenAI's client (you should use this in your graded functions)
CLIENT = OpenAI()

import unittests


##############
# 1) Research with tools
prompt_ = "Radio observations of recurrent novae"
preliminary_report = generate_research_report_with_tools(prompt_)
print("=== Research Report (preliminary) ===\n")
print(preliminary_report)

# 2) Reflection on the report (use the final TEXT to avoid ambiguity)
reflection_text = reflection_and_rewrite(preliminary_report)   # <-- pass text, not messages
print("=== Reflection on Report ===\n")
print(reflection_text['reflection'], "\n")
print("=== Revised Report ===\n")
print(reflection_text['revised_report'], "\n")


# 3) Convert the report to HTML (use the TEXT and correct function name)
html = convert_report_to_html(reflection_text['revised_report'])

print("=== Generated HTML (preview) ===\n")
print((html or "")[:600], "\n... [truncated]\n")

# 4) Display full HTML
display(HTML(html))

###############

# Test the Tavily search tool
topic = "retrieval-augmented generation applications"

tavily_results = research_tools.tavily_search_tool(topic)
for item in tavily_results:
    print(item)

# Tool mapping
TOOL_MAPPING = {
    "tavily_search_tool": research_tools.tavily_search_tool,
    "arxiv_search_tool": research_tools.arxiv_search_tool,
}

## Example
ChatCompletionMessage(
    content=None,
    refusal=None,
    role='assistant',
    annotations=[],
    audio=None,
    function_call=None,
    tool_calls=[
        ChatCompletionMessageFunctionToolCall(
            id='call_ymMki5TBB91efJhMPjgoqjop',
            function=Function(
                arguments='{"query":"radio observations of recurrent novae","max_results":5}',
                name='arxiv_search_tool'
            ),
            type='function'
        )
    ]
)



######################
# GRADED FUNCTION: generate_research_report_with_tools
def generate_research_report_with_tools(prompt: str, model: str = "gpt-4o") -> str:
    """
    Generates a research report using OpenAI's tool-calling with arXiv and Tavily tools.

    Args:
        prompt (str): The user prompt.
        model (str): OpenAI model name.

    Returns:
        str: Final assistant research report text.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant that can search the web and arXiv to write detailed, "
                "accurate, and properly sourced research reports.\n\n"
                "🔍 Use tools when appropriate (e.g., to find scientific papers or web content).\n"
                "📚 Cite sources whenever relevant. Do NOT omit citations for brevity.\n"
                "🌐 When possible, include full URLs (arXiv links, web sources, etc.).\n"
                "✍️ Use an academic tone, organize output into clearly labeled sections, and include "
                "inline citations or footnotes as needed.\n"
                "🚫 Do not include placeholder text such as '(citation needed)' or '(citations omitted)'."
            )
        },
        {"role": "user", "content": prompt}
    ]

    # List of available tools
    tools = [research_tools.arxiv_tool_def, research_tools.tavily_tool_def]

    # Maximum number of turns
    max_turns = 10
    
    # Iterate for max_turns iterations
    for _ in range(max_turns):

        ### START CODE HERE ###

        # Chat with the LLM via the client and set the correct arguments. Hint: Their names match names of variables already defined.
        # Make sure to let the LLM choose tools automatically. Hint: Look at the docs provided earlier!
        response = CLIENT.chat.completions.create( 
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=1, 
        ) 

        ### END CODE HERE ###

        # Get the response from the LLM and append to messages
        msg = response.choices[0].message 
        messages.append(msg) 

        # Stop when the assistant returns a final answer (no tool calls)
        if not msg.tool_calls:      
            final_text = msg.content
            print("✅ Final answer:")
            print(final_text)
            break

        # Execute tool calls and append results
        for call in msg.tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"🛠️ {tool_name}({args})")

            try:
                tool_func = TOOL_MAPPING[tool_name]
                result = tool_func(**args)
            except Exception as e:
                result = {"error": str(e)}

            ### START CODE HERE ###

            # Keep track of tool use in a new message
            new_msg = { 
                # Set role to "tool" (plain string) to signal a tool was used
                "role": "tool",
                # As stated in the markdown when inspecting the ChatCompletionMessage object 
                # every call has an attribute called id
                "tool_call_id": call.id,
                # The name of the tool was already defined above, use that variable
                "name": tool_name,
                # Pass the result of calling the tool to json.dumps
                "content": json.dumps(result)
            }

            ### END CODE HERE ###

            # Append to messages
            messages.append(new_msg)

    return final_text


###############################
# Exercise 2: Reflection and Rewrite
###############################
## Step 1: 
# Objective: Guide the language model to output a structured response in JSON format.
# Format: Ensure the output includes two keys, "reflection" and "revised_report".
# Details: Your reflection should cover strengths, limitations, suggestions, and opportunities. The revised report should incorporate these elements to improve clarity and academic tone.
## Step 2: 
# Parameters: Use the specified model (e.g., "gpt-4o-mini") and set the temperature equal to the temperature parameter of the graded function.
# Structure: Make sure the response setup directs the model properly, ensuring the JSON format is adhered to without additional commentary.




####################
# GRADED FUNCTION: reflection_and_rewrite
def reflection_and_rewrite(report, model: str = "gpt-4o-mini", temperature: float = 0.3) -> dict:
    """
    Generates a structured reflection AND a revised research report.
    Accepts raw text OR the messages list returned by generate_research_report_with_tools.

    Returns:
        dict with keys:
          - "reflection": structured reflection text
          - "revised_report": improved version of the input report
    """

    # Input can be plain text or a list of messages, this function detects and parses accordingly
    report = research_tools.parse_input(report)

    ### START CODE HERE ###

    # Define the prompt. A multi-line f-string is typically used for this.

    # Remember it should ask the model to output ONLY valid JSON with this structure:
    # {{ "reflection": "<text>", "revised_report": "<text>" }}
    user_prompt = f"""
        You are an academic reviewer and editor. Please analyze the following research report and provide a structured reflection and a revised version.
        Your reflection should cover strengths, limitations, suggestions, and opportunities. The revised report should incorporate these elements to improve clarity and academic tone. 

        Input report:
        {report}

        Please output ONLY valid JSON with the following structure:
        {{ "reflection": "<text>", "revised_report": "<text>" }}
        """

    # Get a response from the LLM
    response = CLIENT.chat.completions.create( 
        # Pass in the model
        model=model,
        messages=[ 
            # System prompt is already defined
            {"role": "system", "content": "You are an academic reviewer and editor."},
            # Add user prompt
            {"role": "user", "content": user_prompt},
        ],
        # Set the temperature equal to the temperature parameter passed to the function
        temperature=temperature
    )

    ### END CODE HERE ###

    # Extract output
    llm_output = response.choices[0].message.content.strip()

    # Check if output is valid JSON
    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError:
        raise Exception("The output of the LLM was not valid JSON. Adjust your prompt.")

    return {
        "reflection": str(data.get("reflection", "")).strip(),
        "revised_report": str(data.get("revised_report", "")).strip(),
    }



###############################
# Exercise 3: Convert Report to HTML
###############################

# GRADED FUNCTION: convert_report_to_html
def convert_report_to_html(report, model: str = "gpt-4o", temperature: float = 0.5) -> str:
    """
    Converts a plaintext research report into a styled HTML page using OpenAI.
    Accepts raw text OR the messages list from the tool-calling step.
    """

    # Input can be plain text or a list of messages, this function detects and parses accordingly
    report = research_tools.parse_input(report)

    # System prompt is already provided
    system_prompt = "You convert plaintext reports into full clean HTML documents."

    ### START CODE HERE ###
    
    # Build the user prompt instructing the model to return ONLY valid HTML
    user_prompt = f"""
        Convert the following research report into a well-structured HTML document.
        Use appropriate HTML tags such as <html>, <head>, <body>, <h1>, <h2>, <p>, <a>, and <ul>/<li> for lists.
        Ensure the HTML is clean, valid, and styled for readability.
        Format: Ensure the output is valid, clean HTML with appropriate section headers, formatted paragraphs, and clickable links.

        Input report:
        {report}

        Please output ONLY valid HTML without additional commentary 
        """

    # Call the LLM by interacting with the CLIENT. 
    # Remember to set the correct values for the model, messages (system and user prompts) and temperature
    response = CLIENT.chat.completions.create( 
        model=model,
        messages=[ 
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature
    )
    ### END CODE HERE ###

    # Extract the HTML from the assistant message
    html = response.choices[0].message.content.strip()  

    return html








