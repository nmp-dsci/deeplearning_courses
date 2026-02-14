


python3 -m venv venv

source venv/bin/activate

# run .env fine 
# source ~/.env

uv run mcp_chatbot.py

# Example queries to ask the chatbot:
# - Fetch the content of this website: https://modelcontextprotocol.io/docs/concepts/architecture and save the content in the file "mcp_summary.md", create a visual diagram that summarizes the content of "mcp_summary.md" and save it in a text file
# - Fetch deeplearning.ai and find an interesting term. Search for 2 papers around the term and then summarize your findings and write them to a file called results.txt

api_key