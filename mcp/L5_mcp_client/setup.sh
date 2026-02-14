# in L5_mcp_client/mcp_project

source .venv/bin/activate

uv init 

uv add anthropic python-dotenv nest_asyncio

uv run mcp_chatbot.py


