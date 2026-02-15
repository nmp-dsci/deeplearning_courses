
cd mcp_project
uv init
uv venv
source .venv/bin/activate
uv add mcp arxiv

npx @modelcontextprotocol/inspector uv run research_server.py
