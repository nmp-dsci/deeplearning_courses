


python3 -m venv venv

source venv/bin/activate

python3 -m pip install -r requirements.txt
# run .env fine 
# source ~/.env

uv run mcp_chatbot.py


# Make sure to interact with the chatbot. Here are some query examples:
# - **@folders**
# - **@ai_interpretability**
# - **/prompts**
# - **/prompt generate_search_prompt topic=history num_papers=2**


# docker
# And then create the image using  this command: `docker build -t my-server .`

# Then run the container using: `docker run -p 8001:8001 my-server`