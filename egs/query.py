


load_dotenv()

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
anthropic = Anthropic()

response = anthropic.messages.create(
                max_tokens = 2024,
                model = 'claude-opus-4-6', 
                messages = [{'role':'user', 'content': 'what is the capital of france?'}]
            )
            

            