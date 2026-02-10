# marked assignment for Mileshone 2 
## requirements 
### Step 1: drafing : call LLM for draft essay 
### Step 2: reflection: reflect on the draft using reasoning step 
### Step 3: revision: revise the essay based on the reflection




from dotenv import load_dotenv

load_dotenv()

import aisuite as ai

# Define the client. You can use this variable inside your graded functions!
CLIENT = ai.Client()

import unittests


### EXERCISE 1: `generate_draft` function 








def generate_draft(topic: str, model: str) -> str:
    """
    This function takes a topic and a model name, and generates a draft essay on the given topic using the specified model.

    Args:
        topic (str): The topic for which to generate the essay.
        model (str): The name of the model to use for generating the essay.

    Returns:
        str: The generated draft essay.
    """
    prompt = f"""Write a detailed essay on the following 
    
    Topic:
    {topic}
    The essay should be well-structured, with an introduction, body paragraphs, and a conclusion.
    Use clear and concise language, and provide examples where appropriate.
    
    """
    
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    
    return response.choices[0].message.content

# Test your code!
unittests.test_generate_draft(generate_draft)



################################
### EXERCISE 2: `reflect_on_draft` function
def reflect_on_draft(draft: str, model: str = "openai:o4-mini") -> str:

    ### START CODE HERE ###

    # Define your prompt here. A multi-line f-string is typically used for this.
    prompt = f"""
    you are an expert editor. Your task is to critique the following essay draft and provide specific, actionable feedback for improvement. Focus on areas such as clarity, coherence, structure, grammar, and overall effectiveness in conveying the topic.

    Essay Draft:
    {draft} 

    The feedback should be critical but constructive.
    It should address issues such as structure, clarity, strength of argument, and writing style.

    Output: 
    You do **not** need to rewrite the essay at this step—just analyze and reflect on it.

    """

    ### END CODE HERE ###

    # Get a response from the LLM by creating a chat with the client.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    return response.choices[0].message.content



#################################
# GRADED FUNCTION: revise_draft

def revise_draft(original_draft: str, reflection: str, model: str = "openai:gpt-4o") -> str:

    ### START CODE HERE ###

    # Define your prompt here. A multi-line f-string is typically used for this.
    prompt = f"""
    You are an expert editor. Your task is to revise the following essay draft based on the feedback provided.

    Original Essay Draft:
    {original_draft}

    Feedback:
    {reflection}

    Please revise the essay draft to address the feedback and improve clarity, coherence, structure, grammar, and overall effectiveness in conveying the topic.
    """

    # Get a response from the LLM by creating a chat with the client.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    ### END CODE HERE ###

    return response.choices[0].message.content