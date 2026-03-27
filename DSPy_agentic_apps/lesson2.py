# https://learn.deeplearning.ai/courses/dspy-build-optimize-agentic-apps/lesson/bsrk7/dspy-programming---signatures-and-modules




from dotenv import load_dotenv
load_dotenv()



import dspy
dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))


# DSPy built in model fur Sentiment Classifier
class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a text."""
    text: str = dspy.InputField(desc="input text to classify sentiment")
    sentiment: int = dspy.OutputField(
        desc="sentiment, the higher the more positive", ge=0, le=10
    )


str_signature = dspy.make_signature("text -> sentiment")

## Create a Module to Interact with the LM 

predict = dspy.Predict(SentimentClassifier) 

output = predict(text="I am feeling pretty happy!")
print(output)

print(f"The sentiment is: {output.sentiment}")

dspy.configure(lm=dspy.LM("openai/gpt-4o"))
print(predict(text="I am feeling pretty happy!"))




dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))



## Whait where is my prompt 
dspy.inspect_history(n=1)


#### Try a Different Built-in Module 

cot = dspy.ChainOfThought(SentimentClassifier)

output = cot(text="I am feeling pretty happy!")
print(output)

dspy.inspect_history(n=1)

#### Use a Different Adapter


dspy.configure(adapter=dspy.JSONAdapter())


print(cot(text="I am feeling pretty happy!"))
dspy.inspect_history(n=1)


### Build a Program with Custom Module

class QuestionGenerator(dspy.Signature):
    """Generate a yes or no question in order to guess the celebrity name in users' mind. You can ask in general or directly guess the name if you think the signal is enough. You should never ask the same question in the past_questions."""
    past_questions: list[str] = dspy.InputField(desc="past questions asked")
    past_answers: list[bool] = dspy.InputField(desc="past answers")
    new_question: str = dspy.OutputField(desc="new question that can help narrow down the celebrity name")
    guess_made: bool = dspy.OutputField(desc="If the new_question is the celebrity name guess, set to True, if it is still a general question set to False")


class Reflection(dspy.Signature):
    """Provide reflection on the guessing process"""
    correct_celebrity_name: str = dspy.InputField(desc="the celebrity name in user's mind")
    final_guessor_question: str = dspy.InputField(desc="the final guess or question LM made")
    past_questions: list[str] = dspy.InputField(desc="past questions asked")
    past_answers: list[bool] = dspy.InputField(desc="past answers")
    reflection: str = dspy.OutputField(
        desc="reflection on the guessing process, including what was done well and what can be improved"
    )

def ask(prompt, valid_responses=("y", "n")):
    while True:
        response = input(f"{prompt} ({'/'.join(valid_responses)}): ").strip().lower()
        if response in valid_responses:
            return response
        print(f"Please enter one of: {', '.join(valid_responses)}")


class CelebrityGuess(dspy.Module):
    def __init__(self, max_tries=10):
        super().__init__()
        self.question_generator = dspy.ChainOfThought(QuestionGenerator)
        self.reflection = dspy.ChainOfThought(Reflection)
        self.max_tries = 20
    #
    def forward(self):
        celebrity_name = input("Please think of a celebrity name, once you are ready, type the name and press enter...")
        past_questions = []
        past_answers = []
        #
        correct_guess = False
        for i in range(self.max_tries):
            question = self.question_generator(
                past_questions=past_questions,
                past_answers=past_answers,
            )
            answer = ask(f"{question.new_question}").lower() == "y"
            past_questions.append(question.new_question)
            past_answers.append(answer)
            if question.guess_made and answer:
                correct_guess = True
                break
        #
        if correct_guess:
            print("Yay! I got it right!")
        else:
            print("Oops, I couldn't guess it right.")
        #
        reflection = self.reflection(
            correct_celebrity_name=celebrity_name,
            final_guessor_question=question.new_question,
            past_questions=past_questions,
            past_answers=past_answers,
        )
        print(reflection.reflection)



celebrity_guess = CelebrityGuess()

celebrity_guess()


celebrity_guess.save("celebrity.json", save_program=False)
celebrity_guess.load("celebrity.json")


celebrity_guess.save("celebrity/", save_program=True)

loaded = dspy.load("celebrity/")


