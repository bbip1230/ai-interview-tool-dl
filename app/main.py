import json
import os
from transformers import pipeline

def load_model():
    """Load the BERT model for question answering."""
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def load_questions(file_path=os.path.join("../data", "interview_questions.json")):
    """Load questions from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)["topics"]

def answer_question(qa_pipeline, question, context):
    """Use the model to answer a question."""
    return qa_pipeline(question=question, context=context)["answer"]

def main():
    qa_pipeline = load_model()
    topics = load_questions()

    print("Available Topics:")
    for idx, topic in enumerate(topics.keys(), start=1):
        print(f"{idx}. {topic.replace(_,  ).title()}")

    # User selects a topic
    choice = int(input("\nSelect a topic by number: ")) - 1
    selected_topic = list(topics.keys())[choice]

    print(f"\nYou selected: {selected_topic.replace(_,  ).title()}\n")

    # Ask questions under the selected topic
    questions = topics[selected_topic]
    for idx, item in enumerate(questions, start=1):
        print(f"Q{idx}: {item[question]}")
        answer = answer_question(qa_pipeline, item[question], item[context])
        print(f"💡 Answer: {answer}\n")

if __name__ == "__main__":
    main()
