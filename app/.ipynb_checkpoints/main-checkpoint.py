import json
from transformers import pipeline

def load_model():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def load_custom_dataset(file_path):
    with open(file_path, "r") as f:
        return json.load(f)["data"]

def answer_question(qa_pipeline, question, context):
    return qa_pipeline(question=question, context=context)["answer"]

def main():
    qa_pipeline = load_model()
    dataset = load_custom_dataset("data/custom_interview_questions.json")

    print("Evaluating on Custom Interview Dataset...\n")
    for topic in dataset:
        print(f"Topic: {topic['title'].replace('_', ' ').title()}")
        for paragraph in topic["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                expected_answer = qa["answers"][0]["text"]
                model_answer = answer_question(qa_pipeline, question, context)
                print(f"Q: {question}")
                print(f"🔍 Expected Answer: {expected_answer}")
                print(f"💡 Model Answer: {model_answer}")
                print("-" * 50)

if __name__ == "__main__":
    main()
