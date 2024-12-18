from datasets import load_dataset
from transformers import pipeline

# Load SQuAD dataset
dataset = load_dataset("squad")

# Load pre-trained question-answering pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Test on SQuAD data
def test_squad(dataset, qa_pipeline, num_examples=3):
    print("\nEvaluating QA Pipeline on SQuAD Dataset...\n")
    for i in range(num_examples):
        context = dataset["train"][i]["context"]
        question = dataset["train"][i]["question"]
        answers = dataset["train"][i]["answers"]["text"]

        # Get model answer
        model_answer = qa_pipeline(question=question, context=context)["answer"]

        print(f"Q{i+1}: {question}")
        print(f"Context: {context[:200]}...")  # Print first 200 chars for readability
        print(f"🔍 Expected Answer: {answers[0]}")
        print(f"💡 Model Answer: {model_answer}")
        print("-" * 50)

if __name__ == "__main__":
    test_squad(dataset, qa_pipeline)

