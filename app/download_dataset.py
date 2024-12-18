from datasets import load_dataset
import json
import os

# Function to save a dataset split as a JSON file
def save_to_json(dataset_split, filename):
    data = []
    for example in dataset_split:
        data.append({
            "context": example["context"],
            "question": example["question"],
            "answers": example["answers"]["text"]
        })
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} examples to {filename}")

# Load the SQuAD dataset
print("Loading SQuAD dataset...")
dataset = load_dataset("squad")

# Access the training and validation splits
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

# Define output folder
output_dir = "/ai_interview_tool/data"
os.makedirs(output_dir, exist_ok=True)

# Save training and validation splits as JSON files
save_to_json(train_dataset, os.path.join(output_dir, "squad_train.json"))
save_to_json(validation_dataset, os.path.join(output_dir, "squad_validation.json"))

print("Dataset saved successfully!")

