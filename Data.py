from datasets import load_dataset
import os

# Load MentalChat16K from HuggingFace (instruction, input, output columns)
raw = load_dataset("ShenLab/MentalChat16K", token=os.environ.get("HF_TOKEN"))

SYSTEM_PROMPT = (
    "You are a helpful mental health counselling assistant. "
    "Answer the mental health questions based on the patient's description. "
    "Give helpful, comprehensive, and appropriate answers."
)


def format_mentalchat(example):
    # Mistral-Instruct chat template: <s>[INST] {user} [/INST] {assistant}</s>
    user_turn = f"{SYSTEM_PROMPT}\n\n{example['input']}"
    return {
        "text": f"<s>[INST] {user_turn} [/INST] {example['output']}</s>"
    }


formatted = raw["train"].map(format_mentalchat, remove_columns=raw["train"].column_names)

# MentalChat16K only ships a train split — carve out 10% for evaluation
split = formatted.train_test_split(test_size=0.1, seed=42)
combined_train = split["train"]
combined_test  = split["test"]
