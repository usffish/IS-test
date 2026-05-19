import os
import random
from datasets import load_dataset, concatenate_datasets

random.seed(42)

# Load datasets from Hugging Face Hub
gsm  = load_dataset("gsm8k", "main", token=os.environ.get("HF_TOKEN"))
gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main", token=os.environ.get("HF_TOKEN"))


def format_gsm(data):
    # Format as a Qwen chat turn: user asks the question, assistant gives the solution
    return {
        "text": (
            f"<|im_start|>user\n{data['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{data['answer']}<|im_end|>"
        )
    }


def format_gpqa(data):
    # Shuffle answer choices to prevent the model from learning position bias
    options = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    random.shuffle(options)
    correct_label = "ABCD"[options.index(data["Correct Answer"])]
    options_str = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", options))

    return {
        "text": (
            f"<|im_start|>user\n{data['Question']}\n\n{options_str}<|im_end|>\n"
            f"<|im_start|>assistant\n{correct_label}<|im_end|>"
        ),
        "correct_label": correct_label,
    }


# Apply formatting
fgsm  = gsm.map(format_gsm)
fgpqa = gpqa.map(format_gpqa)

# GPQA has no official split — create an 80/20 train/test split with a fixed seed
# so the same 20% is always held out for evaluation
fgpqa_split    = fgpqa["train"].train_test_split(test_size=0.2, seed=42)
train_set_gpqa = fgpqa_split["train"]
test_set_gpqa  = fgpqa_split["test"]

# Combine GSM8K and GPQA splits for training and evaluation
combined_train = concatenate_datasets([train_set_gpqa, fgsm["train"]])
combined_test  = concatenate_datasets([test_set_gpqa,  fgsm["test"]])
