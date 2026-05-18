import random
from datasets import load_dataset, concatenate_datasets
import os

random.seed(42)

# import datasets
gsm = load_dataset("gsm8k", "main", token=os.environ.get("HF_TOKEN"))
gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main", token=os.environ.get("HF_TOKEN"))

# # Explore
# print ("GSM8K")
# for ex in gsm["test"].select(range(3)):
#     print("Q:", ex["question"])
#     print("A:", ex["answer"])
#     print()
#
# print("GPQA")
# for ex in gpqa["train"].select(range(3)):
#     print("Q:", ex["Question"][:200])
#     print("Correct:", ex["Correct Answer"][:100])
#     print("Subdomain:", ex["Subdomain"])
#     print()

# Format gsm dataset
def format_gsm(data):
    return {"text": f"<|im_start|>user\n{data['question']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{data['answer']}<|im_end|>"}

fgsm = gsm.map(format_gsm)
# print(fgsm.column_names)


# Format_gpqa(data):
def format_gpqa(data):
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

fgpqa = gpqa.map(format_gpqa)

# Split into train and test
fgpqa_split = fgpqa["train"].train_test_split(test_size=0.2, seed=42)
train_set_gpqa = fgpqa_split["train"]
test_set_gpqa = fgpqa_split["test"]


#combine datasets

combined_train = concatenate_datasets([train_set_gpqa,fgsm["train"]])
combined_test = concatenate_datasets([test_set_gpqa,fgsm["test"]])