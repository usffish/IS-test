from datasets import load_dataset
import os
# import datasets
gsm = load_dataset("gsm8k", "main", token=os.environ.get("HF_TOKEN"))
gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main", token=os.environ.get("HF_TOKEN"))
'''
# Explore
print ("GSM8K")
for ex in gsm["test"].select(range(3)):
    print("Q:", ex["question"])
    print("A:", ex["answer"])
    print()

print("GPQA")
for ex in gpqa["train"].select(range(3)):
    print("Q:", ex["Question"][:200])
    print("Correct:", ex["Correct Answer"][:100])
    print("Subdomain:", ex["Subdomain"])
    print()
'''
# Format dataset
def format_gsm(data):
    return {"text": f"<|im_start|>user\n{data['question']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{data['answer']}<|im_end|>"}

fgsm = gsm.map(format_gsm)
# print(fgsm.column_names)