import argparse, json, os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Both benchmarks are evaluated manually to avoid lm-eval's hardcoded
# max_new_tokens=2048 which cannot be overridden via gen_kwargs.
# GSM8K uses the official test split; GPQA uses the held-out 20% from
# Data.py (seed=42) to prevent leakage from the 80% used in training.


def eval_gsm(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # Load in bf16 to match training precision; eval() disables dropout
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16
    ).to(device).eval()

    # Use the official GSM8K test split, capped at 200 examples for speed
    test = load_dataset("gsm8k", "main")["test"].select(range(200))

    def extract_answer(text):
        # GSM8K answers are formatted as "#### 42" — extract that number first
        match = re.search(r"####\s*(-?[\d,]+)", text)
        if match:
            return match.group(1).replace(",", "")
        # Fallback: take the last number in the text
        numbers = re.findall(r"-?[\d,]+", text)
        return numbers[-1].replace(",", "") if numbers else None

    correct = 0
    for i, ex in enumerate(test):
        # Format prompt using Qwen's chat template, leave assistant turn open
        prompt = (f"<|im_start|>user\n{ex['question']}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            # 256 tokens is enough for a short numeric answer
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)

        # Decode only the newly generated tokens (skip the prompt)
        response  = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        predicted = extract_answer(response)
        gold      = extract_answer(ex["answer"])
        if predicted == gold:
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"  GSM8K [{i+1}/{len(test)}] running accuracy: {correct/(i+1):.2%}")

    acc = correct / len(test)
    print(f"GSM8K (n=200): {acc:.2%}")
    return acc


def eval_gpqa(model_path, device):
    # Manually evaluate on the held-out 20% GPQA split from Data.py (seed=42).
    # Importing here triggers Data.py's split logic with the same seed,
    # giving us the identical test_set_gpqa used nowhere during training.
    from Data import test_set_gpqa

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16
    ).to(device).eval()

    correct = 0
    for i, ex in enumerate(test_set_gpqa):
        # ex["text"] contains the full example including the answer turn.
        # Strip everything from <|im_start|>assistant onward, then re-add
        # the opening tag to prime the model to generate the answer letter.
        prompt = ex["text"].split("<|im_start|>assistant")[0] + "<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            # Generate a single token — the model just needs to pick A/B/C/D
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)

        # Decode only the newly generated token (skip the prompt tokens)
        predicted = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:]).strip()
        if predicted == ex["correct_label"]:
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  GPQA  [{i+1}/{len(test_set_gpqa)}] running accuracy: {correct/(i+1):.2%}")

    acc = correct / len(test_set_gpqa)
    print(f"GPQA (held-out 20%, n={len(test_set_gpqa)}): {acc:.2%}")
    return acc


def run_eval(model_path, output_path):
    # Pick the best available device: GPU > Apple Silicon > CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    gsm_acc  = eval_gsm(model_path, device)   # manual eval on GSM8K test split
    gpqa_acc = eval_gpqa(model_path, device)  # manual eval on held-out 20%

    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"model": model_path,
                   "gsm8k": round(float(gsm_acc), 4),
                   "gpqa":  round(float(gpqa_acc), 4)}, f, indent=2)
    print(f"GSM8K: {gsm_acc:.2%}  |  GPQA: {gpqa_acc:.2%}  |  Saved → {output_path}")


def compare(baseline_path, finetuned_path):
    # Load both result JSONs and print a side-by-side delta table
    with open(baseline_path)  as f: b  = json.load(f)
    with open(finetuned_path) as f: ft = json.load(f)

    print(f"\n{'='*50}")
    print(f"{'Dataset':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Change':>10}")
    print(f"{'-'*50}")
    for ds in ["gsm8k", "gpqa"]:
        delta = ft[ds] - b[ds]
        print(f"{ds:<12} {b[ds]:>9.2%} {ft[ds]:>11.2%}   {'↑' if delta >= 0 else '↓'}{abs(delta):>6.2%}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   help="HF model name or local path to evaluate")
    p.add_argument("--output",  help="Path to save results JSON")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "FINETUNED"),
                   help="Compare two saved result JSONs")
    args = p.parse_args()

    if args.model and args.output: run_eval(args.model, args.output)
    elif args.compare:             compare(*args.compare)
    else:                          p.print_help()
