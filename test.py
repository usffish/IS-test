import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.utils import make_table

# GSM8K is safe to evaluate via lm-eval because it has a dedicated test split
# that was never seen during training. GPQA has no official split, so we
# evaluate it manually using the exact 20% held-out set from Data.py to
# prevent any data leakage from the 80% used in training.


def eval_gsm(model_args, device):
    # Run GSM8K via lm-eval using 5-shot evaluation (short answers, no CoT).
    # gsm8k_cot generates up to 2048 tokens per example and the task yaml
    # overrides gen_kwargs, making it impractically slow (~3.6 hrs for 1319 examples).
    # gsm8k (non-CoT) generates short answers and runs in ~30 minutes.
    # limit=200: evaluate on 200 examples for a reliable estimate
    results = simple_evaluate(
        model="hf", model_args=model_args,
        tasks=["gsm8k"],
        apply_chat_template=True, batch_size=8,
        device=device, log_samples=False,
        limit=200,
    )
    print(make_table(results))
    # strict-match: the answer must exactly match the expected number
    return results["results"]["gsm8k"].get("exact_match,strict-match", 0)


def eval_gpqa(model_path, device):
    # Manually evaluate on the held-out 20% GPQA split from Data.py (seed=42).
    # Importing here triggers Data.py's split logic with the same seed,
    # giving us the identical test_set_gpqa used nowhere during training.
    from Data import test_set_gpqa

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Match tokenizer settings from training (main.py)
    tokenizer.pad_token = tokenizer.eos_token

    # Load in bf16 to match training precision; eval() disables dropout
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()

    correct = 0
    for ex in test_set_gpqa:
        # ex["text"] contains the full formatted example including the answer:
        #   <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\nB<|im_end|>
        # Strip the assistant answer so the model only sees the question prompt,
        # then re-add the opening assistant tag to prime generation.
        # Without this the model sees the correct answer before predicting it.
        prompt = ex["text"].split("<|im_start|>assistant")[0] + "<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(device)

        with torch.no_grad():
            # Generate a single token — the model just needs to pick A/B/C/D
            # do_sample=False ensures greedy (deterministic) decoding
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False)

        # Decode only the newly generated token (skip the prompt tokens)
        predicted = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:]).strip()
        if predicted == ex["correct_label"]:
            correct += 1

    acc = correct / len(test_set_gpqa)
    print(f"GPQA (held-out 20%, n={len(test_set_gpqa)}): {acc:.2%}")
    return acc


def run_eval(model_path, output_path):
    # Pick the best available device: GPU > Apple Silicon > CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Build model arg string for lm-eval; append HF token if set in environment
    model_args = f"pretrained={model_path}"
    if os.environ.get("HF_TOKEN"):
        model_args += f",token={os.environ['HF_TOKEN']}"

    gsm_acc  = eval_gsm(model_args, device)   # lm-eval on official test split
    gpqa_acc = eval_gpqa(model_path, device)  # manual eval on held-out 20%

    # Save a compact summary JSON for later comparison
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
