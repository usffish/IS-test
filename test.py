import argparse
import json
import os
import torch
from lm_eval import simple_evaluate
from lm_eval.utils import make_table


def run_eval(model_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_args = f"pretrained={model_path}"
    if os.environ.get("HF_TOKEN"):
        model_args += f",token={os.environ.get('HF_TOKEN')}"

    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["gsm8k_cot", "gpqa_main"],
        apply_chat_template=True,
        batch_size=8,
        device=device,
        log_samples=False,
    )

    print(make_table(results))

    gsm_acc  = results["results"]["gsm8k_cot"].get("exact_match,flexible-extract", 0)
    gpqa_acc = results["results"]["gpqa_main"].get("acc_norm,none", 0)

    summary = {
        "model": model_path,
        "gsm8k": round(float(gsm_acc),  4),
        "gpqa":  round(float(gpqa_acc), 4),
    }

    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"GSM8K: {gsm_acc:.2%}  |  GPQA: {gpqa_acc:.2%}  |  Saved → {output_path}")


def compare(baseline_path, finetuned_path):
    with open(baseline_path)  as f: b  = json.load(f)
    with open(finetuned_path) as f: ft = json.load(f)

    print(f"\n{'='*50}")
    print(f"{'Dataset':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Change':>10}")
    print(f"{'-'*50}")
    for dataset in ["gsm8k", "gpqa"]:
        delta = ft[dataset] - b[dataset]
        arrow = "↑" if delta >= 0 else "↓"
        print(f"{dataset:<12} {b[dataset]:>9.2%} {ft[dataset]:>11.2%}   {arrow}{abs(delta):>6.2%}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   help="HF model name or local path")
    parser.add_argument("--output",  help="Path to save results JSON")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "FINETUNED"))
    args = parser.parse_args()

    if args.model and args.output:
        run_eval(args.model, args.output)
    elif args.compare:
        compare(*args.compare)
    else:
        parser.print_help()