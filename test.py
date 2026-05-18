"""
evaluate.py — Evaluate Qwen2.5-1.5B on GSM8K and GPQA using lm-evaluation-harness.

Usage:
    # Baseline (before fine-tuning)
    python evaluate.py --model Qwen/Qwen2.5-1.5B --output results/baseline.json

    # Fine-tuned (after training)
    python evaluate.py --model ./tuned-qwen --output results/finetuned.json

    # Compare baseline vs fine-tuned
    python evaluate.py --compare results/baseline.json results/finetuned.json
"""

import argparse
import json
import os
import sys

def run_eval(model_path: str, output_path: str):
    """Run lm-eval on GSM8K and GPQA and save results to output_path."""
    from lm_eval import simple_evaluate
    from lm_eval.utils import make_table

    print(f"\n{'='*55}")
    print(f"  Evaluating: {model_path}")
    print(f"{'='*55}\n")

    hf_token = os.environ.get("HF_TOKEN")

    # model args — pass token for gated datasets
    model_args = f"pretrained={model_path}"
    if hf_token:
        model_args += f",token={hf_token}"

    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["gsm8k_cot", "gpqa_main"],
        apply_chat_template=True,   # use Qwen2.5 ChatML format
        batch_size=8,
        device="cuda" if _cuda_available() else "mps" if _mps_available() else "cpu",
        log_samples=False,
    )

    # print formatted table
    print(make_table(results))

    # extract the key metrics
    gsm_acc  = results["results"]["gsm8k_cot"].get("exact_match,flexible-extract",
               results["results"]["gsm8k_cot"].get("exact_match,strict-match", 0))
    gpqa_acc = results["results"]["gpqa_main"].get("acc_norm,none",
               results["results"]["gpqa_main"].get("acc,none", 0))

    summary = {
        "model":   model_path,
        "gsm8k":   round(float(gsm_acc),  4),
        "gpqa":    round(float(gpqa_acc), 4),
        "raw":     results["results"],
    }

    # save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGSM8K  accuracy: {gsm_acc:.2%}")
    print(f"GPQA   accuracy: {gpqa_acc:.2%}")
    print(f"\nResults saved → {output_path}")

    return summary


# ── comparison ────────────────────────────────────────────────────────────────
def compare(baseline_path: str, finetuned_path: str):
    """Print a before/after comparison table from two result JSON files."""
    with open(baseline_path)  as f: baseline  = json.load(f)
    with open(finetuned_path) as f: finetuned = json.load(f)

    print(f"\n{'='*55}")
    print(f"  Results comparison")
    print(f"{'='*55}")
    print(f"  Baseline : {baseline['model']}")
    print(f"  Finetuned: {finetuned['model']}")
    print(f"{'='*55}")
    print(f"{'Dataset':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Change':>10}")
    print(f"{'-'*55}")

    for dataset in ["gsm8k", "gpqa"]:
        b     = baseline[dataset]
        ft    = finetuned[dataset]
        delta = ft - b
        arrow = "↑" if delta >= 0 else "↓"
        print(f"{dataset:<12} {b:>9.2%} {ft:>11.2%}   {arrow}{abs(delta):>6.2%}")

    print(f"{'='*55}\n")

    # also save comparison to file
    comparison = {
        "baseline_model":  baseline["model"],
        "finetuned_model": finetuned["model"],
        "gsm8k": {
            "baseline":  baseline["gsm8k"],
            "finetuned": finetuned["gsm8k"],
            "delta":     round(finetuned["gsm8k"] - baseline["gsm8k"], 4),
        },
        "gpqa": {
            "baseline":  baseline["gpqa"],
            "finetuned": finetuned["gpqa"],
            "delta":     round(finetuned["gpqa"] - baseline["gpqa"], 4),
        },
    }
    out = "results/comparison.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved → {out}")


# ── helpers ───────────────────────────────────────────────────────────────────
def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def _mps_available():
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 on GSM8K and GPQA")

    subparsers = parser.add_subparsers(dest="command")

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Run evaluation on a model")
    eval_parser.add_argument("--model",  required=True, help="HF model name or local path")
    eval_parser.add_argument("--output", required=True, help="Path to save results JSON")

    # compare subcommand
    cmp_parser = subparsers.add_parser("compare", help="Compare two result files")
    cmp_parser.add_argument("baseline",  help="Path to baseline results JSON")
    cmp_parser.add_argument("finetuned", help="Path to fine-tuned results JSON")

    # fallback: allow --model/--output at top level for convenience
    parser.add_argument("--model",   help="HF model name or local path")
    parser.add_argument("--output",  help="Path to save results JSON")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "FINETUNED"),
                        help="Compare two result JSON files")

    args = parser.parse_args()

    if args.command == "eval" or (args.model and args.output):
        model  = args.model  if args.command == "eval" else args.model
        output = args.output if args.command == "eval" else args.output
        run_eval(model, output)

    elif args.command == "compare" or args.compare:
        b, ft = (args.baseline, args.finetuned) if args.command == "compare" else args.compare
        compare(b, ft)

    else:
        parser.print_help()
        sys.exit(1)