import argparse, json, os, torch
from lm_eval import simple_evaluate
from lm_eval.utils import make_table


def run_eval(model_path, output_path):
    # Pick the best available device: GPU > Apple Silicon > CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Build model arg string; append HF token if set in environment
    model_args = f"pretrained={model_path}"
    if os.environ.get("HF_TOKEN"):
        model_args += f",token={os.environ['HF_TOKEN']}"

    # Run evaluation on both benchmarks:
    #   gsm8k_cot  — grade school math with chain-of-thought prompting
    #   gpqa_main  — graduate-level science multiple choice
    results = simple_evaluate(
        model="hf", model_args=model_args,
        tasks=["gsm8k_cot", "gpqa_main"],
        apply_chat_template=True, batch_size=8,
        device=device, log_samples=False,
    )
    print(make_table(results))

    # Extract the primary metric for each task
    gsm_acc  = results["results"]["gsm8k_cot"].get("exact_match,flexible-extract", 0)
    gpqa_acc = results["results"]["gpqa_main"].get("acc_norm,none", 0)

    # Save a compact summary JSON for later comparison
    os.makedirs("results", exist_ok=True)
    json.dump({"model": model_path,
               "gsm8k": round(float(gsm_acc), 4),
               "gpqa":  round(float(gpqa_acc), 4)},
              open(output_path, "w"), indent=2)
    print(f"GSM8K: {gsm_acc:.2%}  |  GPQA: {gpqa_acc:.2%}  |  Saved → {output_path}")


def compare(baseline_path, finetuned_path):
    # Load both result JSONs
    b  = json.load(open(baseline_path))
    ft = json.load(open(finetuned_path))

    # Print a side-by-side delta table
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
