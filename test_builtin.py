import json, os, sys, torch
from lm_eval import simple_evaluate
from lm_eval.utils import make_table

# ---------------------------------------------------------------------------
# Usage:
#   Evaluate a model:   python test_builtin.py eval  <model_path> <output.json>
#   Compare two runs:   python test_builtin.py compare <baseline.json> <finetuned.json>
# ---------------------------------------------------------------------------

# sys.argv holds all command-line arguments as a list: [script, command, ...]
mode = sys.argv[1] if len(sys.argv) > 1 else "help"

# ── EVAL MODE ───────────────────────────────────────────────────────────────
if mode == "eval":
    model_path  = sys.argv[2]   # HF model name or local path
    output_path = sys.argv[3]   # where to save the results JSON

    # Pick the best available device: GPU > Apple Silicon > CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Build model arg string; append HF token from environment if present
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

# ── COMPARE MODE ─────────────────────────────────────────────────────────────
elif mode == "compare":
    # Load both result JSONs using the built-in open() and json.load()
    b  = json.load(open(sys.argv[2]))   # baseline
    ft = json.load(open(sys.argv[3]))   # fine-tuned

    # Print a side-by-side delta table
    print(f"\n{'='*50}")
    print(f"{'Dataset':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Change':>10}")
    print(f"{'-'*50}")
    for ds in ["gsm8k", "gpqa"]:
        delta = ft[ds] - b[ds]
        print(f"{ds:<12} {b[ds]:>9.2%} {ft[ds]:>11.2%}   {'↑' if delta >= 0 else '↓'}{abs(delta):>6.2%}")
    print(f"{'='*50}\n")

# ── FALLBACK ─────────────────────────────────────────────────────────────────
else:
    # print() is a built-in — no imports needed for basic output
    print("Usage:")
    print("  python test_builtin.py eval    <model_path> <output.json>")
    print("  python test_builtin.py compare <baseline.json> <finetuned.json>")
