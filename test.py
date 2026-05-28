import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a helpful mental health counselling assistant. "
    "Answer the mental health questions based on the patient's description. "
    "Give helpful, comprehensive, and appropriate answers."
)


def build_prompt(user_input: str) -> str:
    return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_input} [/INST]"


def eval_mentalchat(model_path, device, n_samples=200):
    from Data import combined_test

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16
    ).to(device).eval()

    test = combined_test.select(range(min(n_samples, len(combined_test))))

    results = []
    for ex in tqdm(test, desc="MentalChat eval"):
        # ex["text"] is the full formatted example — strip off the response
        # by splitting on [/INST] and re-adding the tag to prime generation
        prompt = ex["text"].split("[/INST]")[0] + "[/INST]"
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        results.append({"input": prompt, "response": response})

    return results


def run_eval(model_path, output_path, n_samples=200):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    results = eval_mentalchat(model_path, device, n_samples)

    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"model": model_path, "n_samples": len(results), "responses": results}, f, indent=2)
    print(f"Saved {len(results)} responses → {output_path}")


def compare(baseline_path, finetuned_path):
    with open(baseline_path)  as f: b  = json.load(f)
    with open(finetuned_path) as f: ft = json.load(f)
    print(f"\nBaseline  ({b['model']}):   {b['n_samples']} samples")
    print(f"Fine-tuned ({ft['model']}): {ft['n_samples']} samples")
    print("Use an LLM-as-judge or ROUGE/BERTScore to compare response quality.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   help="HF model name or local path to evaluate")
    p.add_argument("--output",  help="Path to save results JSON")
    p.add_argument("--samples", type=int, default=200, help="Number of test examples to evaluate")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "FINETUNED"),
                   help="Compare two saved result JSONs")
    args = p.parse_args()

    if args.model and args.output: run_eval(args.model, args.output, args.samples)
    elif args.compare:             compare(*args.compare)
    else:                          p.print_help()
