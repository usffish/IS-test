# Fine-Tuning Qwen2.5-1.5B on GSM8K & GPQA

An independent study project exploring supervised fine-tuning (SFT) of a small language model using parameter-efficient methods (LoRA) on math and science reasoning benchmarks.

## Overview

This project fine-tunes **Qwen/Qwen2.5-1.5B** using **LoRA** (Low-Rank Adaptation) on a combined dataset of:

- **GSM8K** — grade school math word problems with chain-of-thought solutions
- **GPQA** — graduate-level multiple choice questions across biology, chemistry, and physics

The goal is to measure how much reasoning ability a 1.5B parameter model can gain from targeted SFT on these two datasets, and to compare pre- and post-fine-tuning performance.

## Results

Evaluated using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with chain-of-thought prompting on GSM8K and normalized accuracy on GPQA.

| Dataset | Baseline (Qwen2.5-1.5B) | Fine-tuned | Δ |
|---|---|---|---|
| GSM8K (`exact_match`) | — | — | — |
| GPQA (`acc_norm`) | — | — | — |

> Results will be filled in after running `python test.py` against both models (see [Evaluation](#evaluation) below).

## Project Structure

```
.
├── main.py           # Training script (LoRA + SFT via TRL)
├── Data.py           # Dataset loading, formatting, and splitting
├── test.py           # Evaluation script using lm-evaluation-harness
├── requirements.txt  # Python dependencies
└── output/           # Training checkpoints (git-ignored; see note below)
```

> **Model weights** are not committed to this repo due to file size. The fine-tuned adapter and merged model can be loaded from the saved local path or uploaded to Hugging Face Hub.

## Setup

**Prerequisites:** Python 3.10+, a CUDA-capable GPU (bf16 training is enabled by default).

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Set your Hugging Face token as an environment variable (required to download GPQA):

```bash
export HF_TOKEN=your_token_here
```

## Training

```bash
python main.py
```

The script will:
1. Load Qwen2.5-1.5B in bf16
2. Wrap it with a LoRA adapter (rank 16, alpha 32) targeting the attention projections
3. Train for 1 epoch on the combined GSM8K + GPQA training split
4. Save the merged model to `./tuned-qwen`

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 1 |
| Batch size | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup steps | 50 |
| Max sequence length | 512 |
| Precision | bf16 |

## Evaluation

The `test.py` script uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to benchmark any model on `gsm8k_cot` and `gpqa_main`.

**Step 1 — Evaluate the baseline:**
```bash
python test.py --model Qwen/Qwen2.5-1.5B --output results/baseline.json
```

**Step 2 — Evaluate the fine-tuned model:**
```bash
python test.py --model ./tuned-qwen --output results/finetuned.json
```

**Step 3 — Compare:**
```bash
python test.py --compare results/baseline.json results/finetuned.json
```

Output format:
```
==================================================
Dataset      Baseline   Fine-tuned     Change
--------------------------------------------------
gsm8k          xx.xx%      xx.xx%      ↑x.xx%
gpqa           xx.xx%      xx.xx%      ↑x.xx%
==================================================
```

## Data Pipeline

`Data.py` handles all dataset preparation:

- **GSM8K** is formatted as a user/assistant chat using the `<|im_start|>` / `<|im_end|>` template native to Qwen
- **GPQA** answer choices are shuffled randomly (seed 42) before formatting to prevent position bias
- GPQA is split 80/20 into train/test; GSM8K uses its native splits
- Both training sets are concatenated into a single `combined_train` dataset

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model training and inference |
| `transformers` | Model and tokenizer loading |
| `trl` | SFTTrainer for supervised fine-tuning |
| `peft` | LoRA adapter configuration |
| `datasets` | Dataset loading and processing |
| `accelerate` | Distributed/mixed-precision training |
| `huggingface_hub` | Model and dataset access |
