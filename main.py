from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from Data import combined_train
import torch

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# 4-bit quantisation (QLoRA) to fit Mistral-7B in GPU memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

args = SFTConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size = 16
    learning_rate=2e-4,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    max_length=512,
    fp16=False,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=combined_train,
    peft_config=lora,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./tuned-mistral")
