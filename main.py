from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from Data import combined_train
import torch

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer.pad_token = tokenizer.eos_token

# Wrap with LoRA
lora = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM
)

#model = get_peft_model(model, lora)
#model.print_trainable_parameters()

# config
args = SFTConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    max_length=512,
    fp16=False, bf16=True,
)

#test_set = fgsm["train"].select(range(50))

# Train gpqa
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset = combined_train,
    peft_config=lora,
    processing_class=tokenizer,
)


trainer.train()

trainer.save_model("./tuned-qwen")