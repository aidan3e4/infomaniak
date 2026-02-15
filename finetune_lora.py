"""
LoRA fine-tuning script for Qwen3-4B-Instruct on receipt extraction.
Single GPU, uses TRL + PEFT + bitsandbytes for 4-bit QLoRA.

Usage:
    pip install transformers trl peft bitsandbytes datasets accelerate
    python finetune_lora.py
"""

import json

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ====================
# Configuration
# ====================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_FILE = "receipt_dataset.jsonl"
OUTPUT_DIR = "./receipt-lora-output"
MAX_SEQ_LENGTH = 2048

# LoRA params
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# Training params
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4  # effective batch size = 2 * 4 = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05

# ====================
# Load model + LoRA
# ====================
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ====================
# Load & format dataset
# ====================
print(f"Loading dataset from {DATASET_FILE}...")
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(raw_data)} samples")


def format_chat(sample: dict) -> dict:
    """Format into Qwen chat template: system + user + assistant."""
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

# ====================
# Train
# ====================
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        seed=42,
    ),
)

trainer.train()

# ====================
# Save LoRA adapter
# ====================
ADAPTER_DIR = f"{OUTPUT_DIR}/final-adapter"
print(f"Saving LoRA adapter to {ADAPTER_DIR}...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Done!")
