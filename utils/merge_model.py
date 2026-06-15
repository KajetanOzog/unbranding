import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Lista modeli do mergowania
MODELS = [
    {
        "name": "qwen-32b",
        "base_path": "/net/scratch/hscra/plgrid/plgvltkv/unbranding/models/qwen-32b",
        "lora_path": "/net/scratch/hscra/plgrid/plgvltkv/unbranding/models/unlearned/qwen-32b/checkpoint-125",
        "save_path": "/net/scratch/hscra/plgrid/plgvltkv/unbranding/models/unlearned/simnpo_qwen-32b_merged",
    },
]

for cfg in MODELS:
    print("=" * 80)
    print(f"Processing model: {cfg['name']}")

    base_path = cfg["base_path"]
    lora_path = cfg["lora_path"]
    save_path = cfg["save_path"]

    # Sprawdzenie ścieżek
    if not os.path.exists(base_path):
        print(f"[ERROR] Base model path does not exist: {base_path}")
        continue

    if not os.path.exists(lora_path):
        print(f"[ERROR] LoRA path does not exist: {lora_path}")
        continue

    os.makedirs(save_path, exist_ok=True)

    print(f"Loading base model from {base_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # merge na CPU
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapters from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_path)

    print(f"[DONE] {cfg['name']} merged successfully!")

print("=" * 80)
print("All models processed.")