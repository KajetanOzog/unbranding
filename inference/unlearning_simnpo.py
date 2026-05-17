import sys
import os
import types
import importlib.machinery

# --- KROK 1: RATOWANIE ŚCIEŻEK (Torch z kontenera ma pierwszeństwo) ---
sys_paths = [
    "/usr/local/lib64/python3.13/site-packages",
    "/usr/local/lib/python3.13/site-packages",
    "/usr/lib64/python3.13/site-packages"
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

extra_path = "/net/scratch/hscra/plgrid/plgvltkv/unbranding/extra_python_libs"
if extra_path not in sys.path:
    sys.path.append(extra_path)

# --- KROK 2: KOMPLEKSOWA ATRAPA TORCHVISION (Uciszenie błędów wizyjnych) ---
def mock_torchvision_package():
    module_names = [
        "torchvision", "torchvision.transforms", "torchvision.transforms.v2", 
        "torchvision.transforms.v2.functional", "torchvision.io", "torchvision.ops", "torchvision.models"
    ]
    class MockInterpolationMode:
        NEAREST = "nearest"; BILINEAR = "bilinear"; BICUBIC = "bicubic"
        BOX = "box"; HAMMING = "hamming"; LANCZOS = "lanczos"; NEAREST_EXACT = "nearest_exact"

    for name in module_names:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [] 
            m.__spec__ = importlib.machinery.ModuleSpec(name, None)
            sys.modules[name] = m
    sys.modules["torchvision.transforms"].InterpolationMode = MockInterpolationMode
    sys.modules["torchvision.transforms.v2"].InterpolationMode = MockInterpolationMode
    sys.modules["torchvision.transforms.v2"].functional = sys.modules["torchvision.transforms.v2.functional"]

mock_torchvision_package()

# --- KROK 3: IMPORTY I MONKEY-PATCHING (Fix dla ARM CPU / Byte type) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    print(f"--- SUCCESS: Python confirmed GPU: {torch.cuda.get_device_name(0)} ---")
else:
    print("--- CRITICAL ERROR: GPU not visible! ---")
    sys.exit(1)

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, 
    TrainingArguments, Trainer, AutoConfig, Gemma2Config, GenerationConfig
)

# MONKEY PATCH: Wyłączamy inicjalizację wag dla Gemma 2, aby uniknąć błędu "normal_kernel_cpu"
try:
    import transformers.models.gemma2.modeling_gemma2 as gemma2_mod
    def skip_init(self, module): return
    gemma2_mod.Gemma2PreTrainedModel._init_weights = skip_init
    print("SUCCESS: Weight initialization bypassed for Gemma 2/4.")
except Exception as e:
    print(f"Warning: Could not patch Gemma 2: {e}")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import argparse

# --- KLASA TRAINERA SimNPO ---
class SimNPOTrainer(Trainer):
    def __init__(self, *args, beta=0.1, gamma=0.5, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha

    def get_log_probs(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2).clamp(min=0)).squeeze(2)
        mask = (shift_labels != -100)
        return (per_token_logps * mask).sum(dim=-1), mask.sum(dim=-1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forget loss (SimNPO)
        outputs_f = model(input_ids=inputs["forget_input_ids"], attention_mask=inputs["forget_attention_mask"])
        log_probs_f, lengths_f = self.get_log_probs(outputs_f.logits, inputs["forget_labels"])
        sim_npo_logits = - (self.beta / (lengths_f + 1e-8)) * log_probs_f - self.gamma
        loss_forget = - (2 / self.beta) * F.logsigmoid(sim_npo_logits).mean()
        
        # Retain loss (Standard)
        outputs_r = model(input_ids=inputs["retain_input_ids"], attention_mask=inputs["retain_attention_mask"], labels=inputs["retain_labels"])
        loss_retain = outputs_r.loss
        
        total_loss = loss_forget + self.alpha * loss_retain
        return (total_loss, outputs_f) if return_outputs else total_loss

def preprocess_function(examples, tokenizer, max_length=512):
    def tokenize_pair(prompt, response):
        full_text = f"{prompt}{response}{tokenizer.eos_token}"
        tokenized_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        tokenized_full = tokenizer(full_text, add_special_tokens=False, max_length=max_length, truncation=True)
        labels = list(tokenized_full["input_ids"])
        prompt_len = len(tokenized_prompt)
        for i in range(min(prompt_len, len(labels))): labels[i] = -100
        return tokenized_full["input_ids"], tokenized_full["attention_mask"], labels

    batch = {"forget_input_ids": [], "forget_attention_mask": [], "forget_labels": [], "retain_input_ids": [], "retain_attention_mask": [], "retain_labels": []}
    for f_p, f_r, r_p, r_r in zip(examples["forget_prompt"], examples["forget_answer"], examples["retain_prompt"], examples["retain_answer"]):
        f_ids, f_mask, f_labels = tokenize_pair(f_p, f_r)
        r_ids, r_mask, r_labels = tokenize_pair(r_p, r_r)
        def pad(lst, val): return lst + [val] * (max_length - len(lst))
        p_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        batch["forget_input_ids"].append(pad(f_ids, p_id)); batch["forget_attention_mask"].append(pad(f_mask, 0)); batch["forget_labels"].append(pad(f_labels, -100))
        batch["retain_input_ids"].append(pad(r_ids, p_id)); batch["retain_attention_mask"].append(pad(r_mask, 0)); batch["retain_labels"].append(pad(r_labels, -100))
    return {k: torch.tensor(v) for k, v in batch.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--bad_data_path", type=str, required=True)
    parser.add_argument("--good_data_path", type=str, required=True)
    parser.add_argument("--output_base_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2); parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    def load_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f: return [json.loads(line) for line in f]
    
    bad_raw = load_jsonl(args.bad_data_path); good_raw = load_jsonl(args.good_data_path)
    min_len = min(len(bad_raw), len(good_raw))
    combined_raw = {
        "forget_prompt": [d["prompt"] for d in bad_raw[:min_len]], 
        "forget_answer": [d["response"] for d in bad_raw[:min_len]], 
        "retain_prompt": [d["prompt"] for d in good_raw[:min_len]], 
        "retain_answer": [d["response"] for d in good_raw[:min_len]]
    }

    for model_path in args.model_paths:
        model_name = os.path.basename(model_path.rstrip("/"))
        current_output_dir = os.path.join(args.output_base_dir, f"unbranded_{model_name}")
        print(f"\n--- PROCESSING: {model_name} ---")

        # 1. Patchowanie konfiguracji
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if "gemma" in model_path.lower():
            if not isinstance(config, Gemma2Config): config = Gemma2Config.from_pretrained(model_path)
            if hasattr(config, "text_config"): delattr(config, "text_config")

        # 2. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        # 3. Model
        print(f"Loading weights for {model_name}...")
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        gen_config = GenerationConfig.from_pretrained(model_path) if os.path.exists(os.path.join(model_path, "generation_config.json")) else None

        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, quantization_config=quant_config, device_map="auto", 
            trust_remote_code=True, generation_config=gen_config, low_cpu_mem_usage=True
        )

        # 4. LoRA - Ujednolicone target_modules dla wszystkich modeli
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = prepare_model_for_kbit_training(model); model = get_peft_model(model, lora_config)

        # 5. Dataset
        dataset = Dataset.from_dict(combined_raw).map(
            lambda x: preprocess_function(x, tokenizer), 
            batched=True, 
            remove_columns=["forget_prompt", "forget_answer", "retain_prompt", "retain_answer"]
        )

        # 6. Training Arguments - KLUCZOWA POPRAWKA remove_unused_columns
        training_args = TrainingArguments(
            output_dir=current_output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            remove_unused_columns=False,  # <--- TO ROZWIĄZUJE TWÓJ OSTATNI BŁĄD
            logging_steps=5,
            save_strategy="no",
            report_to="none"
        )

        trainer = SimNPOTrainer(model=model, args=training_args, train_dataset=dataset)
        
        print("Starting training loop...")
        trainer.train()
        
        model.save_pretrained(current_output_dir)
        tokenizer.save_pretrained(current_output_dir)
        del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()