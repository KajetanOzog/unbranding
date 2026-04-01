import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompts_dir", type=str, default="prompts")
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()

def inference(model, tokenizer, question):
    # Tworzymy strukturę rozmowy dla modeli typu Instruct/Chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question"},
        {"role": "user", "content": question},
    ]

    # Próbujemy użyć chat_template, jeśli model go wspiera
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        # Jeśli model to wersja "base" (nie-chat), używamy prostego schematu:
        prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20, # Krótkie odpowiedzi są zazwyczaj lepsze dla faktów
            temperature=0,    # Wyłączamy losowość dla powtarzalnych wyników
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Dekodujemy tylko NOWE tokeny (odpowiedź)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Czyszczenie odpowiedzi (model czasem dodaje "Answer: " na początku)
    if "\n" in result: # Jeśli model się rozpisał, bierzemy tylko pierwszą linię
        result = result.split("\n")[0]
    
    return result.replace("Answer:", "").strip()

def main():
    args = get_args()
    torch.manual_seed(42)

    print(f"Ładowanie modelu...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()

    total_prompts = 0

    for root, dirs, files in os.walk(args.prompts_dir):
        for filename in files:
            if filename.startswith('.'): continue

            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(root, args.prompts_dir)
            target_folder = os.path.join(args.output_dir, rel_path)
            os.makedirs(target_folder, exist_ok=True)
            
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(target_folder, f"results-{base_name}.jsonl")

            print(f"Przetwarzanie: {input_path}")
            
            try:
                with open(input_path, "r", encoding="utf-8") as f_in, \
                     open(output_path, "w", encoding="utf-8") as f_out:
                    
                    for line in f_in:
                        line = line.strip()
                        if not line: continue
                        
                        data = json.loads(line)
                        question = data.get("prompt", "")
                        
                        if question:
                            # Główna zmiana: bezpośrednie pytanie do modelu
                            response = inference(model, tokenizer, question)
                            data["model_response"] = response
                            
                            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                            total_prompts += 1
                
            except Exception as e:
                print(f"  ! Błąd w {filename}: {e}")

    print(f"\nSukces! Wygenerowano {total_prompts} odpowiedzi.")

if __name__ == "__main__":
    main()