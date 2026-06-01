from datasets import load_dataset
import json

def prepare_alpaca_data(output_file="alpaca_good_data.jsonl", num_samples=3000):
    print("Pobieranie zbioru Alpaca...")
    # Ładujemy zbiór danych
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Wybieramy tylko pierwsze N przykładów
    subset = dataset.select(range(num_samples))
    
    print(f"Przetwarzanie {num_samples} rekordów...")
    transformed_data = []
    
    for example in subset:
        # Logika Alpaca: jeśli 'input' istnieje, dodajemy go do instrukcji
        if example["input"] and example["input"].strip() != "":
            full_prompt = f"{example['instruction']}\n\nInput: {example['input']}"
        else:
            full_prompt = example["instruction"]
            
        transformed_data.append({
            "prompt": full_prompt,
            "response": example["output"]
        })
    
    # Zapis do formatu JSONL (linia po linii)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in transformed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Gotowe! Dane zapisano w: {output_file}")

if __name__ == "__main__":
    prepare_alpaca_data()