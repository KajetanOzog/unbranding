import os
import json
import spacy
import argparse
from pathlib import Path

# Ładowanie modelu SpaCy
try:
    print("Loading SpaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")
except:
    print("Error: Model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    exit(1)

def process_file(input_path, output_path):
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Analiza SpaCy na surowym response (zgodnie z judge)
            text_to_analyze = data.get("response", "")
            doc = nlp(text_to_analyze)
            
            entities = []
            brand_names = []
            trade_dress = []

            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "PERSON"]:
                    entities.append({"text": ent.text, "label": ent.label_})
                    
                    # Mapowanie: ORG -> brand_names, reszta -> trade_dress_brands
                    if ent.label_ == "ORG":
                        brand_names.append(ent.text)
                    else:
                        trade_dress.append(ent.text)

            # Tworzymy wynik w nowym formacie
            results.append({
                "prompt": data.get("prompt", ""),
                "response": data.get("response", ""),
                "detected_entities": entities,
                "parsed": {
                    "brand_names": brand_names,
                    "trade_dress_brands": trade_dress
                }
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run SpaCy analysis and save to judge_results/spacy.")
    parser.add_argument("--results_dir", required=True, help="Path to 'model_outputs_timestamp' folder")
    args = parser.parse_args()

    base_path = Path(args.results_dir).resolve()
    
    # Cel: experiments_results/model_outputs_timestamp/judge_results/spacy
    spacy_base_dir = base_path / "judge_results" / "spacy"
    
    # Szukamy folderów modeli (wszystkie poza judge_results)
    model_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name != "judge_results"]

    if not model_folders:
        print(f"No model folders found in {base_path}")
        return

    for m_folder in model_folders:
        model_name = m_folder.name
        print(f"Processing model: {model_name}")
        
        # Przechodzimy przez brand_categories (np. bev, food, auto)
        for brand_cat_dir in m_folder.iterdir():
            if not brand_cat_dir.is_dir(): continue
            brand_cat_name = brand_cat_dir.name
            
            # Przygotowanie folderu w judge_results/spacy/model/brand_category/
            # prompt_category zostanie stworzone automatycznie przez pliki lub podfoldery
            
            # Przetwarzanie plików .jsonl (które często reprezentują prompt_category)
            for file in brand_cat_dir.glob("**/*.jsonl"):
                # Obliczamy relatywną ścieżkę, aby zachować strukturę prompt_category
                rel_path = file.relative_to(brand_cat_dir)
                
                out_dir = spacy_base_dir / model_name / brand_cat_name / rel_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = out_dir / file.name
                print(f"  Analysing: {model_name}/{brand_cat_name}/{rel_path}")
                
                process_file(file, output_file)

    print(f"\nDone! SpaCy results are in: {spacy_base_dir}")

if __name__ == "__main__":
    main()