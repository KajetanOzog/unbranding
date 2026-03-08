import json
import spacy
import os
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# RESPONSES_PATH = "SCIEZKA DO WYNIKOW" 
RESPONSES_PATH = "/net/tscratch/people/plgvltkv/unbranding/results/llama/food_resp.jsonl"

OUTPUT_DIR = "/net/tscratch/people/plgvltkv/unbranding/results"
OUTPUT_FILE = "results.jsonl"


def extract_brands(text):
    doc = nlp(text)
    brands = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            brands.append(ent.text)

    return brands


def main():
    # ✅ Tworzenie katalogu jeśli nie istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    prompts_with_brands = 0
    total_brands = 0
    brand_counter = Counter()

    with open(RESPONSES_PATH, "r", encoding="utf-8") as infile, \
         open(full_output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            data = json.loads(line)
            prompt = data.get("prompt", "")
            response = data.get("response", "")

            brands = extract_brands(response)

            if brands:
                prompts_with_brands += 1
                total_brands += len(brands)
                brand_counter.update(brands)

            result = {
                "prompt": prompt,
                "response": response,
                "brands": brands
            }

            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n===== PODSUMOWANIE =====")
    print(f"Liczba promptów z markami: {prompts_with_brands}")
    print(f"Łączna liczba wykrytych marek: {total_brands}")

    print("\nNajczęściej występujące marki:")
    for brand, count in brand_counter.most_common():
        print(f"{brand}: {count}")

    print(f"\nPlik zapisany w: {full_output_path}")


if __name__ == "__main__":
    main()