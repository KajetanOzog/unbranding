import json
import spacy
import os
from datetime import date
from collections import Counter

RESPONSES_PATHS = [
    "/net/tscratch/people/plgvltkv/unbranding/results/llama/auto_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/llama/bev_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/llama/food_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/llama/sport_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/llama/tech_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/qwen/auto_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/qwen/bev_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/qwen/food_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/qwen/sport_resp.jsonl",
    "/net/tscratch/people/plgvltkv/unbranding/results/qwen/tech_resp.jsonl",
]

print("Loading SpaCy model (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm")

def extract_brands_and_trade_dress(text):
    """
    Uses SpaCy to extract potential brands and products.
    Checks for 'ORG' (organizations) and 'PRODUCT' (products) labels.
    """
    doc = nlp(text)
    detected_entities = []

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            detected_entities.append({
                "text": ent.text,
                "label": ent.label_
            })

    return detected_entities

def load_responses(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_output_path(responses_path):
    """
    Builds the output path similarly to the first script,
    but saves the results to a dedicated spacy folder.
    """
    model_name = os.path.basename(os.path.dirname(responses_path))
    filename = os.path.basename(responses_path)
    category = filename.replace("_resp.jsonl", "")
    today = date.today().isoformat()

    # Changed target directory to spacy_results
    output_dir = f"/net/tscratch/people/plgvltkv/unbranding/spacy_results/{model_name}/any_brand_check"
    os.makedirs(output_dir, exist_ok=True)

    return f"{output_dir}/{category}_{today}.jsonl"

def main():
    total_responses_processed = 0
    total_brands_found = 0
    responses_with_brands = 0
    brand_counter = Counter()

    for responses_path in RESPONSES_PATHS:
        if not os.path.exists(responses_path):
            print(f"WARNING: File {responses_path} does not exist. Skipping.")
            continue

        print(f"\nProcessing: {responses_path}")
        responses = load_responses(responses_path)
        results = []

        file_brands_count = 0

        for r in responses:
            # Clean the response from tokens like [BLANK]
            response_text = r.get("response", "").replace("[BLANK]", "")
            
            # Extraction using SpaCy
            entities = extract_brands_and_trade_dress(response_text)

            brands_only_texts = [e["text"] for e in entities]
            
            if entities:
                responses_with_brands += 1
                file_brands_count += len(entities)
                total_brands_found += len(entities)
                brand_counter.update(brands_only_texts)

            results.append({
                "prompt": r.get("prompt", ""),
                "response": response_text,
                "detected_entities": entities,
                "has_brand": len(entities) > 0
            })
            total_responses_processed += 1

        output_path = build_output_path(responses_path)
        print(f"Saving results to: {output_path} (Found entities: {file_brands_count})")

        with open(output_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "="*20 + " SUMMARY " + "="*20)
    print(f"Total responses processed: {total_responses_processed}")
    print(f"Responses containing potential brands/products: {responses_with_brands}")
    print(f"Total number of detected entities: {total_brands_found}")
    
    print("\nTop 10 most frequently detected entities:")
    for brand, count in brand_counter.most_common(10):
        print(f" - {brand}: {count}")

if __name__ == "__main__":
    main()