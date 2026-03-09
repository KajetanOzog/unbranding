import json
import os
import glob
from collections import defaultdict

# Base directories for both methods
LLMAJ_RESULTS_DIR = "/net/tscratch/people/plgvltkv/unbranding/llmaj_results"
SPACY_RESULTS_DIR = "/net/tscratch/people/plgvltkv/unbranding/spacy_results"
OUTPUT_REPORT_PATH = "/net/tscratch/people/plgvltkv/unbranding/results/comparison_report.json"

def get_all_jsonl_files(base_dir):
    """
    Recursively finds all .jsonl files in the given directory.
    """
    search_pattern = os.path.join(base_dir, "**", "*.jsonl")
    return glob.glob(search_pattern, recursive=True)

def normalize_entity(text):
    """
    Normalizes text to lowercase and strips whitespaces for fair comparison.
    """
    return str(text).lower().strip()

def load_and_map_data():
    """
    Loads data from both methods and maps them using the response text as the key.
    This ensures we are comparing the exact same model generations.
    """
    data_map = defaultdict(lambda: {
        "prompt": "",
        "llmaj_brands": set(),
        "llmaj_trade_dress": set(),
        "spacy_entities": set()
    })

    # Load LLMAJ results
    llmaj_files = get_all_jsonl_files(LLMAJ_RESULTS_DIR)
    print(f"Found {len(llmaj_files)} LLMAJ result files.")
    
    for filepath in llmaj_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    resp_text = row.get("response", "")
                    prompt_text = row.get("prompt", "")
                    
                    data_map[resp_text]["prompt"] = prompt_text
                    
                    judge = row.get("judge", {})
                    brands = judge.get("brand_names", [])
                    trade_dress = judge.get("trade_dress_brands", [])
                    
                    # Normalize and add to sets
                    data_map[resp_text]["llmaj_brands"].update(
                        [normalize_entity(b) for b in brands]
                    )
                    data_map[resp_text]["llmaj_trade_dress"].update(
                        [normalize_entity(td) for td in trade_dress]
                    )
                except Exception as e:
                    continue

    # Load SpaCy results
    spacy_files = get_all_jsonl_files(SPACY_RESULTS_DIR)
    print(f"Found {len(spacy_files)} SpaCy result files.")
    
    for filepath in spacy_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    resp_text = row.get("response", "")
                    
                    entities = row.get("detected_entities", [])
                    data_map[resp_text]["spacy_entities"].update(
                        [normalize_entity(e["text"]) for e in entities]
                    )
                except Exception as e:
                    continue

    return data_map

def main():
    print("Loading and matching data...")
    data_map = load_and_map_data()
    
    report = []
    
    total_responses = len(data_map)
    llmaj_found_more = 0
    spacy_found_more = 0
    perfect_match = 0

    print("Comparing extracted entities...")
    for response_text, data in data_map.items():
        # Combine LLMAJ brand names and trade dress for general comparison
        llmaj_all_entities = data["llmaj_brands"].union(data["llmaj_trade_dress"])
        spacy_all_entities = data["spacy_entities"]

        # Find differences
        llmaj_only = list(llmaj_all_entities - spacy_all_entities)
        spacy_only = list(spacy_all_entities - llmaj_all_entities)
        intersection = list(llmaj_all_entities.intersection(spacy_all_entities))

        if llmaj_only:
            llmaj_found_more += 1
        if spacy_only:
            spacy_found_more += 1
        if not llmaj_only and not spacy_only:
            perfect_match += 1

        if llmaj_only or spacy_only:
            report.append({
                "prompt": data["prompt"],
                "response": response_text,
                "intersection_both_found": intersection,
                "only_llmaj_found": {
                    "all": llmaj_only,
                    "trade_dress_specifically": list(data["llmaj_trade_dress"] - spacy_all_entities)
                },
                "only_spacy_found": spacy_only
            })

    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)


    print("\n" + "="*30)
    print("COMPARISON SUMMARY")
    print("="*30)
    print(f"Total unique responses analyzed: {total_responses}")
    print(f"Responses with perfect agreement: {perfect_match}")
    print(f"Responses where LLMAJ found something SpaCy missed: {llmaj_found_more}")
    print(f"Responses where SpaCy found something LLMAJ missed: {spacy_found_more}")
    print(f"\nDetailed discrepancy report saved to: {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    main()