import json
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description="Sort JSONL by 'prompt' and remove duplicate prompts.")
parser.add_argument("input_file", help="Path to JSONL file")

args = parser.parse_args()
input_path = Path(args.input_file)

data = []

# wczytanie pliku
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# sortowanie
data_sorted = sorted(data, key=lambda x: x.get("question", ""))

# deduplikacja
seen = set()
deduped = []

for item in data_sorted:
    prompt = item.get("question", "")
    if prompt not in seen:
        seen.add(prompt)
        deduped.append(item)

# zapis do pliku tymczasowego
temp_path = input_path.with_suffix(".tmp")

with open(temp_path, "w", encoding="utf-8") as f:
    for item in deduped:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# nadpisanie oryginalnego pliku
os.replace(temp_path, input_path)

print(f"Original entries: {len(data)}")
print(f"After deduplication: {len(deduped)}")
print(f"File overwritten: {input_path}")