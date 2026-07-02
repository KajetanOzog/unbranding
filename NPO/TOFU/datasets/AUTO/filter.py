import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

BASE = Path(".")

FORGET_DIR = BASE / "FORGET"
RETAIN_DIR = BASE / "RETAIN"
ALPACA_FILE = BASE / "filtered_alpaca_5k.jsonl"

EVAL_DIR = BASE / "EVAL"
EVAL_DIR.mkdir(exist_ok=True)

FORGET_TEST_SIZE = 10
RETAIN_CARS_TEST_SIZE = 3
ALPACA_TEST_SIZE = 200


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


########################
# FORGET TEST
########################

forget_test = []

for file in sorted(FORGET_DIR.glob("*.jsonl")):
    data = read_jsonl(file)

    idx = random.sample(range(len(data)), FORGET_TEST_SIZE)

    test = [data[i] for i in idx]
    train = [x for i, x in enumerate(data) if i not in idx]

    forget_test.extend(test)

    write_jsonl(file, train)

write_jsonl(EVAL_DIR / "test_forget.jsonl", forget_test)

print(f"Forget test size: {len(forget_test)}")


########################
# RETAIN CARS TEST
########################

retain_cars_test = []

for file in sorted(RETAIN_DIR.glob("*.jsonl")):
    data = read_jsonl(file)

    idx = random.sample(range(len(data)), RETAIN_CARS_TEST_SIZE)

    test = [data[i] for i in idx]
    train = [x for i, x in enumerate(data) if i not in idx]

    retain_cars_test.extend(test)

    write_jsonl(file, train)

write_jsonl(EVAL_DIR / "test_retain_auto.jsonl", retain_cars_test)

print(f"Retain cars test size: {len(retain_cars_test)}")


########################
# ALPACA TEST
########################

alpaca = read_jsonl(ALPACA_FILE)

idx = random.sample(range(len(alpaca)), ALPACA_TEST_SIZE)

alpaca_test = [alpaca[i] for i in idx]
alpaca_train = [x for i, x in enumerate(alpaca) if i not in idx]

write_jsonl(EVAL_DIR / "test_retain.jsonl", alpaca_test)
write_jsonl(ALPACA_FILE, alpaca_train)

print(f"Alpaca test size: {len(alpaca_test)}")


########################
# SUMMARY
########################

print()
print("Created:")
print("EVAL/test_forget.jsonl")
print("EVAL/test_retain_auto.jsonl")
print("EVAL/test_retain.jsonl")

print()
print("Remaining train sizes:")
print(f"Alpaca: {len(alpaca_train)}")