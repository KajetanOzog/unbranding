#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os


def gold_of(rec: dict):
    """Task-specific ground truth that, with the prompt, uniquely keys a record."""
    task = rec["task"]
    if task in ("benchmark", "scenario"):
        return sorted(rec["expected_brands"])
    if task == "choices":
        return rec["answer"]
    if task == "thesis":
        return rec["label"]
    if task == "forget":
        return None            # forget has no ground truth; keyed on prompt alone
    if task in ("retain", "world_facts"):
        return rec["reference"]
    raise ValueError(f"unknown task: {task!r}")


def make_id(rec: dict) -> str:
    cat = rec.get("brand_category") or "world"
    brand = rec.get("brand") or "all"
    task = rec["task"]
    payload = rec["prompt"] + "\x00" + json.dumps(gold_of(rec), ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{cat}__{brand}__{task}__{h}"


def eval_files(eval_dir: str):
    return sorted(glob.glob(os.path.join(eval_dir, "**", "*.jsonl"), recursive=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="dataset/eval")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    seen = {}          # id -> running count, guarantees physical-row uniqueness
    changed = total = collisions = 0
    samples = []

    for fp in eval_files(args.eval_dir):
        rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        for r in rows:
            total += 1
            new_id = make_id(r)
            if new_id in seen:
                seen[new_id] += 1
                new_id = f"{new_id}__{seen[new_id]}"
                collisions += 1
            else:
                seen[new_id] = 1
            if new_id != r.get("id"):
                changed += 1
                if len(samples) < 6:
                    samples.append((fp, r.get("id"), new_id))
                r["id"] = new_id
        if not args.check:
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    action = "CHECK" if args.check else "WROTE"
    print(f"{action}: {total} records, {changed} ids changed, "
          f"{collisions} collisions suffixed, {len(seen)} unique ids")
    for fp, old, new in samples:
        print(f"  {fp}\n    {old}  ->  {new}")


if __name__ == "__main__":
    main()
