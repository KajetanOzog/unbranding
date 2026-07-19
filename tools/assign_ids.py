#!/usr/bin/env python3
"""(Re)generate stable, content-derived ids for the core evaluation set.

Operates on the CANONICAL schema (schema.md):
  prompts/eval/<cat>/<BRAND>/benchmark.jsonl
  prompts/eval/<cat>/scenario_*.jsonl

id = "<cat>__<group>__<prompt_category>__<h>"
  group = brand folder name (benchmark) or "scenario"
  h     = sha1(prompt + "\\x00" + "|".join(sorted(expected_brands)))[:8]

So the id is fully reproducible from a record's content: same prompt + same
ground truth => same id. Genuinely identical (prompt, expected_brands) rows in
the same group would collide; those get a numeric suffix so every physical row
stays unique. The script is idempotent — running it twice is a no-op.

Usage:
  python tools/assign_ids.py            # regenerate ids in place
  python tools/assign_ids.py --check    # report changes only, write nothing
"""
import argparse
import glob
import hashlib
import json

# Mirror of config.yaml category_aliases (brand_category is already canonical in
# the migrated data; kept here only as a safety net).
CATEGORY_ALIASES = {"automotive": "auto", "bev": "beverages"}


def canonical_category(raw: str) -> str:
    return CATEGORY_ALIASES.get(raw, raw)


def make_id(cat: str, group: str, pcat: str, prompt: str, expected_brands) -> str:
    payload = prompt + "\x00" + "|".join(sorted(expected_brands))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{cat}__{group}__{pcat}__{h}"


def core_files(eval_dir):
    yield from sorted(glob.glob(f"{eval_dir}/*/*/benchmark.jsonl"))
    yield from sorted(glob.glob(f"{eval_dir}/*/scenario_*.jsonl"))


def group_of(fp: str) -> str:
    return fp.split("/")[-2] if fp.endswith("benchmark.jsonl") else "scenario"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="prompts/eval")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    seen = {}          # id -> running count, guarantees physical-row uniqueness
    changed = 0
    total = 0
    collisions = 0
    samples = []

    for fp in core_files(args.eval_dir):
        group = group_of(fp)
        rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        for r in rows:
            total += 1
            cat = canonical_category(r["brand_category"])
            new_id = make_id(cat, group, r["prompt_category"], r["prompt"],
                             r["expected_brands"])
            if new_id in seen:
                seen[new_id] += 1
                new_id = f"{new_id}__{seen[new_id]}"
                collisions += 1
            else:
                seen[new_id] = 1
            if new_id != r.get("id"):
                changed += 1
                if len(samples) < 8:
                    samples.append((fp, r.get("id"), new_id))
                r["id"] = new_id
        if not args.check:
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    action = "CHECK" if args.check else "WROTE"
    print(f"{action}: {total} records, {changed} ids changed, "
          f"{collisions} collisions suffixed, {len(seen)} unique ids")
    if samples:
        print("\nsample changes (file | old -> new):")
        for fp, old, new in samples:
            print(f"  {fp}\n    {old}  ->  {new}")


if __name__ == "__main__":
    main()
