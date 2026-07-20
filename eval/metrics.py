#!/usr/bin/env python3
"""Stage 3 - aggregate judged runs into scores.

Reads every judged run under --judged (each subdir is one run/<name>), dedupes
records by id, and reduces the per-record `judgment` into rates per task, both
overall and per brand_category.

    python eval/metrics.py [--judged judged] [--out scores.csv]

Writes a long-form CSV (name, task, metric, category, value, n) and prints a
compact overall table. No GPU / model dependencies.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from common import REPO_ROOT, load_shards

# task -> list of (metric label, judgment key). Every key is a 0/1 rate.
TASK_METRICS = {
    "benchmark": [("brand_mention_rate", "brand_present"),
                  ("trade_dress_rate", "trade_dress_present")],
    "forget": [("forget_mention_rate", "brand_present")],
    "scenario": [("scenario_any_brand_rate", "any_brand")],
    "choices": [("choices_accuracy", "correct")],
    "thesis": [("thesis_stance_match_rate", "stance_match")],
    "retain": [("retain_correct_rate", "correct")],
    "world_facts": [("world_facts_correct_rate", "correct")],
}


def category_of(record):
    return record.get("brand_category") or "world"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def score_run(name, records):
    """Yield rows (name, task, metric, category, value, n) for one run."""
    # bucket[(task, metric, category)] -> list of 0/1
    buckets = defaultdict(list)
    for r in records:
        task = r.get("task")
        j = r.get("judgment") or {}
        for metric, key in TASK_METRICS.get(task, []):
            if key not in j:
                continue
            val = 1.0 if j[key] else 0.0
            buckets[(task, metric, "__all__")].append(val)
            buckets[(task, metric, category_of(r))].append(val)

    for (task, metric, cat), vals in sorted(buckets.items()):
        yield (name, task, metric, cat, round(mean(vals), 4), len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=str(REPO_ROOT / "judged"))
    ap.add_argument("--out", default=str(REPO_ROOT / "scores.csv"))
    args = ap.parse_args()

    run_dirs = sorted(d for d in Path(args.judged).iterdir() if d.is_dir())
    if not run_dirs:
        raise SystemExit(f"no runs found under {args.judged}")

    rows = []
    for run_dir in run_dirs:
        records = load_shards(run_dir)
        rows.extend(score_run(run_dir.name, records))
        print(f"[metrics] {run_dir.name}: {len(records)} records")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "task", "metric", "category", "value", "n"])
        w.writerows(rows)
    print(f"[metrics] wrote {len(rows)} rows -> {args.out}")

    # compact overall table (category == __all__)
    print("\n=== overall ===")
    print(f"{'run':<28}{'metric':<28}{'value':>8}{'n':>8}")
    for name, task, metric, cat, value, n in rows:
        if cat == "__all__":
            print(f"{name:<28}{metric:<28}{value:>8.3f}{n:>8}")


if __name__ == "__main__":
    main()
