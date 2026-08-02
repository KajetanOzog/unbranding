#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from common import REPO_ROOT, load_shards


TASK_METRICS = {
    "benchmark": [
        ("target_brand_mention_rate", "target_brand_present", "boolean"),
        ("target_trade_dress_rate", "target_trade_dress_present", "boolean"),
        ("any_brand_mention_rate", "brands_mentioned", "nonempty_list"),
        ("any_trade_dress_rate", "trade_dress_brands", "nonempty_list"),
    ],
    "forget": [
        ("target_brand_mention_rate", "target_brand_present", "boolean"),
        ("target_trade_dress_rate", "target_trade_dress_present", "boolean"),
        ("any_brand_mention_rate", "brands_mentioned", "nonempty_list"),
        ("any_trade_dress_rate", "trade_dress_brands", "nonempty_list"),
    ],
    "scenario": [
        ("any_brand_mention_rate", "brands_mentioned", "nonempty_list"),
        ("any_trade_dress_rate", "trade_dress_brands", "nonempty_list"),
    ],
    "choices": [
        ("accuracy", "choice_correct", "boolean"),
        ("target_brand_mention_rate", "target_brand_present", "boolean"),
        ("target_trade_dress_rate", "target_trade_dress_present", "boolean"),
        ("any_brand_mention_rate", "brands_mentioned", "nonempty_list"),
        ("any_trade_dress_rate", "trade_dress_brands", "nonempty_list"),
    ],
    "thesis": [
        ("stance_match_rate", "stance", "label_match"),
        ("target_brand_mention_rate", "target_brand_present", "boolean"),
        ("target_trade_dress_rate", "target_trade_dress_present", "boolean"),
        ("any_brand_mention_rate", "brands_mentioned", "nonempty_list"),
        ("any_trade_dress_rate", "trade_dress_brands", "nonempty_list"),
    ],
    "retain": [("correct_rate", "qa_correct", "boolean")],
    "world_facts": [("correct_rate", "qa_correct", "boolean")],
}


def reduce_value(value, reduction, record):
    if reduction == "boolean" and isinstance(value, bool):
        return float(value)
    if reduction == "nonempty_list" and isinstance(value, list):
        return float(bool(value))
    if reduction == "label_match" and isinstance(value, str):
        return float(value.lower() == str(record.get("label", "")).lower())
    if reduction == "number" and isinstance(value, (int, float)):
        return float(value)
    return None


def score_run(name, records):
    values = defaultdict(list)
    invalid = defaultdict(int)

    for record in records:
        task = record.get("task")
        judgment = record.get("judgment") or {}
        metrics = list(TASK_METRICS.get(task, []))
        if "quality_1_5" in judgment:
            metrics.append(("quality_1_5", "quality_1_5", "number"))

        for metric, key, reduction in metrics:
            if key not in judgment:
                continue
            result = reduce_value(judgment[key], reduction, record)
            categories = ("__all__", record.get("brand_category") or "world")
            for category in categories:
                bucket = (task, metric, category)
                if result is None:
                    invalid[bucket] += 1
                else:
                    values[bucket].append(result)

    for task, metric, category in sorted(set(values) | set(invalid)):
        bucket = (task, metric, category)
        valid_values = values[bucket]
        value = (
            round(sum(valid_values) / len(valid_values), 4)
            if valid_values
            else None
        )
        yield (
            name,
            task,
            metric,
            category,
            value,
            len(valid_values),
            invalid[bucket],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", default=str(REPO_ROOT / "judged"))
    parser.add_argument("--out", default=str(REPO_ROOT / "scores.csv"))
    args = parser.parse_args()

    judged_dir = Path(args.judged)
    run_dirs = sorted(path for path in judged_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise SystemExit(f"no runs found under {judged_dir}")

    rows = []
    for run_dir in run_dirs:
        records = load_shards(run_dir)
        rows.extend(score_run(run_dir.name, records))
        print(f"[metrics] {run_dir.name}: {len(records)} records")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["name", "task", "metric", "category", "value", "n_valid", "n_invalid"]
        )
        writer.writerows(rows)

    print(f"[metrics] wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
