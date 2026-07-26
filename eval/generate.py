#!/usr/bin/env python3
"""Stage 1 - generate base model responses over the evaluation set.

Reads the self-describing records from prompts/eval/**, keeps the full envelope
(see README.md), and fills the empty `response` field with the model's output.

    python eval/generate.py --model <path-or-hf-id> [--name LABEL]

Output:
    runs/<name>/shard_<shard_id>.jsonl     (envelope + response + model label)

<name> defaults to the model directory name; it becomes the row label in
metrics.py. --num-shards / --shard-id support SLURM array jobs (record i goes
to shard i % num_shards); the default is a single shard covering everything.
"""
import argparse
import random
from pathlib import Path

from common import (
    DEFAULT_CONFIG, DEFAULT_EVAL_DIR, REPO_ROOT, Model,
    load_config, load_eval_records, shard, write_jsonl,
)


def build_input(record) -> str:
    """The user-facing prompt for one record (task-aware only where needed)."""
    prompt = record["prompt"]
    if record["task"] == "choices":
        # Present the options so multiple-choice accuracy is meaningful.
        opts = "\n".join(f"- {c}" for c in record["choices"])
        return f"{prompt}\nChoose exactly one of the following options:\n{opts}"
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path or HF id of the base model")
    ap.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--name", default=None, help="run label (default: model dir name)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #records (0=all)")
    args = ap.parse_args()

    random.seed(args.seed)
    name = args.name or Path(args.model).name
    out_path = Path(args.out) / name / f"shard_{args.shard_id}.jsonl"

    records = load_eval_records(args.eval_dir)
    records = shard(records, args.num_shards, args.shard_id)
    if args.limit:
        records = records[: args.limit]
    print(f"[generate] {name}: {len(records)} records "
          f"(shard {args.shard_id}/{args.num_shards}) -> {out_path}")

    overrides = load_config(args.config).raw.get("model_overrides", [])
    model = Model(
        args.model, max_tokens=args.max_tokens,
        temperature=args.temperature, seed=args.seed, overrides=overrides,
    )
    responses = model.generate([build_input(r) for r in records])

    for r, resp in zip(records, responses):
        r["response"] = resp
        r["model"] = name

    write_jsonl(out_path, records)
    print(f"[generate] wrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
