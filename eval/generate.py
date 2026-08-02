#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    DEFAULT_EVAL_DIR,
    REPO_ROOT,
    Model,
    build_input,
    load_config,
    load_eval_records,
    read_jsonl,
    shard,
    write_jsonl,
)


def completed_records(path):
    if not path.exists():
        return {}
    return {
        record["id"]: record
        for record in read_jsonl(path)
        if record.get("response")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config.model_entry(args.model)
    records = load_eval_records(args.eval_dir)
    records = shard(records, args.num_shards, args.shard_id)
    if args.limit:
        records = records[:args.limit]

    output = Path(args.out) / args.model / f"shard_{args.shard_id}.jsonl"
    completed = {} if args.no_resume else completed_records(output)
    pending = [record for record in records if record["id"] not in completed]

    print(
        f"[generate] {args.model}: {len(pending)}/{len(records)} records "
        f"in shard {args.shard_id}/{args.num_shards}"
    )
    if not pending:
        return

    model = Model(
        model_config["path"],
        runtime=config.runtime,
        model_config=model_config,
        tokenizer_path=model_config.get("tokenizer"),
    )
    responses = model.generate([build_input(record) for record in pending])
    for record, response in zip(pending, responses):
        record["response"] = response
        record["model"] = args.model

    write_jsonl(
        output,
        [completed.get(record["id"], record) for record in records],
    )
    print(f"[generate] wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
