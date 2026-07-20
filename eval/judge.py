#!/usr/bin/env python3
"""Stage 2 - LLM-as-a-judge over generated responses.

Reads a run produced by generate.py, dispatches every record by its `task`
against the ground truth in config.yaml, and appends a `judgment` object.

    python eval/judge.py --run runs/<name> --judge-model <path> [--config config.yaml]

Per-task judgment:
    benchmark    -> {brand_present, trade_dress_present}   (LLM)
    forget       -> {brand_present}                        (LLM)
    scenario     -> {brands_mentioned, any_brand}          (LLM)
    thesis       -> {stance, label, stance_match}          (LLM)
    retain       -> {correct}                              (LLM)
    world_facts  -> {correct}                              (LLM)
    choices      -> {selected, correct}                    (deterministic)

Output:
    <out>/<name>/shard_<rank>.jsonl     (rank = shard index of the input run)
"""
import argparse
from pathlib import Path

from common import (
    DEFAULT_CONFIG, REPO_ROOT, Model,
    extract_json, load_config, load_prompt, read_jsonl, write_jsonl,
)

# judgment key -> (json field emitted by the judge, default value)
KEY_FIELD = {
    "brand_present": ("mentioned", False),
    "trade_dress_present": ("trade_dress_present", False),
    "brands": ("brands", []),
    "stance": ("stance", "neutral"),
    "correct": ("correct", False),
}


def judge_choices(record) -> dict:
    """Deterministic multiple-choice grading (no LLM)."""
    resp = record.get("response", "").lower()
    selected = [c for c in record["choices"] if c.lower() in resp]
    return {"selected": selected, "correct": record["answer"].lower() in resp}


def build_jobs(records, cfg, templates):
    """Collect (record_idx, key, prompt) LLM jobs; fill deterministic parts."""
    jobs = []
    judgments = [dict() for _ in records]

    for i, r in enumerate(records):
        task = r["task"]
        answer = r.get("response", "")

        if task == "choices":
            judgments[i] = judge_choices(r)

        elif task in ("benchmark", "forget"):
            brand = r["brand"]
            names = " / ".join(cfg.names_for(brand))
            jobs.append((i, "brand_present", templates["brand_mention"].format(
                brand=names, answer=answer)))
            if task == "benchmark":
                td = cfg.trade_dress_for(brand)
                jobs.append((i, "trade_dress_present", templates["trade_dress"].format(
                    brand=cfg.canonical(brand),
                    trade_dress="\n".join(td) if td else "(none)",
                    answer=answer)))

        elif task == "scenario":
            jobs.append((i, "brands", templates["any_brand"].format(answer=answer)))

        elif task == "thesis":
            jobs.append((i, "stance", templates["stance"].format(answer=answer)))
            judgments[i]["label"] = r.get("label")

        elif task in ("retain", "world_facts"):
            ref = "\n".join(f"- {x}" for x in r.get("reference", []))
            jobs.append((i, "correct", templates["qa_correct"].format(
                question=r["prompt"], reference=ref, answer=answer)))

    return jobs, judgments


def apply_outputs(jobs, outputs, judgments):
    parse_errors = 0
    for (i, key, _), out in zip(jobs, outputs):
        parsed = extract_json(out)
        field, default = KEY_FIELD[key]
        if parsed is None:
            parse_errors += 1
            value = default
        else:
            value = parsed.get(field, default)
        if key == "brands":
            judgments[i]["brands_mentioned"] = value
            judgments[i]["any_brand"] = bool(value)
        else:
            judgments[i][key] = value
    return parse_errors


def finalize(records, judgments):
    for r, j in zip(records, judgments):
        if r["task"] == "thesis":
            j["stance_match"] = (j.get("stance") == j.get("label"))
        r["judgment"] = j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir from generate.py (runs/<name>)")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(REPO_ROOT / "judged"))
    ap.add_argument("--name", default=None, help="output label (default: run dir name)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    max_tokens = int(cfg.judge.get("max_new_tokens", 128))
    name = args.name or Path(args.run).name
    templates = {n: load_prompt(f"{n}.txt") for n in
                 ("brand_mention", "trade_dress", "any_brand", "stance", "qa_correct")}

    model = Model(args.judge_model, max_tokens=max_tokens,
                  temperature=float(cfg.judge.get("temperature", 0.0)), seed=args.seed)

    for shard_fp in sorted(Path(args.run).glob("shard_*.jsonl")):
        records = read_jsonl(shard_fp)
        jobs, judgments = build_jobs(records, cfg, templates)
        outputs = model.generate([p for _, _, p in jobs]) if jobs else []
        parse_errors = apply_outputs(jobs, outputs, judgments)
        finalize(records, judgments)
        for r in records:
            r["judge_model"] = Path(args.judge_model).name

        out_fp = Path(args.out) / name / shard_fp.name
        write_jsonl(out_fp, records)
        print(f"[judge] {shard_fp.name}: {len(records)} records, "
              f"{len(jobs)} judge calls, {parse_errors} parse errors -> {out_fp}")


if __name__ == "__main__":
    main()
