#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from common import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    Model,
    extract_json,
    load_config,
    read_jsonl,
    write_jsonl,
)
from evaluator_config import EvaluationConfig


def load_prompts(config_path, evaluation):
    root = Path(config_path).resolve().parent
    prompts = {}
    for name, evaluator in evaluation.evaluators.items():
        system_path = root / evaluator.system_prompt
        user_path = root / evaluator.user_prompt
        try:
            prompts[name] = (
                system_path.read_text(encoding="utf-8").strip(),
                user_path.read_text(encoding="utf-8").strip(),
            )
        except OSError as error:
            raise SystemExit(f"cannot load prompts for {name}: {error}") from error
    return prompts


def output_matches(value, output_type):
    if output_type == "boolean":
        return isinstance(value, bool)
    if output_type == "string":
        return isinstance(value, str)
    if output_type == "string_list":
        return isinstance(value, list) and all(
            isinstance(item, str) for item in value
        )
    if output_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if output_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def normalize_output(raw, evaluator):
    parsed = extract_json(raw)
    if parsed is None:
        return None
    value = parsed.get(evaluator.output.field)
    if not output_matches(value, evaluator.output.type):
        return None
    return value


def choice_judgment(record):
    response = record.get("response", "").lower()
    selected = [
        choice for choice in record["choices"] if choice.lower() in response
    ]
    return {
        "selected_choices": selected,
        "choice_correct": record["answer"].lower() in response,
    }


def record_context(record, config):
    brand_slug = record.get("brand")
    brand = config.canonical(brand_slug) if brand_slug else ""
    brand_config = config.brand_cfg(brand_slug) if brand_slug else {}
    return {
        "answer": record.get("response", ""),
        "prompt": record.get("prompt", ""),
        "brand": brand,
        "aliases": "\n".join(
            f"- {alias}" for alias in brand_config.get("aliases", [])
        ) or "(none)",
        "trade_dress": "\n".join(
            f"- {item}" for item in brand_config.get("trade_dress", [])
        ) or "(none)",
        "reference": "\n".join(
            f"- {item}" for item in record.get("reference", [])
        ) or "(none)",
        "choices": "\n".join(
            f"- {item}" for item in record.get("choices", [])
        ) or "(none)",
        "label": str(record.get("label") or ""),
    }


def build_jobs(records, config, evaluation, prompts):
    jobs = defaultdict(list)
    judgments = [choice_judgment(record) if record["task"] == "choices" else {}
                 for record in records]

    for index, record in enumerate(records):
        context = record_context(record, config)
        for evaluator_name in evaluation.tasks[record["task"]]:
            _, user_template = prompts[evaluator_name]
            jobs[evaluator_name].append(
                (index, user_template.format(**context))
            )
    return jobs, judgments


def evaluate_jobs(jobs, judgments, model, evaluation, prompts):
    invalid = 0
    for evaluator_name, batch in jobs.items():
        evaluator = evaluation.evaluators[evaluator_name]
        system_prompt, _ = prompts[evaluator_name]
        outputs = model.generate(
            [prompt for _, prompt in batch],
            system=system_prompt,
        )
        for (record_index, _), output in zip(batch, outputs):
            value = normalize_output(output, evaluator)
            judgments[record_index][evaluator_name] = value
            invalid += value is None
    return invalid


def judge_run(run_dir, output_dir, model, config, evaluation, prompts, resume):
    total = 0
    for input_path in sorted(Path(run_dir).glob("shard_*.jsonl")):
        output_path = Path(output_dir) / input_path.name
        if resume and output_path.exists():
            total += len(read_jsonl(output_path))
            continue

        records = read_jsonl(input_path)
        jobs, judgments = build_jobs(records, config, evaluation, prompts)
        invalid = evaluate_jobs(
            jobs,
            judgments,
            model,
            evaluation,
            prompts,
        )
        for record, judgment in zip(records, judgments):
            record["judgment"] = judgment
            record["judge_model"] = model.model_name

        write_jsonl(output_path, records)
        total += len(records)
        calls = sum(len(batch) for batch in jobs.values())
        print(
            f"[judge] {input_path.name}: {len(records)} records, "
            f"{calls} calls, {invalid} invalid outputs"
        )
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, nargs="+")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-tokenizer")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(REPO_ROOT / "judged"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        evaluation = EvaluationConfig.model_validate(
            config.judge["evaluation"]
        )
    except ValidationError as error:
        raise SystemExit(f"invalid judge configuration:\n{error}") from error
    prompts = load_prompts(args.config, evaluation)

    model_config = config.model_entry(config.judge["model"])
    model_path = args.judge_model or model_config["path"]
    model = Model(
        model_path,
        runtime=config.judge["runtime"],
        model_config=model_config,
        tokenizer_path=args.judge_tokenizer or model_config.get("tokenizer"),
    )

    for run_dir in map(Path, args.run):
        if not run_dir.is_dir():
            raise SystemExit(f"not a run directory: {run_dir}")
        output_dir = Path(args.out) / run_dir.name
        total = judge_run(
            run_dir,
            output_dir,
            model,
            config,
            evaluation,
            prompts,
            resume=not args.no_resume,
        )
        print(f"[judge] {run_dir.name}: {total} records")


if __name__ == "__main__":
    main()
