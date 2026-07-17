from pathlib import Path

from utils.datasets import (
    load_jsonl,
    load_prompt,
)

from utils.json_parser import extract_json


PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "trade_dress.txt"
)


def evaluate_trade_dress(answers_path, brand_cfg, judge):

    dataset = load_jsonl(answers_path)

    prompt_template = load_prompt(PROMPT_PATH)

    prompts = []

    parse_errors = 0

    for sample in dataset:

        prompts.append(
            prompt_template.format(
                brand=brand_cfg["judge_brand"],
                trade_dress="\n".join(
                    brand_cfg["trade_dress"]
                ),
                answer=sample["model_answer"],
            )
        )

    outputs = judge.evaluate_batch(prompts)

    confidences = []

    present_count = 0

    for output in outputs:

        parsed = extract_json(output)
        if parsed is None:
            parse_errors += 1
            continue

        present = parsed.get("trade_dress_present", False)

        if present:
            present_count += 1

    if len(confidences) == 0:

        return {
            "trade_dress_score": 0.0,
            "mean_confidence": 0.0,
            "present_rate": 0.0,
            "num_samples": 0,
            "parse_errors": parse_errors,
        }

    valid_samples = (
    len(dataset)
    - parse_errors
    )

    present_rate = (
        present_count / valid_samples
        if valid_samples > 0
        else 0.0
    )

    return {
        "trade_dress_score": present_rate,
        "present_rate": present_rate,
        "num_samples": valid_samples,
        "parse_errors": parse_errors,
    }