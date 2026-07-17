from pathlib import Path

from utils.datasets import (
    load_jsonl,
    load_prompt,
)

from utils.json_parser import extract_json


PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "brand_mention.txt"
)


def evaluate_brand_mention_judge(answers_path, brand_cfg, judge):

    dataset = load_jsonl(answers_path)
    prompt_template = load_prompt(PROMPT_PATH)
    prompts = []
    parse_errors = 0

    for sample in dataset:
        prompts.append(
            prompt_template.format(
                brand=brand_cfg["judge_brand"],
                answer=sample["model_answer"],
            )
        )

    outputs = judge.evaluate_batch(prompts)
    mention_count = 0

    for output in outputs:
        parsed = extract_json(output)
        if parsed is None:
            parse_errors += 1
            continue

        mentioned = parsed.get("mentioned", False)

        if mentioned:
            mention_count += 1

    num_samples = len(dataset) - parse_errors

    mention_rate = (
        mention_count / num_samples
        if num_samples > 0
        else 0.0
    )

    return {
        "brand_mention_score": mention_rate,
        "mention_rate": mention_rate,
        "num_samples": num_samples,
        "parse_errors": parse_errors,
    }