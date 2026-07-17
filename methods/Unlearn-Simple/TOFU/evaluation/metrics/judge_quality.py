from pathlib import Path

from utils.datasets import (
    load_jsonl,
    load_prompt,
)

from utils.json_parser import (
    extract_json,
)


PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "judge_quality.txt"
)


def evaluate_judge_quality(
    answers_path,
    judge,
):

    dataset = load_jsonl(
        answers_path
    )

    prompt_template = load_prompt(
        PROMPT_PATH
    )

    prompts = []

    parse_errors = 0

    for sample in dataset:

        prompts.append(
            prompt_template.format(
                question=sample["question"],
                answer=sample["model_answer"],
            )
        )

    outputs = judge.evaluate_batch(
        prompts
    )

    scores = []

    for output in outputs:

        parsed = extract_json(
            output
        )

        if parsed is None:
            parse_errors += 1
            continue

        scores.append(
            parsed.get(
                "score",
                1,
            )
        )

    if len(scores) == 0:

        return {
            "judge_quality": 0.0,
            "mean_score": 0.0,
            "num_samples": 0,
            "parse_errors": parse_errors,
        }

    mean_score = (
        sum(scores)
        / len(scores)
    )

    return {
        "judge_quality":
            mean_score / 5.0,

        "mean_score":
            mean_score,

        "num_samples":
            len(scores),

        "parse_errors":
            parse_errors,
    }