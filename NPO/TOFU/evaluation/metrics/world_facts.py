import re

from utils.datasets import load_jsonl


def normalize(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def evaluate_world_facts(answers_path):

    dataset = load_jsonl(
        answers_path
    )

    correct_count = 0

    for sample in dataset:

        model_answer = normalize(
            sample["model_answer"]
        )

        reference_answers = [
            normalize(answer)
            for answer in sample[
                "reference_answers"
            ]
        ]

        correct = any(
            reference_answer
            in model_answer
            for reference_answer
            in reference_answers
        )

        if correct:
            correct_count += 1

    num_samples = len(dataset)

    if num_samples == 0:

        return {
            "world_facts_accuracy": 0.0,
            "correct": 0,
            "num_samples": 0,
        }

    return {
        "world_facts_accuracy":
            correct_count / num_samples,

        "correct":
            correct_count,

        "num_samples":
            num_samples,
    }