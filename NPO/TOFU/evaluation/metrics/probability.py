from utils.datasets import (
    load_jsonl,
)


def evaluate_probability(
    answers_path,
    prob_model,
):

    dataset = load_jsonl(
        answers_path
    )

    scores = []

    for sample in dataset:

        question = sample[
            "question"
        ]

        if (
            "reference_answer"
            in sample
        ):

            score = (
                prob_model.evaluate(
                    question,
                    sample[
                        "reference_answer"
                    ],
                )
            )

        elif (
            "reference_answers"
            in sample
        ):

            score = max(
                prob_model.evaluate(
                    question,
                    answer,
                )
                for answer in sample[
                    "reference_answers"
                ]
            )

        else:
            continue

        scores.append(
            score
        )

    if len(scores) == 0:

        return {
            "probability": 0.0,
            "num_samples": 0,
        }

    mean_score = (
        sum(scores)
        / len(scores)
    )

    return {
        "probability":
            mean_score,

        "num_samples":
            len(scores),
    }