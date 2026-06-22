from pathlib import Path

from metrics.probability import (
    evaluate_probability,
)


def run_probabilities(
    answers_dir,
    prob_model,
):

    results = {}

    answers_dir = Path(
        answers_dir
    )

    for answers_file in answers_dir.glob(
        "*.jsonl"
    ):

        dataset_name = (
            answers_file.stem
        )

        if dataset_name == "forget":

            result = (
                evaluate_probability(
                    answers_file,
                    prob_model,
                )
            )

            results[
                "forget_probability"
            ] = result[
                "probability"
            ]

        elif dataset_name == "domain_retain":

            result = (
                evaluate_probability(
                    answers_file,
                    prob_model,
                )
            )

            results[
                "domain_retain_probability"
            ] = result[
                "probability"
            ]

        elif dataset_name == "world_facts":

            result = (
                evaluate_probability(
                    answers_file,
                    prob_model,
                )
            )

            results[
                "world_facts_probability"
            ] = result[
                "probability"
            ]

        elif dataset_name == "alpaca":

            result = (
                evaluate_probability(
                    answers_file,
                    prob_model,
                )
            )

            results[
                "alpaca_probability"
            ] = result[
                "probability"
            ]

    return results