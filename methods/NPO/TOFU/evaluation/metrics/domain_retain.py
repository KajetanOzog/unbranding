from metrics.rouge import (
    evaluate_rouge,
)


def evaluate_domain_retain(
    answers_path,
):

    result = evaluate_rouge(
        answers_path
    )

    return {
        "domain_retain":
            result["rouge_l"],

        **result,
    }