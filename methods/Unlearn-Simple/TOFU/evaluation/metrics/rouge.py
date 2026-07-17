from rouge_score import rouge_scorer

from utils.datasets import load_jsonl


def evaluate_rouge(answers_path):

    dataset = load_jsonl(
        answers_path
    )

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    scores = []

    for sample in dataset:

        prediction = sample[
            "model_answer"
        ]

        if "reference_answer" in sample:

            rouge_l = scorer.score(
                sample["reference_answer"],
                prediction,
            )["rougeL"].fmeasure

        elif "reference_answers" in sample:

            references = sample[
                "reference_answers"
            ]

            if len(references) == 0:
                continue

            rouge_l = max(
                scorer.score(
                    reference,
                    prediction,
                )["rougeL"].fmeasure
                for reference in references
            )

        else:
            continue

        scores.append(
            rouge_l
        )

    if len(scores) == 0:

        return {
            "rouge_l": 0.0,
            "mean_score": 0.0,
            "num_samples": 0,
        }

    mean_score = (
        sum(scores)
        / len(scores)
    )

    return {
        "rouge_l": mean_score,
        "mean_score": mean_score,
        "num_samples": len(scores),
    }