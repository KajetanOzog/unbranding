from pathlib import Path
import csv

from utils.config import (
    load_configs,
    get_brand_name,
    get_model_family,
    get_results_config,
    get_judge_model_path,
)

from utils.judge import Judge

from metrics.trade_dress import (
    evaluate_trade_dress,
)

from metrics.brand_mention_judge import (
    evaluate_brand_mention_judge,
)


FILES = [
    "thesis.jsonl",
    "short_thesis.jsonl",
    "choices.jsonl",
    "old_forget.jsonl"
]


def save_metrics_csv(
    results,
    output_path,
):

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["metric", "value"]
        )

        for metric, value in (
            sorted(results.items())
        ):

            writer.writerow(
                [metric, value]
            )


def save_summary(
    results,
    output_path,
):

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for metric, value in (
            sorted(results.items())
        ):

            f.write(
                f"{metric}: {value}\n"
            )


def main():

    experiment_cfg, brand_cfg = (
        load_configs()
    )

    brand_name = get_brand_name(
        experiment_cfg
    )

    model_family = (
        get_model_family(
            experiment_cfg
        )
    )

    results_cfg = (
        get_results_config(
            experiment_cfg
        )
    )

    answers_dir = (
        Path(
            results_cfg[
                "answers_dir"
            ]
        )
        / brand_name
        / model_family
        / "base"
    )

    metrics_dir = (
        Path(
            results_cfg[
                "metrics_dir"
            ]
        )
        / brand_name
        / model_family
        / "base"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    judge = Judge(
        model_path=
        get_judge_model_path(
            experiment_cfg
        )
    )

    results = {}

    for file_name in FILES:

        answers_file = (
            answers_dir
            / file_name
        )

        if not answers_file.exists():
            continue

        dataset_name = (
            answers_file.stem
        )

        trade_result = (
            evaluate_trade_dress(
                answers_file,
                brand_cfg,
                judge,
            )
        )

        mention_result = (
            evaluate_brand_mention_judge(
                answers_file,
                brand_cfg,
                judge,
            )
        )

        results[
            f"{dataset_name}_trade_dress"
        ] = trade_result[
            "trade_dress_score"
        ]

        results[
            f"{dataset_name}_brand_mention_judge"
        ] = mention_result[
            "brand_mention_score"
        ]

    save_metrics_csv(
        results,
        metrics_dir
        / "extra_metrics.csv",
    )

    save_summary(
        results,
        metrics_dir
        / "extra_summary.txt",
    )

    print()
    print("=" * 80)
    print("Base evaluation completed")
    print("=" * 80)


if __name__ == "__main__":
    main()