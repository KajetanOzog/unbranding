from pathlib import Path
import csv

from utils.config import (
    load_configs,
    get_brand_name,
    get_model_family,
    get_metric_list,
    get_results_config,
    get_judge_model_path,
)

from utils.judge import Judge

from utils.metrics_runner import (
    run_metrics,
)

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

    metrics_list = (
        get_metric_list(
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
    results = run_metrics(
        answers_dir=answers_dir,
        metrics_list=metrics_list,
        brand_cfg=brand_cfg,
        judge=judge,
    )

    save_metrics_csv(
        results,
        metrics_dir
        / "metrics.csv",
    )

    save_summary(
        results,
        metrics_dir
        / "summary.txt",
    )

    print()
    print("=" * 80)
    print("Base evaluation completed")
    print("=" * 80)


if __name__ == "__main__":
    main()