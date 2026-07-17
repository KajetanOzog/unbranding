from pathlib import Path
import csv

from utils.config import (
    load_configs,
    get_brand_name,
    get_model_family,
    get_results_config,
    get_base_model_path,
)

from utils.probability_model import (
    ProbabilityModel,
)

from utils.probability_runner import (
    run_probabilities,
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

    experiment_cfg, _ = (
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

    probabilities_dir = (
        Path(
            results_cfg[
                "metrics_dir"
            ]
        )
        / brand_name
        / model_family
        / "base"
    )

    probabilities_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_model_path = (
        get_base_model_path(
            experiment_cfg
        )
    )

    prob_model = (
        ProbabilityModel(
            model_path=
            base_model_path
        )
    )

    results = run_probabilities(
        answers_dir,
        prob_model,
    )

    prob_model.unload()

    save_metrics_csv(
        results,
        probabilities_dir
        / "probabilities.csv",
    )

    save_summary(
        results,
        probabilities_dir
        / "probabilities_summary.txt",
    )

    print()
    print("=" * 80)
    print(
        "Base probability evaluation completed"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()