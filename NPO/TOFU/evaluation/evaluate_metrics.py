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
    rows,
    output_path,
):

    if len(rows) == 0:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


def save_summary(
    rows,
    output_path,
):

    if len(rows) == 0:
        return

    columns = list(
        rows[0].keys()
    )

    widths = {}

    for column in columns:

        widths[column] = max(
            len(column),
            *[
                len(
                    str(
                        row[column]
                    )
                )
                for row in rows
            ],
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        header = " | ".join(
            column.ljust(
                widths[column]
            )
            for column in columns
        )

        separator = "-+-".join(
            "-" * widths[column]
            for column in columns
        )

        f.write(
            header + "\n"
        )

        f.write(
            separator + "\n"
        )

        for row in rows:

            line = " | ".join(
                str(
                    row[column]
                ).ljust(
                    widths[column]
                )
                for column in columns
            )

            f.write(
                line + "\n"
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

    answers_root = (
        Path(
            results_cfg[
                "answers_dir"
            ]
        )
        / brand_name
        / model_family
    )

    metrics_root = (
        Path(
            results_cfg[
                "metrics_dir"
            ]
        )
        / brand_name
        / model_family
    )

    judge = Judge(
        model_path=
        get_judge_model_path(
            experiment_cfg
        )
    )

    experiment_dirs = sorted(
        [
            p
            for p in answers_root.iterdir()
            if (
                p.is_dir()
                and p.name != "base"
            )
        ]
    )

    print(
        f"Found {len(experiment_dirs)} experiments"
    )

    for experiment_dir in (
        experiment_dirs
    ):

        print()
        print(
            "=" * 80
        )

        print(
            experiment_dir.name
        )

        print(
            "=" * 80
        )

        epoch_dirs = sorted(
            [
                p
                for p in experiment_dir.iterdir()
                if (
                    p.is_dir()
                    and p.name.startswith(
                        "epoch_"
                    )
                )
            ],
            key=lambda p: int(
                p.name.split(
                    "_"
                )[1]
            ),
        )

        rows = []

        for epoch_dir in (
            epoch_dirs
        ):

            epoch = int(
                epoch_dir.name.split(
                    "_"
                )[1]
            )

            print()
            print(
                f"Epoch {epoch}"
            )

            metrics = run_metrics(
                answers_dir=epoch_dir,
                metrics_list=metrics_list,
                brand_cfg=brand_cfg,
                judge=judge,
            )

            row = {
                "epoch": epoch
            }

            row.update(
                metrics
            )

            rows.append(
                row
            )

        output_dir = (
            metrics_root
            / experiment_dir.name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_metrics_csv(
            rows,
            output_dir
            / "metrics.csv",
        )

        save_summary(
            rows,
            output_dir
            / "summary.txt",
        )

    print()
    print(
        "=" * 80
    )
    print(
        "Evaluation completed"
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()