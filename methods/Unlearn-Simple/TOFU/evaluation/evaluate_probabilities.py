from pathlib import Path
import csv

from utils.config import (
    load_configs,
    get_brand_name,
    get_model_family,
    get_results_config,
    get_target_epochs,
    get_experiments_root,
)

from utils.checkpoints import (
    find_target_checkpoints,
)

from utils.probability_model import (
    ProbabilityModel,
)

from utils.probability_runner import (
    run_probabilities,
)


def save_probabilities_csv(
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

    answers_root = (
        Path(
            results_cfg[
                "answers_dir"
            ]
        )
        / brand_name
        / model_family
    )

    probabilities_root = (
        Path(
            results_cfg[
                "metrics_dir"
            ]
        )
        / brand_name
        / model_family
    )


    experiments_root = (
        get_experiments_root(
            experiment_cfg
        )
    )

    experiment_dirs = sorted(
        [
            p
            for p in Path(
                experiments_root
            ).iterdir()
            if p.is_dir()
        ]
    )

    print(
        f"Found {len(experiment_dirs)} experiments"
    )

    target_epochs = (
        get_target_epochs(
            experiment_cfg
        )
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

        rows = []

        checkpoints = (
            find_target_checkpoints(
                experiment_dir,
                target_epochs,
            )
        )

        for (
            epoch,
            checkpoint_path,
        ) in checkpoints.items():

            epoch_dir = (
                answers_root
                / experiment_dir.name
                / f"epoch_{epoch}"
            )

            print()
            print(
                f"Epoch {epoch}"
            )

            prob_model = (
                ProbabilityModel(
                    model_path=str(
                        checkpoint_path
                    )
                )
            )

            probabilities = (
                run_probabilities(
                    answers_dir=epoch_dir,
                    prob_model=prob_model,
                )
            )

            row = {
                "epoch": epoch
            }

            row.update(
                probabilities
            )

            rows.append(
                row
            )

            prob_model.unload()

        output_dir = (
            probabilities_root
            / experiment_dir.name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_probabilities_csv(
            rows,
            output_dir
            / "probabilities.csv",
        )

        save_summary(
            rows,
            output_dir
            / "probabilities_summary.txt",
        )

    print()
    print(
        "=" * 80
    )

    print(
        "Probability evaluation completed"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()