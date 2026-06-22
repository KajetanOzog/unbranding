from pathlib import Path


def get_base_output_dir(
    answers_dir,
    brand_name,
    model_family,
):

    return (
        Path(answers_dir)
        / brand_name
        / model_family
        / "base"
    )


def get_experiment_output_dir(
    answers_dir,
    brand_name,
    model_family,
    experiment_name,
    epoch,
):

    return (
        Path(answers_dir)
        / brand_name
        / model_family
        / experiment_name
        / f"epoch_{epoch}"
    )