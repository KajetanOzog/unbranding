from pathlib import Path

from pathlib import Path


def find_target_checkpoints(
    experiment_dir,
    target_epochs,
):

    experiment_dir = Path(
        experiment_dir
    )

    checkpoint_dirs = sorted(
        experiment_dir.glob(
            "checkpoint-*"
        ),
        key=lambda p: int(
            p.name.split("-")[1]
        ),
    )

    if len(checkpoint_dirs) == 0:
        raise ValueError(
            f"No checkpoints found in {experiment_dir}"
        )

    selected = {}

    for epoch in target_epochs:

        if epoch > len(checkpoint_dirs):
            raise ValueError(
                f"Epoch {epoch} not available. "
                f"Found only {len(checkpoint_dirs)} checkpoints."
            )

        selected[epoch] = checkpoint_dirs[
            epoch - 1
        ]

    return selected