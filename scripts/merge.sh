#!/bin/bash
#SBATCH --job-name=unlearn_gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=0:20:00
#SBATCH --output=logs3/slurm-%j.out

# Po wejściu na węzeł:
CONTAINER="/net/software/aarch64/containers/vllm/cyfronet-gh200-vllm12.sif"
export APPTAINERENV_PYTHONPATH="/net/scratch/hscra/plgrid/plgvltkv/unbranding/extra_python_libs"


source /net/scratch/hscra/plgrid/plgvltkv/unbranding/.venv3/bin/activate
apptainer exec -e --nv --bind /net:/net $CONTAINER \
python3.13 -m pip install \
    "peft" \
    -t "$TARGET_PATH"

apptainer exec -e --nv --bind /net:/net $CONTAINER \
python3.13 /net/scratch/hscra/plgrid/plgvltkv/unbranding/utils/merge_model.py