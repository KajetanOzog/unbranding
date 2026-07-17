#!/bin/bash
#SBATCH --job-name=llm_1gpu_bench
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=grid_%A_%a.out

export APPTAINERENV_PYTHONPATH="$SCRATCH/unbranding/extra_python_libs"
export HF_HOME=$SCRATCH/hf_cache

export TOKENIZERS_PARALLELISM=false

CONTAINER="/net/software/aarch64/containers/vllm/cyfronet-gh200-vllm12.sif"
export APPTAINERENV_CUDA_HOME=/net/software/aarch64/el9/CUDA/12.9.1

LRS=("1e-4" "3e-4" "5e-4")

LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "Running LR=$LR"

apptainer exec -e --nv --bind /net:/net $CONTAINER python3.13 -u $SCRATCH/unbranding/Unlearn-Simple/TOFU/forget.py lr=$LR