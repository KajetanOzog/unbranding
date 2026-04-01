#!/bin/bash
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

module load GCCcore/11.3.0 Python/3.10.4 CUDA/11.7.0
source $SCRATCH/unbranding/.venv/bin/activate

python $SCRATCH/unbranding/inference/prompts2.py \
    --model_path "meta-llama/Meta-Llama-3.1-8B" \
    --prompts_dir "/net/tscratch/people/plgvltkv/unbranding/prompts" \
    --output_dir "/net/tscratch/people/plgvltkv/unbranding/results3" \