#!/bin/bash
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out

module load GCCcore/11.3.0 Python/3.10.4 CUDA/11.7.0
source $SCRATCH/unbranding/.venv/bin/activate

pip install spacy
python -m spacy download en_core_web_sm

python $SCRATCH/unbranding/inference/eval_any_brand_ner.py \
    --results_dir $SCRATCH/unbranding/experiments_results/model_outputs_prompts_seed42_2026-04-16_22-38-58
