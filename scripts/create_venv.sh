#!/bin/bash
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00  # Zwiększyłem czas, bo modele 14B ładują się chwilę
#SBATCH --output=logs/slurm-%j.out

module load GCCcore/11.3.0 Python/3.10.4 CUDA/11.7.0

cd $SCRATCH/unbranding

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install --no-cache-dir torch transformers accelerate sentencepiece