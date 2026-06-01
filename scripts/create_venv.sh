#!/bin/bash
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00  # Zwiększyłem czas, bo modele 14B ładują się chwilę
#SBATCH --output=logs/slurm-%j.out

module purge
module load  GCCcore/13.2.0 CUDA/12.8.0

cd $SCRATCH/unbranding

rm -rf .venv3
python -m venv .venv3
source .venv3/bin/activate

which python
which pip

pip install --upgrade pip setuptools wheel
pip install torch transformers accelerate sentencepiece
pip install vllm huggingface_hub
pip install bitsandbytes transformers accelerate datasets peft sentencepiece