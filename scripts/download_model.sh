#!/bin/bash
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00 
#SBATCH --output=logs/slurm-%j.out

module purge
module load GCCcore/13.2.0 Python/3.11.5

cd $SCRATCH/unbranding

source .venv/bin/activate

pip install huggingface_hub

export HF_HOME=$SCRATCH/hf_cache
export HF_TOKEN=...

# huggingface-cli download Qwen/Qwen2.5-32B\
#   --local-dir models/qwen-32b

# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct\
#   --local-dir models/llama-3.1-8b

# huggingface-cli download Qwen/Qwen2.5-14B-Instruct\
#   --local-dir models/qwen-14b

huggingface-cli download mistralai/Mistral-Small-Instruct-2409\
  --local-dir models/mistral-small

# huggingface-cli download google/gemma-4-31B-it\
#   --local-dir models/gemma-4-31b-it

  