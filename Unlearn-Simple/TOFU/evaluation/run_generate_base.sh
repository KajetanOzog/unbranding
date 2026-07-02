#!/bin/bash
#SBATCH --job-name=base_eval
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --output=base_eval_%j.log

export APPTAINERENV_PYTHONPATH="$SCRATCH/unbranding/evaluation"
export HF_HOME=$SCRATCH/hf_cache

export TOKENIZERS_PARALLELISM=false

CONTAINER="/net/software/aarch64/containers/vllm/cyfronet-gh200-vllm12.sif"

export APPTAINERENV_CUDA_HOME=/net/software/aarch64/el9/CUDA/12.9.1

export APPTAINERENV_VLLM_CACHE_ROOT=$SCRATCH/vllm_cache
mkdir -p $SCRATCH/vllm_cache

export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=$SCRATCH/torchinductor_cache
mkdir -p $SCRATCH/torchinductor_cache

apptainer exec \
    -e \
    --nv \
    --bind /net:/net \
    $CONTAINER \
    python3.13 -u \
    $SCRATCH/unbranding/Unlearn-Simple/TOFU/evaluation/generate_base_answers.py
