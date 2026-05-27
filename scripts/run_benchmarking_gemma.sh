#!/bin/bash
#SBATCH --job-name=llm_1gpu_bench
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=bench_1gpu_%j.log

export APPTAINERENV_PYTHONPATH="$SCRATCH/gemma_env"
export HF_HOME=$SCRATCH/hf_cache

export TOKENIZERS_PARALLELISM=false

CONTAINER="/net/software/aarch64/containers/vllm/cyfronet-gh200-vllm12.sif"

apptainer exec -e --nv --bind /net:/net $CONTAINER \
python3.13 $SCRATCH/unbranding/inference/run_benchmarking_gemma.py \
    --benchmarking_dir $SCRATCH/unbranding/prompts \
    --output_dir $SCRATCH/unbranding/experiments_results \
    --model_path $SCRATCH/unbranding/models/gemma-4-31B-it