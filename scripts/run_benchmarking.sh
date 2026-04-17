#!/bin/bash
#SBATCH --job-name=llm_1gpu_bench
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=bench_1gpu_%j.log

export APPTAINERENV_PYTHONPATH="/net/scratch/hscra/plgrid/plgkajetan/unbranding/extra_python_libs"
export HF_HOME=$SCRATCH/hf_cache

export TOKENIZERS_PARALLELISM=false

CONTAINER="/net/software/aarch64/containers/vllm/cyfronet-gh200-vllm12.sif"

apptainer exec -e --nv --bind /net:/net $CONTAINER \
python3.13 -u /net/scratch/hscra/plgrid/plgkajetan/unbranding/inference/run_benchmarking.py \
    --benchmarking_dir /net/scratch/hscra/plgrid/plgkajetan/unbranding/prompts \
    --output_dir /net/scratch/hscra/plgrid/plgkajetan/unbranding/experiments_results \
    --model_paths \
        /net/scratch/hscra/plgrid/plgkajetan/unbranding/models/llama-3.1-8b \
        /net/scratch/hscra/plgrid/plgkajetan/unbranding/models/qwen-14b \
    --seed 42 \
    --judge_path /net/scratch/hscra/plgrid/plgkajetan/unbranding/models/qwen-32b