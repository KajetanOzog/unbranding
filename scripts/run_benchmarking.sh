#!/bin/bash
#SBATCH --job-name=llm_1gpu_bench
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=bench_1gpu_%j.log

export APPTAINERENV_PYTHONPATH="$SCRATCH/unbranding/extra_python_libs"
export HF_HOME=$SCRATCH/hf_cache

export TOKENIZERS_PARALLELISM=false

CONTAINER="/net/software/aarch64/containers/vllm/cyfronet-gh200-vllm12.sif"

apptainer exec -e --nv --bind /net:/net $CONTAINER \
python3.13 -u $SCRATCH/unbranding/inference/run_benchmarking_new.py \
    --benchmarking_dir $SCRATCH/unbranding/prompts \
    --output_dir $SCRATCH/unbranding/experiments_results \
    --model_paths \
        $SCRATCH/unbranding/models/Llama-3.1-8B-Instruct \
        $SCRATCH/unbranding/models/Qwen2.5-3B-Instruct \
        $SCRATCH/unbranding/models/Qwen2.5-7B-Instruct \
        $SCRATCH/unbranding/models/Qwen2.5-14B-Instruct \
        $SCRATCH/unbranding/models/Qwen3-4B \
        $SCRATCH/unbranding/models/Qwen3-8B \
        $SCRATCH/unbranding/models/Qwen3-14B \
        $SCRATCH/unbranding/models/Qwen3-32B \
        $SCRATCH/unbranding/models/Mistral-Small-Instruct-2409 \
        $SCRATCH/unbranding/models/Mistral-7B-Instruct-v0.3 \
    --seed 42 \
    --judge_path $SCRATCH/unbranding/models/Qwen3-32B