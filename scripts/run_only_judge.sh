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
python3.13 -u $SCRATCH/unbranding/inference/run_only_judge.py \
    --results_dir $SCRATCH/unbranding/experiments_results/model_outputs_prompts_seed42_2026-05-25_00-08-46 \
    --judge_path $SCRATCH/unbranding/models/Qwen3-32B