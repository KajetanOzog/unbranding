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
python3.13 -u $SCRATCH/unbranding/inference/run_benchmarking.py \
    --benchmarking_dir $SCRATCH/unbranding/prompts_by_brand2 \
    --output_dir $SCRATCH/unbranding/experiments_results_NPO/attempt2 \
    --model_paths \
        $SCRATCH/unbranding/models/unlearned-npo/qwen3-8b-NPO \
        $SCRATCH/unbranding/models/qwen-32b \
    --seed 42 \
    --judge_path $SCRATCH/unbranding/models/qwen-32b