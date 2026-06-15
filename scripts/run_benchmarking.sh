#!/bin/bash
#SBATCH --job-name=llm_1gpu_bench
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=00:30:00
#SBATCH --output=logs2/slurm-%j.out

TARGET_PATH="/net/scratch/hscra/plgrid/plgvltkv/unbranding/extra_python_libs"
export APPTAINERENV_PYTHONPATH="$TARGET_PATH"
export HF_HOME=$SCRATCH/hf_cache
export TOKENIZERS_PARALLELISM=false

CONTAINER="/net/software/aarch64/containers/vllm/cyfronet-gh200-vllm12.sif"

# source /net/scratch/hscra/plgrid/plgvltkv/unbranding/.venv3/bin/activate

# Czyścimy folder i instalujemy TYLKO to co psuło import, 
# dbając o limity wersji które pokazał vLLM
echo "--- Reinstalacja kompatybilnych bibliotek ---"
apptainer exec -e --nv --bind /net:/net $CONTAINER \
python3.13 -m pip install \
    "transformers>=4.44.0,<5.0.0" \
    "tokenizers" \
    "numpy<2.3" \
    -t "$TARGET_PATH"

echo "--- Start inference ---"
apptainer exec -e --nv --bind /net:/net $CONTAINER \
python3.13 -u /net/scratch/hscra/plgrid/plgvltkv/unbranding/inference/run_benchmarking.py \
    --benchmarking_dir /net/scratch/hscra/plgrid/plgvltkv/unbranding/prompts \
    --output_dir /net/scratch/hscra/plgrid/plgvltkv/unbranding/experiments_results \
    --model_paths \
        /net/scratch/hscra/plgrid/plgvltkv/unbranding/models/unlearned/simnpo_qwen-32b_merged \
    --seed 42 \
    --judge_path /net/scratch/hscra/plgrid/plgvltkv/unbranding/models/qwen-32b


            # /net/scratch/hscra/plgrid/plgvltkv/unbranding/models/unbranded_qwen-32b_merged \
                #     /net/scratch/hscra/plgrid/plgvltkv/unbranding/models/unbranded_llama-3.1-8b_merged \
    #     /net/scratch/hscra/plgrid/plgvltkv/unbranding/models/unbranded_qwen-14b_merged \
