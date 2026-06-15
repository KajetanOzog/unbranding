#!/bin/bash
#SBATCH --job-name=llm_1gpu_bench
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs5/slurm-%j.out

set -e

echo "=== Python info ==="
which python || true
python --version || true

# Create dedicated clean environment once
VENV_DIR="$SCRATCH/unbranding/.venv-spacy"

if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating virtualenv ==="

    python3.9 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"

    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip setuptools wheel

    pip install "spacy<3.8"

    python -m spacy download en_core_web_sm
else
    source "$VENV_DIR/bin/activate"
fi

echo "=== Active python ==="
which python
python --version

echo "=== Installed spaCy ==="
python -c "import spacy; print(spacy.__version__)"

echo "=== Running evaluation ==="

python $SCRATCH/unbranding/experiments_results/a/b.py \
  --input_file $SCRATCH/unbranding/experiments_results/a/a.jsonl \
  --output_file $SCRATCH/unbranding/experiments_results/a/c.jsonl

# python $SCRATCH/unbranding/inference/eval_any_brand_ner.py \
#     --results_dir $SCRATCH/unbranding/experiments_results/model_outputs_prompts_seed42_2026-05-26_17-45-49