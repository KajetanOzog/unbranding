#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen-%j.out
#SBATCH --error=logs/gen-%j.out
# Run on a node with an NVIDIA GH200 GPU (96 GB GPU memory).
set -euo pipefail

exec scripts/container.sh eval/generate.py "$@"
