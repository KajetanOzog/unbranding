#!/bin/bash
#SBATCH --job-name=unbrand-gen
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen-%j.out
#SBATCH --error=logs/gen-%j.out
set -euo pipefail

exec scripts/container.sh eval/generate.py "$@"
