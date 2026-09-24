#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/judge-%j.out
#SBATCH --error=logs/judge-%j.out
# Run on a node with an NVIDIA GH200 GPU (96 GB GPU memory).
set -euo pipefail

if [[ "$*" == *"--run"* ]]; then
  exec scripts/container.sh eval/judge.py "$@"
fi

shopt -s nullglob
RUNS=(runs/*/)

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "no runs found; run generation first" >&2
  exit 1
fi

exec scripts/container.sh eval/judge.py --run "${RUNS[@]}" "$@"
