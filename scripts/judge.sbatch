#!/bin/bash
#SBATCH --job-name=unbrand-judge
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/judge-%j.out
#SBATCH --error=logs/judge-%j.out
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
