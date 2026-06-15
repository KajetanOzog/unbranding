#!/bin/bash
#SBATCH --job-name=unlearn_gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1                                
#SBATCH --ntasks-per-node=4                      
#SBATCH --gres=gpu:4                             
#SBATCH --cpus-per-task=32                       
#SBATCH --mem=240G
#SBATCH --time=2:00:00
#SBATCH --output=logs2/slurm-%j.out

# 1. Czyszczenie środowiska modułów i sztywne wskazanie CUDA pod ARM
export CUDA_HOME=/net/software/aarch64/el8/CUDA/12.4.0  
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "--- ŚCIEŻKA DO NVCC ---"
which nvcc
echo "--- WERSJA NVCC ---"
nvcc --version
echo "-----------------------"

# 2. Konfiguracja ścieżek tymczasowych (OMINIĘCIE LIMITU QUOTA)
export SCRATCH_DIR="/net/scratch/hscra/plgrid/plgvltkv/unbranding"
export PIP_CACHE_DIR="${SCRATCH_DIR}/.pip_cache"
export TMPDIR="${SCRATCH_DIR}/.tmp"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.triton"

mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

# 3. Przejście do folderu projektu (Upewnij się, czy folder ma na końcu "-2" czy nie)
cd ${SCRATCH_DIR}/Unlearn-Simple-2/TOFU

source ${SCRATCH_DIR}/.venv/bin/activate

# 8. Losowanie portu dla komunikacji między kartami
export master_port=$(shuf -i 20000-65000 -n 1)

# 9. Uruchomienie treningu (wywołanie bezpośrednie z obecnego katalogu)
torchrun --nproc_per_node=4 --master_port=$master_port forget.py \
    --config-name=forget.yaml \
    # split=forget05 \
    # npo_coeff=0.1375 \
    # beta=2.5