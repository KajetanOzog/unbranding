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

# 1. Ustawienie środowiska CUDA pod maszyny ARM
export CUDA_HOME=/net/software/aarch64/el8/CUDA/12.4.0  
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 2. Konfiguracja bezlimitowych ścieżek scratch
export SCRATCH_DIR="/net/scratch/hscra/plgrid/plgvltkv/unbranding"
export PIP_CACHE_DIR="${SCRATCH_DIR}/.pip_cache"
export TMPDIR="${SCRATCH_DIR}/.tmp"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.triton"

mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

# 3. Przejście do folderu projektu
cd ${SCRATCH_DIR}/Unlearn-Simple-2/TOFU

# 4. Rekonstrukcja środowiska wirtualnego
rm -rf ${SCRATCH_DIR}/.venv
python3.11 -m venv ${SCRATCH_DIR}/.venv
source ${SCRATCH_DIR}/.venv/bin/activate

# 5. Instalacja podstawowych narzędzi i stabilnego PyTocha pod ARM (ZMIANA Z >=2.6.0 NA ==2.5.1)
pip install --upgrade pip
pip install wheel setuptools packaging ninja
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 6. Instalacja pozostałych pakietów z requirements
pip install -r requirements.txt

# 7. Instalacja idealnie zgranych wersji core (bez wymuszania pydantic<2.0)
pip install "transformers==4.46.3" "peft==0.12.0" "deepspeed==0.15.4"

# 8. AUTOMATYCCZNY PATCH KONFIGURACJI (Blokada błędu ułamka w Pydantic v2)
# Podmieniamy tekst "auto" na inta w plikach konfiguracyjnych deepspeed
python3 -c "
import json, glob
for f in glob.glob('**/*.json', recursive=True):
    try:
        with open(f, 'r') as file: data = json.load(file)
        if 'zero_optimization' in data and 'stage3_prefetch_bucket_size' in data['zero_optimization']:
            data['zero_optimization']['stage3_prefetch_bucket_size'] = 15099494
            with open(f, 'w') as file: json.dump(data, file, indent=4)
            print(f'=== ZAŁATANO BUFOR W: {f} ===')
    except Exception: pass
"

# 9. Losowanie portu dla komunikacji między kartami GH200
export master_port=$(shuf -i 20000-65000 -n 1)

# 10. Uruchomienie treningu
torchrun --nproc_per_node=4 --master_port=$master_port forget.py \
    --config-name=forget.yaml \
    split=forget05 \
    npo_coeff=0.1375 \
    beta=2.5 \
    save_dir="/net/scratch/hscra/plgrid/plgvltkv/tutaj_wpisz_nowa_sciezke"