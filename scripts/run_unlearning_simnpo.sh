#!/bin/bash
#SBATCH --job-name=unlearn_gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

export SCRATCH="/net/scratch/hscra/plgrid/plgvltkv"
export EXTRA_LIBS="$SCRATCH/unbranding/extra_python_libs"
export HF_HOME="$SCRATCH/hf_cache"
export TOKENIZERS_PARALLELISM=false

CONTAINER="/net/software/aarch64/containers/vllm/cyfronet-gh200-vllm12.sif"

# 1. Naprawa Gemmy
echo '--- Deep Gemma Model Fix (Architecture & Tokenizer) ---'
apptainer exec -e --nv --bind /net:/net $CONTAINER python3.13 -c "
import json, os
base_path = '$SCRATCH/unbranding/models/gemma-4-31b-it'

# Fix config.json (zmieniamy gemma4 -> gemma2)
config_path = os.path.join(base_path, 'config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f: data = json.load(f)
    if data.get('model_type') == 'gemma4':
        data['model_type'] = 'gemma2'
        with open(config_path, 'w') as f: json.dump(data, f, indent=2)
        print('SUCCESS: config.json updated (gemma4 -> gemma2).')

# Fix tokenizer_config.json (czyszczenie śmieci)
tok_path = os.path.join(base_path, 'tokenizer_config.json')
if os.path.exists(tok_path):
    with open(tok_path, 'r') as f: data = json.load(f)
    for k in ['additional_special_tokens', 'extra_special_tokens', 'special_tokens_map']:
        if k in data: del data[k]
    with open(tok_path, 'w') as f: json.dump(data, f, indent=2)
    print('SUCCESS: tokenizer_config.json cleaned.')
"

# 2. Czyszczenie i instalacja
rm -rf $EXTRA_LIBS
mkdir -p $EXTRA_LIBS

# ... (początek skryptu bez zmian) ...

apptainer exec -e --nv --bind /net:/net $CONTAINER bash -c "
    export PYTHONPATH=\$PYTHONPATH:$EXTRA_LIBS
    
    echo '--- Instalacja brakujących narzędzi (Complete Deps) ---'
    # Dodajemy xxhash, packaging i zestaw do obsługi zapytań HTTP
    python3.13 -m pip install --target=$EXTRA_LIBS --no-cache-dir --no-deps \
        'transformers==4.48.0' 'tokenizers==0.22.2' 'peft>=0.14.0' \
        'accelerate>=1.2.0' 'datasets>=3.1.0' 'bitsandbytes>=0.45.0' \
        'sentencepiece' 'huggingface-hub' 'safetensors' 'tqdm' 'regex' \
        'pyyaml' 'fsspec' 'dill' 'multiprocess' 'aiohttp' 'aiosignal' \
        'frozenlist' 'multidict' 'yarl' 'propcache' 'attrs' \
        'pandas' 'pyarrow' 'python-dateutil' 'six' 'numpy' \
        'xxhash' 'packaging' 'requests' 'urllib3' 'idna' 'charset-normalizer'

    echo '--- DEZAKTYWACJA SPRAWDZANIA WERSJI ---'
    python3.13 -c \"
import os
path = '$EXTRA_LIBS/transformers/dependency_versions_check.py'
if os.path.exists(path):
    with open(path, 'r') as f:
        content = f.read()
    new_content = content.replace('require_version_core(deps[pkg])', 'pass')
    with open(path, 'w') as f:
        f.write(new_content)
    print('SUCCESS: Transformers version check disabled.')
\"

    echo '--- Rozpoczynam unlearning ---'
    python3.13 -u /net/scratch/hscra/plgrid/plgvltkv/unbranding/inference/unlearning_simnpo.py \
        --bad_data_path '$SCRATCH/unbranding/unlearning_dataset/bad_dataset.jsonl' \
        --good_data_path '$SCRATCH/unbranding/unlearning_dataset/good_dataset.jsonl' \
        --output_base_dir '$SCRATCH/unbranding/models' \
        --model_paths \
            '$SCRATCH/unbranding/models/gemma-4-31b-it' \
            '$SCRATCH/unbranding/models/llama-3.1-8b' \
            '$SCRATCH/unbranding/models/mistral-small' \
            '$SCRATCH/unbranding/models/qwen-14b' \
            '$SCRATCH/unbranding/models/qwen-32b'
"