# unbranding

Ogólny zamysł jest taki, żeby:
* w folderze `models` trzymać modele,
* w folderze `prompts` dać przykładowe prompty z brandami,
* w folderze `results` trzymać odpowiedzi od modeli na te prompty.

### 1. Środowisko (venv)
```bash
sbatch scripts/create_venv.sh
```

### 2. Testy (Single Prompt)
To wtedy najlepiej sobie odpalic:
```bash
sbatch scripts/llama_inference.sh
```
lub
```bash
sbatch scripts/qwen_inference.sh
```

> Tylko wtedy w `inference/llama_inference.py` albo w `inference/qwen_inference.py` musisz ustawic sciezke do modelu i podac jeden prompt przykladowy.

### 3. Odpalenie dla listy promptów
```bash
sbatch scripts/run_prompts.sh
```

> Tylko wtedy tez w `inference/run_prompts.py` musisz dac sciezki do modelu, promptow i miejsca gdzie zapisze wyniki. W formacie jsonl sa prompty i wyniki.