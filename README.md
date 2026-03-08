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

### 4. Ewaluacja wyników (LLM judge)

Po wygenerowaniu odpowiedzi modeli można uruchomić skrypty ewaluacyjne z folderu `inference`.

Te skrypty używają dodatkowego modelu (LLM judge), który analizuje wygenerowane odpowiedzi i wykrywa odniesienia do brandów.

Przykładowo:

```bash
sbatch scripts/run_any_brand_eval.sh
```

lub

```bash
sbatch scripts/run_concrete_brand_eval.sh
```

Skrypty:

* wczytują odpowiedzi modeli z folderu `results`
* analizują je pod kątem:

  * występowania nazw brandów
  * występowania **trade dress** (charakterystycznych cech brandu bez podania nazwy)
* zapisują wyniki do folderu `llmaj_results`

Struktura wyników wygląda wtedy tak:

```
llmaj_results/
   llama/
      any_brand_check/
      concrete_brands/

   qwen/
      any_brand_check/
      concrete_brands/
```

Każdy plik wynikowy jest w formacie `jsonl` i zawiera:

* prompt
* odpowiedź modelu
* wynik analizy brandów

---

### 5. Analiza wyników

Po wygenerowaniu wyników ewaluacji można uruchomić skrypty analityczne z folderu `utils`.

Przykładowo:

```bash
python utils/analyze_llmaj_results.py --date 2026-03-04
```

Skrypt:

* wczytuje wyniki z folderu `llmaj_results`
* agreguje statystyki dla każdego modelu i kategorii promptów
* generuje raport tekstowy
* zapisuje dodatkowe pliki z analizą

Wygenerowana struktura:

```
llmaj_results/
   analysis_YYYY-MM-DD/

      report.txt
      any_brand_summary.csv
      brand_leakage_ranking.csv

      heatmap_brand_rate.png
      heatmap_trade_rate.png
```

Raport (`report.txt`) zawiera m.in.:

* statystyki **brand leakage** dla każdej kategorii promptów
* statystyki **trade dress detection**
* ranking najczęściej pojawiających się brandów
* porównanie modeli