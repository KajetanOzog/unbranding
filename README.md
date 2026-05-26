# unbranding

Ogólny zamysł jest taki, żeby:
* w folderze `models` trzymać modele,
* w folderze `prompts` dać przykładowe prompty z brandami,
* w folderze `results` trzymać odpowiedzi od modeli na te prompty.
* w folderze `experiments_results` wyniki eksperymentów

### 1. Zintegrowany Benchmark i Ewaluacja (vLLM)

Odpalenie:
```bash
sbatch scripts/run_benchmarking.sh
```

- --benchmarking_dir – ścieżka do folderu z promptami (prompty muszą być w plikach .jsonl, podzielone na podfoldery kategorii).

- --output_dir – główny folder na wyniki (skrypt sam utworzy w nim podfolder z timestampem i seedem).

- --model_paths – ścieżki do modeli, które będą generować tekst. Możesz podać kilka po spacji (skrypt załaduje je po kolei).

- --judge_path – (opcjonalne) ścieżka do modelu sędziego (np. Qwen-32b). Jeśli podana, po wygenerowaniu odpowiedzi skrypt od razu oceni je pod kątem marek i "trade dress".

- --seed – ziarno losowości (domyślnie 42).

> Aby uniknąć błędów tokenizera i problemów z importami w kontenerze, na górze pliku .sh muszą znaleźć się te dwie zmienne:

   ```bash
   export APPTAINERENV_PYTHONPATH="$(realpath ../extra_python_libs)"
   export TOKENIZERS_PARALLELISM=false
   ```

### 2. Ewaluacja wyników (LLM judge)

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

### 3. Analiza wyników

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

### 4. Używane modele
* gemma-4-31B-it
* Llama-3.1-8B-Instruct
* Mistral-Small-Instruct-2409
* Mistral-Small-Instruct-2409
* Qwen2.5-3B-Instruct
* Qwen2.5-7B-Instruct
* Qwen2.5-14B-Instruct
* Qwen3-4B
* Qwen3-7B
* Qwen3-14B
* Qwen3-32B