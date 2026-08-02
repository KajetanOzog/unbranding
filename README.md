# Unbranding LLM

```text
dataset/eval/*.jsonl
  → generate.py → runs/<model>/shard_*.jsonl
  → judge.py    → judged/<model>/shard_*.jsonl
  → metrics.py  → scores.csv
```

Wszystkie polecenia wykonuj z katalogu głównego repozytorium.

## 1. Przygotuj kontener

Sprawdź ścieżki w `container.env`, a następnie pobierz obraz:

```bash
containers/pull.sh
```

Obraz zostanie zapisany pod `UNBRANDING_IMAGE`. Jeżeli już istnieje, skrypt go
nie pobierze ponownie.

## 2. Skonfiguruj modele

Modele i parametry runtime znajdują się w `config.yaml`:

```yaml
models:
  qwen3-8b-base:
    path: Qwen/Qwen3-8B
    chat_template:
      enable_thinking: false

judge:
  model: qwen3-8b-base
```

- `qwen3-8b-base` jest nazwą używaną w poleceniach i nazwą katalogu wynikowego.
- `path` może być identyfikatorem Hugging Face albo ścieżką do checkpointu.
- Dla checkpointu bez tokenizera dodaj `tokenizer: Qwen/Qwen3-8B`.
- `runtime` steruje generowaniem, a `judge.runtime` niezależnie steruje judge'em.

Wszystkie parametry runtime są jawne w `config.yaml`; kod nie dodaje własnych
wartości domyślnych.

## 3. Wygeneruj odpowiedzi

```bash
scripts/container.sh eval/generate.py --model qwen3-8b-base
```

Polecenie odczyta `dataset/eval/`, wygeneruje `response` dla każdego rekordu i
zapisze:

```text
runs/qwen3-8b-base/shard_0.jsonl
```

Przykładowy rekord po tym kroku:

```json
{
  "id": "auto__audi__forget__...",
  "task": "forget",
  "prompt": "...",
  "response": "...",
  "model": "qwen3-8b-base"
}
```

Ponowne uruchomienie zachowa gotowe odpowiedzi. Pełne przeliczenie:

```bash
scripts/container.sh eval/generate.py \
  --model qwen3-8b-base \
  --no-resume
```

### Generowanie równoległe

Poniższe polecenia tworzą niezależne pliki i mogą działać równolegle:

```bash
scripts/container.sh eval/generate.py \
  --model qwen3-8b-base --num-shards 2 --shard-id 0

scripts/container.sh eval/generate.py \
  --model qwen3-8b-base --num-shards 2 --shard-id 1
```

Output:

```text
runs/qwen3-8b-base/shard_0.jsonl
runs/qwen3-8b-base/shard_1.jsonl
```

## 4. Wybierz oceny judge'a

Evaluatory są zdefiniowane w `judge.evaluation.evaluators` w `config.yaml`.
Każdy wskazuje system prompt, user prompt i oczekiwane pole JSON:

```yaml
target_brand_present:
  system_prompt: eval/prompts/system/target_brand.txt
  user_prompt: eval/prompts/user/target_brand_present.txt
  output: {field: mentioned, type: boolean}
```

Lista evaluatorów wykonywanych dla danego taska znajduje się w `tasks`:

```yaml
tasks:
  forget:
    - target_brand_present
    - target_trade_dress_present
    - brands_mentioned
    - trade_dress_brands
```

Aby włączyć dodatkową ocenę, dopisz jej nazwę do taska, na przykład:

```yaml
retain: [qa_correct, quality_1_5]
```

Treść promptów można zmieniać niezależnie w:

```text
eval/prompts/system/
eval/prompts/user/
```

## 5. Uruchom judge'a

```bash
scripts/container.sh eval/judge.py --run runs/qwen3-8b-base
```

Judge odczyta wszystkie shardy runu, wykona evaluatory przypisane do każdego
taska i zapisze:

```text
judged/qwen3-8b-base/shard_*.jsonl
```

Przykładowy wynik:

```json
{
  "judgment": {
    "target_brand_present": true,
    "target_trade_dress_present": false,
    "brands_mentioned": ["Audi"],
    "trade_dress_brands": []
  },
  "judge_model": "Qwen3-8B"
}
```

Niepoprawny JSON albo niewłaściwy typ daje `null`. Dla taska `choices` pola
`selected_choices` i `choice_correct` są wyliczane bez modelu judge'a.

Istniejące ocenione shardy są pomijane. Pełne przeliczenie:

```bash
scripts/container.sh eval/judge.py \
  --run runs/qwen3-8b-base \
  --no-resume
```

Kilka runów można ocenić po jednym załadowaniu judge'a:

```bash
scripts/container.sh eval/judge.py \
  --run runs/model-a runs/model-b
```

## 6. Policz metryki

```bash
scripts/container.sh eval/metrics.py \
  --judged judged \
  --out scores.csv
```

Każdy podkatalog `judged/` jest traktowany jako osobny run. Wynikiem jest:

```csv
name,task,metric,category,value,n_valid,n_invalid
qwen3-8b-base,forget,target_brand_mention_rate,__all__,0.125,350,2
qwen3-8b-base,forget,target_brand_mention_rate,auto,0.1,70,0
```

- `value` — średnia z poprawnych ocen,
- `n_valid` — liczba ocen użytych w średniej,
- `n_invalid` — liczba odrzuconych ocen,
- `__all__` — wynik łączny; pozostałe wiersze są per kategoria.

Ten krok nie uruchamia modelu i nie wymaga GPU.

## SLURM

Te same dwa etapy GPU można wysłać na klaster:

```bash
mkdir -p logs
sbatch scripts/generate.sbatch --model qwen3-8b-base
sbatch scripts/judge.sbatch --run runs/qwen3-8b-base
```

Logi trafią do `logs/gen-<job_id>.out` i `logs/judge-<job_id>.out`. Pliki
wynikowe pozostają odpowiednio w `runs/` i `judged/`.

## Pełne uruchomienie

```bash
containers/pull.sh
scripts/container.sh eval/generate.py --model qwen3-8b-base
scripts/container.sh eval/judge.py --run runs/qwen3-8b-base
scripts/container.sh eval/metrics.py --judged judged --out scores.csv
```

Końcowy wynik znajduje się w `scores.csv`.
