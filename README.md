# Unbranding LLM

Projekt badawczy o **"unbrandingu" (odmarkowieniu) dużych modeli językowych** —
sprawianiu, by modele **przestały odwoływać się do konkretnych marek**
(np. Coca-Cola, BMW, Nike, Apple) zarówno wprost (nazwa marki), jak i pośrednio
przez **trade dress** (charakterystyczne cechy marki bez nazwy: logo, kolory,
slogany, założyciele, kształty).

Marki pogrupowane są w 5 kategorii — **auto, beverages, food, sport, tech**
(po 4 marki na kategorię, 20 łącznie).

Pipeline składa się z trzech etapów:

1. **Oduczanie (machine unlearning)** — trening usuwający wiedzę o markach
   metodami **NPO** i **SimNPO** (oparte na frameworku TOFU) — katalog `methods/`.
2. **Ewaluacja** — trzy-etapowy harness `eval/` (generacja → sędzia → metryki),
   napędzany wyłącznie przez `prompts/eval/**` i `config.yaml`.
3. **Analiza** — agregacja `scores.csv` (brand leakage, trade dress, retencja).

---

## Układ repozytorium

```
unbranding_llm/
├── config.yaml          # JEDNO źródło prawdy: marki, aliasy, trade dress, model-sędzia
├── eval/                # Harness ewaluacyjny (3 etapy)
│   ├── common.py        #   wspólne helpery: config, ładowanie rekordów, Model (vLLM lazy)
│   ├── generate.py      #   Etap 1: model bazowy wypełnia `response`
│   ├── judge.py         #   Etap 2: LLM-as-a-judge dokleja `judgment`
│   ├── metrics.py       #   Etap 3: agregacja judged → scores.csv (CPU-only)
│   └── prompts/         #   szablony sędziego (brand_mention, trade_dress, stance...)
├── prompts/
│   ├── eval/            # Zbiór ewaluacyjny — samoopisujące się rekordy JSONL (patrz niżej)
│   │   └── <kategoria>/<marka>/{benchmark,thesis,choices,forget}.jsonl
│   │       + <kategoria>/{scenario,retain}.jsonl + world_facts.jsonl
│   └── train/           # Zbiory treningowe: forget/ (do zapomnienia) + retain/ (do zachowania)
├── methods/             # Metody oduczania (framework TOFU)
│   ├── NPO/             #   Negative Preference Optimization
│   └── Unlearn-Simple/  #   SimNPO
├── tools/               # assign_ids.py (deterministyczne id), skrypty run_benchmarking*
├── inference/, utils/   # Starsze skrypty generacji/analizy (stopniowo migrowane do eval/)
└── scripts/             # Skrypty SLURM (sbatch)
```

Katalogi wyjściowe (`runs/`, `judged/`, `scores.csv`) powstają w trakcie
działania i nie są śledzone w git.

---

## Kontrakt danych ewaluacyjnych

Cały `prompts/eval/**` to pliki **JSONL** o **jednej wspólnej kopercie**.
Rekord jest samoopisujący się — nie zależy od nazwy pliku ani folderu.

```json
{ "id": "...", "brand_category": "auto", "brand": "audi",
  "task": "benchmark", "prompt": "...", "response": "", ...payload }
```

| pole | znaczenie |
|---|---|
| `id` | `<brand_category>__<brand>__<task>__<hash8>`, deterministyczny z treści (`tools/assign_ids.py`). Cross-brand → `brand=all`; world_facts → `brand_category=world` |
| `brand_category` | `auto`\|`beverages`\|`food`\|`sport`\|`tech`, lub `null` (world_facts) |
| `brand` | slug marki (`audi`, `coca_cola`, `red_bull`…), lub `null` dla plików cross-brand |
| `task` | typ zadania — po nim dispatchuje runner i sędzia |
| `prompt` | tekst podawany modelowi |
| `response` | miejsce na odpowiedź modelu (na wejściu puste) |

### Taski, payload i metryki

| task | plik | payload | metryka |
|---|---|---|---|
| `benchmark` | `<cat>/<brand>/benchmark.jsonl` | `prompt_category`, `expected_brands` | brand leakage (jawny + trade dress) |
| `scenario` | `<cat>/scenario.jsonl` | `prompt_category`, `expected_brands: []` | leakage: jakakolwiek marka |
| `choices` | `<cat>/<brand>/choices.jsonl` | `choices: []`, `answer` | trafność multiple-choice (deterministyczna) |
| `thesis` | `<cat>/<brand>/thesis.jsonl` | `label` | zgodność opinii/sentymentu |
| `forget` | `<cat>/<brand>/forget.jsonl` | — | czy model wypowiada markę |
| `retain` | `<cat>/retain.jsonl` | `reference: []` | utrzymanie wiedzy kategorii |
| `world_facts` | `world_facts.jsonl` | `reference: []` | ogólna wiedza (TOFU) |

**Zasady:** jedna marka na rekord `benchmark` (`expected_brands` = `[marka folderu]`);
`id` reprodukowalny (`tools/assign_ids.py`, `--check` do podglądu); etapy tylko
**dopisują** pola (`prompt → response → judgment → scores.csv`).

---

## Pipeline ewaluacyjny (`eval/`)

Wszystko napędzają dwa źródła prawdy: `prompts/eval/**` (rekordy) i `config.yaml`
(baza wiedzy o markach + parametry sędziego). Żadnych ścieżek/stałych w kodzie.

### Etap 1 — generacja (vLLM)

Model bazowy wypełnia puste `response` w każdym rekordzie.

```bash
python eval/generate.py --model <ścieżka-lub-HF-id> [--name ETYKIETA]
# → runs/<name>/shard_<shard_id>.jsonl
```

`--name` domyślnie = nazwa katalogu modelu (staje się wierszem w tabeli metryk).
`--num-shards` / `--shard-id` wspierają joby tablicowe SLURM (rekord `i` trafia do
sharda `i % num_shards`).

### Etap 2 — sędzia (LLM-as-a-judge)

Dla każdego rekordu dokleja obiekt `judgment`, dispatchując po `task`:

```bash
python eval/judge.py --run runs/<name> --judge-model <ścieżka> [--config config.yaml]
# → judged/<name>/shard_<rank>.jsonl
```

| task | judgment | sposób |
|---|---|---|
| `benchmark` | `brand_present`, `trade_dress_present` | LLM |
| `forget` | `brand_present` | LLM |
| `scenario` | `brands_mentioned`, `any_brand` | LLM |
| `thesis` | `stance`, `label`, `stance_match` | LLM |
| `retain` / `world_facts` | `correct` | LLM |
| `choices` | `selected`, `correct` | deterministycznie (bez LLM) |

Model-sędzia i jego parametry pochodzą z sekcji `judge:` w `config.yaml`
(domyślnie `Qwen/Qwen2.5-32B-Instruct`).

### Etap 3 — metryki (bez GPU)

Czyta wszystkie judged runs, deduplikuje po `id` i redukuje `judgment` do
współczynników per task — łącznie i per `brand_category`.

```bash
python eval/metrics.py [--judged judged] [--out scores.csv]
```

Zapisuje long-form CSV (`name, task, metric, category, value, n`) i drukuje
zwięzłą tabelę zbiorczą. Nie wymaga zależności GPU/modelu.

---

## Oduczanie (`methods/`)

Metody **NPO** i **SimNPO** oparte na frameworku **TOFU** (fine-tune → forget →
evaluate), konfigurowane przez Hydra + DeepSpeed, opcjonalnie z LoRA. Dane
treningowe pochodzą z `prompts/train/` (forget set = prompty z markami do
zapomnienia; retain set = dane bez marek chroniące ogólne zdolności modelu).

```bash
python methods/NPO/TOFU/forget.py            # oduczanie NPO
python methods/Unlearn-Simple/TOFU/forget.py # oduczanie SimNPO
```

---

## Kluczowe pojęcia

| Pojęcie | Znaczenie |
|---|---|
| **Forget set** | Dane z markami, których model ma się "oduczyć". |
| **Retain set** | Dane bez marek — chronią ogólne zdolności modelu. |
| **Trade dress** | Pośrednie odniesienia do marki (logo, kolory, slogany, założyciele) bez nazwy. |
| **Brand leakage** | Jak często mimo oduczania w odpowiedziach pojawiają się marki. |
| **NPO / SimNPO** | Metody unlearningu oparte na frameworku TOFU. |
| **LLM-as-a-Judge** | Duży model (Qwen-32B) oceniający obecność marek w odpowiedziach. |
