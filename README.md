# Unbranding LLM

A research project on **"unbranding" large language models** — making models
**stop referring to specific brands** (e.g. Coca-Cola, BMW, Nike, Apple) both
directly (the brand name) and indirectly via **trade dress** (distinctive brand
signals without the name: logos, colors, slogans, founders, shapes).

Brands are grouped into 5 categories — **auto, beverages, food, sport, tech**
(4 brands per category, 20 in total).

The pipeline has three stages:

1. **Unlearning (machine unlearning)** — training that removes brand knowledge
   using **NPO** and **SimNPO** (built on the TOFU framework) — see `methods/`.
2. **Evaluation** — the three-stage `eval/` harness (generate → judge → metrics),
   driven solely by `prompts/eval/**` and `config.yaml`.
3. **Analysis** — aggregation into `scores.csv` (brand leakage, trade dress, retention).

---

## Repository layout

```
unbranding_llm/
├── config.yaml          # single source of truth: brands, aliases, trade dress, judge, model quirks
├── eval/                # evaluation harness (3 stages)
│   ├── common.py        #   shared helpers: config, record loading, Model (lazy vLLM)
│   ├── generate.py      #   Stage 1: base model fills `response`
│   ├── judge.py         #   Stage 2: LLM-as-a-judge appends `judgment`
│   ├── metrics.py       #   Stage 3: aggregate judged runs → scores.csv (CPU-only)
│   └── prompts/         #   judge templates (brand_mention, trade_dress, stance...)
├── prompts/
│   ├── eval/            # evaluation set — self-describing JSONL records (see below)
│   │   └── <category>/<brand>/{benchmark,thesis,choices,forget}.jsonl
│   │       + <category>/{scenario,retain}.jsonl + world_facts.jsonl
│   └── train/           # training sets: forget/ (to unlearn) + retain/ (to keep)
├── methods/             # unlearning methods (TOFU framework)
│   ├── NPO/             #   Negative Preference Optimization
│   └── Unlearn-Simple/  #   SimNPO
└── tools/               # assign_ids.py (deterministic record ids)
```

Output directories (`runs/`, `judged/`, `scores.csv`) are produced at run time
and are not tracked in git.

---

## Evaluation data contract

Everything under `prompts/eval/**` is **JSONL** with a **single shared envelope**.
Each record is self-describing — it does not depend on its file or folder name.

```json
{ "id": "...", "brand_category": "auto", "brand": "audi",
  "task": "benchmark", "prompt": "...", "response": "", ...payload }
```

| field | meaning |
|---|---|
| `id` | `<brand_category>__<brand>__<task>__<hash8>`, deterministic from content (`tools/assign_ids.py`). Cross-brand → `brand=all`; world_facts → `brand_category=world` |
| `brand_category` | `auto`\|`beverages`\|`food`\|`sport`\|`tech`, or `null` (world_facts) |
| `brand` | brand slug (`audi`, `coca_cola`, `red_bull`…), or `null` for cross-brand files |
| `task` | task type — the runner and judge dispatch on it |
| `prompt` | text fed to the model |
| `response` | slot for the model's answer (empty on input) |

### Tasks, payload and metrics

| task | file | payload | metric |
|---|---|---|---|
| `benchmark` | `<cat>/<brand>/benchmark.jsonl` | `prompt_category`, `expected_brands` | brand leakage (explicit + trade dress) |
| `scenario` | `<cat>/scenario.jsonl` | `prompt_category`, `expected_brands: []` | leakage: any brand |
| `choices` | `<cat>/<brand>/choices.jsonl` | `choices: []`, `answer` | multiple-choice accuracy (deterministic) |
| `thesis` | `<cat>/<brand>/thesis.jsonl` | `label` | opinion/sentiment agreement |
| `forget` | `<cat>/<brand>/forget.jsonl` | — | whether the model utters the brand |
| `retain` | `<cat>/retain.jsonl` | `reference: []` | category-knowledge retention |
| `world_facts` | `world_facts.jsonl` | `reference: []` | general knowledge (TOFU) |

**Rules:** one brand per `benchmark` record (`expected_brands` = `[folder brand]`);
`id` is reproducible (`tools/assign_ids.py`, `--check` to preview); stages only
**append** fields (`prompt → response → judgment → scores.csv`).

---

## Evaluation pipeline (`eval/`)

Everything is driven by two sources of truth: `prompts/eval/**` (records) and
`config.yaml` (brand knowledge base + judge parameters). No paths or constants
are hardcoded in the code.

### Stage 1 — generation (vLLM)

The base model fills the empty `response` in every record.

```bash
python eval/generate.py --model <path-or-hf-id> [--name LABEL]
# → runs/<name>/shard_<shard_id>.jsonl
```

`--name` defaults to the model directory name (it becomes the row label in the
metrics table). `--num-shards` / `--shard-id` support SLURM array jobs (record
`i` goes to shard `i % num_shards`).

### Stage 2 — judge (LLM-as-a-judge)

For each record it appends a `judgment` object, dispatching on `task`:

```bash
python eval/judge.py --run runs/<name> --judge-model <path> [--config config.yaml]
# → judged/<name>/shard_<rank>.jsonl
```

| task | judgment | how |
|---|---|---|
| `benchmark` | `brand_present`, `trade_dress_present` | LLM |
| `forget` | `brand_present` | LLM |
| `scenario` | `brands_mentioned`, `any_brand` | LLM |
| `thesis` | `stance`, `label`, `stance_match` | LLM |
| `retain` / `world_facts` | `correct` | LLM |
| `choices` | `selected`, `correct` | deterministic (no LLM) |

The judge model and its parameters come from the `judge:` section of
`config.yaml` (default `Qwen/Qwen2.5-32B-Instruct`).

### Stage 3 — metrics (no GPU)

Reads all judged runs, dedupes by `id`, and reduces `judgment` to per-task rates
— overall and per `brand_category`.

```bash
python eval/metrics.py [--judged judged] [--out scores.csv]
```

Writes a long-form CSV (`name, task, metric, category, value, n`) and prints a
compact summary table. No GPU/model dependencies.

### Per-model quirks

Family-specific generation quirks (extra stop strings, chat-template kwargs such
as Qwen3's `enable_thinking`) live in the `model_overrides:` section of
`config.yaml`, matched by substring against the model directory name. Adding a
new family is a config edit, not a code change — the code never sniffs model
names.

---

## Unlearning (`methods/`)

The **NPO** and **SimNPO** methods are built on the **TOFU** framework
(fine-tune → forget → evaluate), configured via Hydra + DeepSpeed, optionally
with LoRA. Training data comes from `prompts/train/` (forget set = brand prompts
to unlearn; retain set = brand-free data protecting the model's general
abilities).

```bash
python methods/NPO/TOFU/forget.py            # NPO unlearning
python methods/Unlearn-Simple/TOFU/forget.py # SimNPO unlearning
```

---

## Key concepts

| Concept | Meaning |
|---|---|
| **Forget set** | Data with the brands the model should "unlearn". |
| **Retain set** | Brand-free data — protects the model's general abilities. |
| **Trade dress** | Indirect brand references (logos, colors, slogans, founders) without the name. |
| **Brand leakage** | How often brands still appear in responses despite unlearning. |
| **NPO / SimNPO** | Unlearning methods built on the TOFU framework. |
| **LLM-as-a-Judge** | A large model (Qwen-32B) scoring brand presence in responses. |
