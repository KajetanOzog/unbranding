"""Shared helpers for the evaluation pipeline (generate -> judge -> metrics).

Everything is driven by two sources of truth:
  * prompts/eval/**  - self-describing records (see README.md, "Kontrakt danych")
  * config.yaml      - brand knowledge base + judge defaults

The vLLM/torch imports are lazy (inside `Model`) so that metrics.py and the
deterministic parts of judge.py run on a plain CPU box with no GPU stack.
"""
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
DEFAULT_EVAL_DIR = REPO_ROOT / "prompts" / "eval"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

TASKS = (
    "benchmark", "scenario", "choices",
    "thesis", "forget", "retain", "world_facts",
)


# --------------------------------------------------------------------------- #
#  config.yaml
# --------------------------------------------------------------------------- #
def slugify(name: str) -> str:
    """Canonical brand name -> record slug ('Coca-Cola' -> 'coca_cola')."""
    s = name.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


class Config:
    """Thin wrapper over config.yaml with slug <-> canonical lookups."""

    def __init__(self, data: dict):
        self.raw = data
        self.judge = data.get("judge", {})
        self.categories = data.get("categories", [])
        self.category_aliases = data.get("category_aliases", {})
        self.brands = data.get("brands", {})
        # slug -> canonical brand name
        self._by_slug = {slugify(name): name for name in self.brands}

    def canonical(self, brand_slug: str) -> str:
        if brand_slug in self._by_slug:
            return self._by_slug[brand_slug]
        raise KeyError(f"unknown brand slug: {brand_slug!r}")

    def brand_cfg(self, brand_slug: str) -> dict:
        return self.brands[self.canonical(brand_slug)]

    def names_for(self, brand_slug: str) -> list:
        """Canonical name + all aliases (leakage surface for one brand)."""
        name = self.canonical(brand_slug)
        return [name] + list(self.brands[name].get("aliases", []))

    def trade_dress_for(self, brand_slug: str) -> list:
        return list(self.brand_cfg(brand_slug).get("trade_dress", []))

    def all_brand_names(self) -> list:
        """Every canonical name + alias across all brands (any-brand roster)."""
        out = []
        for name, cfg in self.brands.items():
            out.append(name)
            out.extend(cfg.get("aliases", []))
        return out

    def norm_category(self, cat):
        return self.category_aliases.get(cat, cat)


def load_config(path=DEFAULT_CONFIG) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


# --------------------------------------------------------------------------- #
#  eval records (jsonl envelope)
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_eval_records(eval_dir=DEFAULT_EVAL_DIR):
    """All eval records, sorted by file then original order (stable shards)."""
    records = []
    for fp in sorted(Path(eval_dir).glob("**/*.jsonl")):
        records.extend(read_jsonl(fp))
    return records


def shard(records, num_shards, shard_id):
    """Deterministic contiguous-by-index slice: record i -> i % num_shards."""
    if num_shards <= 1:
        return records
    return [r for i, r in enumerate(records) if i % num_shards == shard_id]


def load_shards(run_dir):
    """Read every shard_*.jsonl under a run dir, dedupe by id (last wins)."""
    by_id = {}
    for fp in sorted(Path(run_dir).glob("shard_*.jsonl")):
        for r in read_jsonl(fp):
            by_id[r["id"]] = r
    return list(by_id.values())


# --------------------------------------------------------------------------- #
#  judge output parsing
# --------------------------------------------------------------------------- #
def extract_json(text):
    """Best-effort JSON object extraction from a judge completion."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    parsed = None
    for candidate in re.findall(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return parsed


def load_prompt(name):
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  vLLM model wrapper (lazy import -> no GPU deps for metrics.py)
# --------------------------------------------------------------------------- #
def _is_qwen3(model_name):
    return "Qwen3" in model_name


def _is_mistral7(model_name):
    return "Mistral-7B-Instruct-v0.3" in model_name


class Model:
    """vLLM generation model + chat-template formatting, shared by both stages."""

    def __init__(self, model_path, max_tokens=256, temperature=0.0,
                 seed=42, gpu_memory_utilization=0.90, max_model_len=4096):
        from vllm import LLM, SamplingParams          # noqa: local import
        from transformers import AutoTokenizer

        self.model_path = str(model_path)
        self.model_name = Path(model_path).name
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            disable_log_stats=True,
            seed=seed,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        stop = ["[INST]", "</s>"] if _is_mistral7(self.model_name) else None
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, seed=seed, stop=stop,
        )

    def format(self, prompt, system=None):
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if _is_qwen3(self.model_name):
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, prompts, system=None):
        formatted = [self.format(p, system=system) for p in prompts]
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
