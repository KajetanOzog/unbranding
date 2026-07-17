import json
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BRANDS_CONFIG = (
    PROJECT_ROOT / "config" / "brands.json"
)

DEFAULT_EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "config" / "experiment.yaml"
)


def load_brands_config(path=DEFAULT_BRANDS_CONFIG):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_experiment_config(path=DEFAULT_EXPERIMENT_CONFIG):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_configs(brands_path=DEFAULT_BRANDS_CONFIG, experiment_path=DEFAULT_EXPERIMENT_CONFIG):
    brands_cfg = load_brands_config(brands_path)

    experiment_cfg = load_experiment_config(
        experiment_path
    )

    brand_name = experiment_cfg["brand"]

    if brand_name not in brands_cfg:
        raise ValueError(
            f"Brand '{brand_name}' not found in brands.json"
        )

    brand_cfg = brands_cfg[brand_name]

    return (
        experiment_cfg,
        brand_cfg,
    )

def get_brand_name(experiment_cfg):
    return experiment_cfg["brand"]

def get_judge_model_path(experiment_cfg):
    return experiment_cfg["models"]["judge"]["path"]

def get_metric_list(experiment_cfg):
    return experiment_cfg["metrics"]

def get_base_model_path(experiment_cfg):
    return experiment_cfg["models"]["base"]["path"]

def get_experiments_root(experiment_cfg):
    return experiment_cfg["experiment"]["experiments_root"]

def get_model_family(experiment_cfg):
    return experiment_cfg["experiment"]["model_family"]

def get_target_epochs(experiment_cfg):
    return experiment_cfg["experiment"]["target_epochs"]

def get_generation_config(experiment_cfg):
    return experiment_cfg["generation"]

def get_evaluation_config(experiment_cfg):
    return experiment_cfg["evaluation"]

def get_results_config(experiment_cfg):
    return experiment_cfg["results"]
