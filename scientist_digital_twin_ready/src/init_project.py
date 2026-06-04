"""
Phase 0 - Project scaffold initializer.
Creates all required directories and validates config.
"""
import os
import yaml
from pathlib import Path

REQUIRED_DIRS = [
    "data/raw/papers",
    "data/raw/transcripts",
    "data/raw/audio/segments",
    "data/processed/papers",
    "data/indices",
    "data/sessions",
    "data/voice",
    "src/ingestion",
    "src/processing",
    "src/retrieval",
    "src/persistence",
    "src/generation",
    "src/voice",
    "src/utils",
    "config",
    "evals",
]

CONFIG_PATH = Path("config/scientist.yaml")
REQUIRED_FIELDS = ["name", "wikidata_qid", "semantic_scholar_id", "wikipedia_title"]


def create_dirs(base: Path = Path(".")) -> None:
    for d in REQUIRED_DIRS:
        (base / d).mkdir(parents=True, exist_ok=True)
    print("✅ All directories created.")


def validate_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    scientist = config.get("scientist", {})
    missing = [k for k in REQUIRED_FIELDS if not scientist.get(k)]
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    print(f"✅ Config valid: {scientist['name']} | QID={scientist['wikidata_qid']}")
    return config


if __name__ == "__main__":
    base = Path(".")
    create_dirs(base)
    config = validate_config()
    print("\n--- Scientist Config ---")
    import json
    print(json.dumps(config["scientist"], indent=2))
