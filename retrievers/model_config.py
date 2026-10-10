"""Load Dense model candidates and the project default from TOML."""

import re
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/dense_models.toml"
COMMIT_SHA_PATTERN = r"[0-9a-f]{40}"
_MODEL_ALIAS_PATTERN = r"[a-z][a-z0-9_-]*"


def load_model_config(path=CONFIG_PATH):
    """Load model settings, rejecting invalid aliases, defaults and revision pins."""
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("dense_models.toml: models must be a non-empty table")
    names = set()
    for alias, model in models.items():
        if not re.fullmatch(_MODEL_ALIAS_PATTERN, alias):
            raise ValueError("dense_models.toml: invalid model alias")
        if not isinstance(model, dict):
            raise ValueError(f"dense_models.toml: invalid model table: {alias}")
        name, revision = model.get("model_name"), model.get("revision")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(
                "dense_models.toml: model names must be non-empty and unique"
            )
        if not isinstance(revision, str) or not re.fullmatch(
            COMMIT_SHA_PATTERN, revision
        ):
            raise ValueError(
                f"dense_models.toml: {alias} needs a pinned 40-character SHA"
            )
        names.add(name)
    default = config.get("default_model")
    if not isinstance(default, str) or default not in models:
        raise ValueError(
            "dense_models.toml: default_model must name a configured alias"
        )
    return config


# 공개 상수는 TOML 설정에 의존하므로 검증 후 초기화한다.
_CONFIG = load_model_config()
MODEL_CONFIGS = _CONFIG["models"]
DEFAULT_MODEL_ALIAS = _CONFIG["default_model"]
ALIASES = {alias: config["model_name"] for alias, config in MODEL_CONFIGS.items()}
MODELS = {config["model_name"]: config["revision"] for config in MODEL_CONFIGS.values()}
DEFAULT_MODEL = ALIASES[DEFAULT_MODEL_ALIAS]
DEFAULT_MODEL_REVISION = MODELS[DEFAULT_MODEL]
