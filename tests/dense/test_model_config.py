"""Verify Dense model settings and pinned revisions."""

import pytest

from retrievers.model_config import (
    CONFIG_PATH,
    DEFAULT_MODEL,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_MODEL_REVISION,
    MODELS,
    load_model_config,
)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('default_model = "bge"', 'default_model = "missing"', "default_model"),
        ("5617a9f61b028005a4858fdac845db406aefb181", "main", "pinned"),
        ("[models.bge]", '[models."../bge"]', "alias"),
    ],
)
def test_invalid_toml_settings_fail_early(tmp_path, old, new, message):
    path = tmp_path / "dense_models.toml"
    path.write_text(CONFIG_PATH.read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        load_model_config(path)


def test_toml_drives_default_embedding_model():
    from retrievers.dense import DenseConfig

    config = load_model_config()
    selected = config["models"][config["default_model"]]
    assert DEFAULT_MODEL_ALIAS == config["default_model"]
    assert DEFAULT_MODEL == selected["model_name"]
    assert DEFAULT_MODEL_REVISION == selected["revision"]
    assert DenseConfig().model_name == DEFAULT_MODEL
    assert DenseConfig().revision == DEFAULT_MODEL_REVISION == MODELS[DEFAULT_MODEL]
