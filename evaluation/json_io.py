"""Shared JSON output for evaluation reports and experiment checkpoints."""

import json


def write_json(path, value):
    """Replace a JSON file via a sibling temporary file, rejecting NaN and infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
