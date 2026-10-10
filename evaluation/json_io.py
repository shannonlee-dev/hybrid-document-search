"""Shared JSON output for evaluation reports and experiment checkpoints."""

import json
from pathlib import Path


def write_json(path, value):
    """Replace a JSON file via a sibling temporary file, rejecting NaN and infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 같은 파일시스템에서 교체해 기록 도중 기존 체크포인트가 잘리지 않게 한다.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
