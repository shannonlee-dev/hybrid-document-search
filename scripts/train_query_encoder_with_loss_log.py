"""Run the unchanged PR #40 training CLI and also keep every progress.json write.

The CLI overwrites progress.json every 64 steps. This launcher wraps the module's
write_json so each progress record is also appended to progress-log.jsonl. The
wrapper is observation-only: arguments, RNG, data order and numerics are untouched.
"""

import json
import sys
from pathlib import Path

import evaluation.query_encoder_training as training

_original = training.write_json


def _logged(path, payload):
    _original(path, payload)
    path = Path(path)
    if path.name == "progress.json":
        with (path.parent / "progress-log.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(payload) + "\n")


training.write_json = _logged

from scripts.train_query_encoder import main  # noqa: E402

raise SystemExit(main(sys.argv[1:]))
