"""Write a query checkpoint identity file from a training run's own records.

Usage:
  python -m scripts.make_query_checkpoint_identity --train-dir RUN/train-seed43 \
      --encoding PATH/encoding.json --output config/query_checkpoints/NAME.json

Every field comes from recorded evidence: the checkpoint SHA-256 and selected
step from frozen.json, the layer count and base snapshot hashes from the
training protocol.json (whose own SHA-256 is bound in frozen.json), and the base
model name from the vector cache encoding.json whose revision must agree.
"""

import argparse
import json
from pathlib import Path

from retrievers.adapted_query import (
    IDENTITY_SCHEMA_VERSION,
    file_sha256,
    load_checkpoint_identity,
)


def build_identity(train_dir, encoding_path):
    train_dir = Path(train_dir)
    frozen = json.loads((train_dir / "frozen.json").read_text())
    protocol = json.loads((train_dir / "protocol.json").read_text())
    encoding = json.loads(Path(encoding_path).read_text())
    if file_sha256(train_dir / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("training protocol does not match frozen.json")
    if file_sha256(train_dir / "selected-last-block.pt") != frozen["checkpoint_sha256"]:
        raise ValueError("checkpoint does not match frozen.json")
    if encoding["model"]["revision"] != protocol["revision"]:
        raise ValueError("vector cache model revision differs from training")
    identity = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "layers": protocol["trainable_layers"],
        "base_model": {
            "model_name": encoding["model"]["model_name"],
            "revision": protocol["revision"],
            "files_sha256": protocol["model_files_sha256"],
        },
        "source": {
            "training_run_id": frozen["run_id"],
            "selected_step": frozen["selected_step"],
            "training_protocol_sha256": frozen["protocol_sha256"],
            "encoding_record_sha256": file_sha256(encoding_path),
        },
    }
    return identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--encoding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    identity = build_identity(args.train_dir, args.encoding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    load_checkpoint_identity(args.output)
    print(json.dumps(identity, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
