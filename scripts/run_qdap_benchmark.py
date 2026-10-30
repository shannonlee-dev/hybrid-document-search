"""Run Issue33's full-Train QDAP-S/L and the eight-method BC Card comparison."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from evaluation.json_io import read_json, write_json
from evaluation.qdap_benchmark import judgments_for, load_pools, prepare, rows, sha
from evaluation.qdap_training import fit_official, select_validation_checkpoint
from fusion.qdap import QDAPLarge, QDAPSmall


def load_large(model_path, device="cuda"):
    from transformers import AutoModel, AutoTokenizer

    encoder = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return QDAPLarge(encoder).to(device), AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )


def token_batch(tokenizer, queries, indices, device):
    tokens = tokenizer(
        [queries[i]["text"] for i in indices],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in tokens.items()}


def train(args):
    out = args.output
    if (out / "selection.json").exists():
        raise ValueError("selection already frozen; do not retrain")
    protocol = read_json(out / "protocol.json")
    for name, digest in protocol["prepared_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError(f"prepared input changed: {name}")
    with np.load(out / "train.npz") as z:
        vectors, utilities = z["vectors"], z["quality"]
    with np.load(out / "validation.npz") as z:
        val_vectors = z["vectors"]
    train_queries = read_json(out / "train-queries.json")
    val_queries = read_json(out / "validation-queries.json")
    data = args.inputs / "data/processed/bccard"
    labels = judgments_for(data, "validation", val_queries, rows(data / "corpus.jsonl"))
    pools = load_pools(out, "validation")
    modes = [args.ce_mode] if args.ce_mode else ["literal", "reference"]
    seeds = [42, 43, 44] if args.variant == "S" else [42]
    for mode in modes:
        for seed in seeds:
            path = out / f"{args.variant}-{mode}-{seed}"
            if (path / "training.json").exists() and len(
                read_json(path / "training.json")["history"]
            ) == 5:
                print("Already trained", path, flush=True)
                continue
            torch.manual_seed(seed)
            if args.variant == "S":
                model = QDAPSmall(vectors.shape[1]).to(args.device)
                x = torch.as_tensor(vectors, device=args.device)
                vx = torch.as_tensor(val_vectors, device=args.device)

                def forward(ix):
                    return model(x[ix])

                def validation():
                    return np.concatenate(
                        [model(chunk).cpu().numpy() for chunk in vx.split(256)]
                    )
            else:
                model, tokenizer = load_large(args.model, args.device)

                def forward(ix):
                    return model(
                        **token_batch(tokenizer, train_queries, ix, args.device)
                    )

                def validation():
                    return np.concatenate(
                        [
                            model(
                                **token_batch(
                                    tokenizer,
                                    val_queries,
                                    range(st, min(st + 16, len(val_queries))),
                                    args.device,
                                )
                            )
                            .cpu()
                            .numpy()
                            for st in range(0, len(val_queries), 16)
                        ]
                    )

            fit_official(
                model,
                utilities,
                forward,
                validation,
                pools,
                labels,
                path,
                ce_mode=mode,
                seed=seed,
                microbatch=64 if args.variant == "S" else 8,
            )
            write_json(
                path / "checkpoint.json",
                {
                    "sha256": sha(path / "selected.pt"),
                    "variant": args.variant,
                    "model": str(args.model),
                    "selection_split": "validation",
                },
            )
            model = None
            if args.variant == "S":
                x = vx = None
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


def freeze(args):
    out = args.output
    if (out / "selection.json").exists():
        raise ValueError("selection already frozen")
    selected = {}
    for variant in ("S", "L"):
        records = []
        for mode in ("literal", "reference"):
            for seed in [42, 43, 44] if variant == "S" else [42]:
                path = out / f"{variant}-{mode}-{seed}"
                report = read_json(path / "training.json")
                if len(report["history"]) != 5:
                    raise ValueError("complete five epochs before freeze")
                config = read_json(path / "checkpoint.json")
                if sha(path / "selected.pt") != config["sha256"]:
                    raise ValueError("checkpoint changed")
                records.append(
                    {
                        **report["selected"],
                        "checkpoint": str(path / "selected.pt"),
                        "checkpoint_sha256": config["sha256"],
                        "variant": variant,
                    }
                )
        selected[variant] = select_validation_checkpoint(records, split="validation")
    with np.load(out / "validation.npz") as z:
        fixed = int(z["quality"].mean(0).argmax())
    data = args.inputs / "data/processed/bccard"
    cache = args.inputs / "artifacts/bccard-adaptive-vectors-resumed"
    scores = args.inputs / "artifacts/paper-guided/dat-local/bccard"
    # Local HF snapshots can use safetensors, PyTorch bin, or sharded weights.
    model_weights = sorted(args.model.glob("*.safetensors")) + sorted(
        args.model.glob("pytorch_model*.bin")
    )
    if not model_weights:
        raise ValueError("local model snapshot contains no recognized weight files")
    evaluation_inputs = [
        *sorted(data.glob("*.jsonl")),
        data / "manifest.json",
        cache / "query-vectors-test.npy",
        cache / "document-vectors.npy",
        scores / "dense.npy",
        scores / "sparse.npy",
        scores / "queries.json",
        args.inputs / "docs/experiments/adaptive/router.json",
        *model_weights,
        *sorted(args.model.rglob("*.json")),
        *sorted(args.model.glob("*.model")),
    ]
    write_json(
        out / "selection.json",
        {
            "selection_split": "validation",
            "selected": selected,
            "paper_fixed_alpha": fixed / 100,
            "protocol_sha256": sha(out / "protocol.json"),
            "model": str(args.model),
            "evaluation_input_sha256": {
                str(path): sha(path) for path in evaluation_inputs
            },
            "source_sha256": {
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    Path("fusion/qdap.py"),
                    Path("evaluation/qdap_training.py"),
                    Path("evaluation/qdap_benchmark.py"),
                    Path("evaluation/qdap_execution.py"),
                    Path("evaluation/qdap.py"),
                ]
            },
        },
    )
    print("FROZEN", selected, "Fixed", fixed / 100, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "train", "freeze", "evaluate", "latency", "verify"]
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/shannon/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
        ),
    )
    parser.add_argument("--variant", choices=["S", "L"], default="S")
    parser.add_argument("--ce-mode", choices=["literal", "reference"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(4):
        if args.stage == "prepare":
            prepare(args.inputs, args.output)
        elif args.stage == "train":
            train(args)
        elif args.stage == "freeze":
            freeze(args)
        else:
            from evaluation.qdap_execution import evaluate, latency, verify

            {"evaluate": evaluate, "latency": latency, "verify": verify}[args.stage](
                args
            )


if __name__ == "__main__":
    main()
