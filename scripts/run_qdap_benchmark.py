"""Run Issue33's full-Train QDAP-S/L and the eight-method BC Card comparison."""

import argparse
import gc
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from evaluation.json_io import read_json, write_json
from evaluation.qdap_artifacts import (
    check,
    evaluation_files,
    hashes,
    model_identity,
    provenance,
    sha,
    vector_cache,
    verify_prepared,
)


def load_large(model_path, device="cuda"):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from fusion.qdap import QDAPLarge

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
    import torch

    from evaluation.qdap_benchmark import judgments_for, load_pools, rows
    from evaluation.qdap_training import fit_official
    from fusion.qdap import QDAPSmall

    out = args.output
    if (out / "selection.json").exists():
        raise ValueError("selection already frozen; do not retrain")
    protocol = verify_prepared(out)
    with np.load(out / "train.npz") as z:
        vectors, utilities = z["vectors"], z["quality"]
    with np.load(out / "validation.npz") as z:
        val_vectors = z["vectors"]
    train_queries = read_json(out / "train-queries.json")
    val_queries = read_json(out / "validation-queries.json")
    data = args.inputs / "data/processed/bccard" if args.inputs else None
    if protocol.get("schema_version") == 2:
        labels = [
            {int(k): v for k, v in row.items()}
            for row in read_json(out / "validation-qrels.json")
        ]
    else:
        labels = judgments_for(
            data, "validation", val_queries, rows(data / "corpus.jsonl")
        )
    identity = model_identity(args) if args.variant == "L" else None
    if identity is not None:
        encoded = protocol["encoding"]["model"]
        if (
            encoded["model_name"] != identity["name"]
            or encoded["revision"] != identity["revision"]
        ):
            raise ValueError("training model differs from frozen vector model revision")
    pools = load_pools(out, "validation")
    modes = [args.ce_mode] if args.ce_mode else ["literal", "reference"]
    seeds = [42, 43, 44] if args.variant == "S" else [42]
    for mode in modes:
        for seed in seeds:
            path = out / f"{args.variant}-{mode}-{seed}"
            if path.exists():
                raise FileExistsError(
                    f"training candidate already exists: {path}; choose a new run"
                )
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
                    "execution": provenance(),
                    "model_identity": identity,
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
    from evaluation.qdap_training import select_validation_checkpoint

    out = args.output
    if (out / "selection.json").exists():
        raise ValueError("selection already frozen")
    protocol = verify_prepared(out, ["validation.npz"])
    selected = {}
    identity = model_identity(args)
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
                if variant == "L":
                    if config.get("model_identity") != identity:
                        raise ValueError(
                            "training model identity differs from freeze model"
                        )
                records.append(
                    {
                        **report["selected"],
                        "checkpoint": (path / "selected.pt")
                        .relative_to(out)
                        .as_posix(),
                        "checkpoint_sha256": config["sha256"],
                        "variant": variant,
                    }
                )
        selected[variant] = select_validation_checkpoint(records, split="validation")
    with np.load(out / "validation.npz") as z:
        fixed = int(z["quality"].mean(0).argmax())
    evaluation_inputs = evaluation_files(args)
    protocol = read_json(out / "protocol.json")
    for key, path in evaluation_inputs.items():
        if key in protocol.get("input_sha256", {}):
            check(path, protocol["input_sha256"][key], key)
    write_json(
        out / "selection.json",
        {
            "schema_version": 2,
            "run_id": protocol["run_id"],
            "selection_split": "validation",
            "selected": selected,
            "paper_fixed_alpha": fixed / 100,
            "protocol_sha256": sha(out / "protocol.json"),
            "model_identity": identity,
            "evaluation_input_sha256": hashes(evaluation_inputs),
            "source_sha256": {
                str(p): sha(p)
                for p in [
                    Path("scripts/run_qdap_benchmark.py"),
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "train", "freeze", "evaluate", "latency", "verify"]
    )
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--output", "--run-dir", dest="output", type=Path, required=True
    )
    parser.add_argument("--vector-cache", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--checkpoint-s", type=Path)
    parser.add_argument("--checkpoint-l", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        help="Local model snapshot used for this run",
    )
    parser.add_argument("--variant", choices=["S", "L"], default="S")
    parser.add_argument("--ce-mode", choices=["literal", "reference"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if (
        args.stage in ("prepare", "freeze", "evaluate", "latency", "verify")
        and args.inputs is None
    ):
        parser.error("--inputs is required for this stage")
    if (
        args.stage in ("freeze", "evaluate", "latency")
        or (args.stage == "train" and args.variant == "L")
    ) and args.model is None:
        parser.error("--model is required for this stage")
    if args.stage == "freeze" or (args.stage == "train" and args.variant == "L"):
        if not args.model_name or not args.model_revision:
            parser.error(
                "--model-name and --model-revision are required for model identity"
            )
    if args.stage == "verify":
        from evaluation.qdap_report import enterprise_report

        report = enterprise_report(args.inputs, args.output / "per-query.jsonl")
        print(
            {
                "scope": "artifact replay only",
                "queries": report["dataset"]["query_count"],
                "methods": list(report["methods"]),
            }
        )
        return
    import torch

    torch.set_num_threads(4)
    with threadpool_limits(4):
        if args.stage == "prepare":
            from evaluation.qdap_benchmark import prepare

            prepare(args.inputs, args.output, vector_cache(args))
        elif args.stage == "train":
            train(args)
        elif args.stage == "freeze":
            freeze(args)
        else:
            from evaluation.qdap_execution import evaluate, latency

            {"evaluate": evaluate, "latency": latency}[args.stage](args)


if __name__ == "__main__":
    main()
