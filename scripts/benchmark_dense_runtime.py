"""Dense runtime benchmark의 CLI와 모델별 worker 프로세스 실행을 담당한다."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from evaluation import dense_runtime
from evaluation.json_io import write_json
from evaluation.latency import DEFAULT_REPEATS, DEFAULT_WARMUP
from evaluation.metrics import EVALUATION_TOP_K
from evaluation.runtime_config import CUDA_DEVICE, DEFAULT_THREADS
from scripts._cli import _positive_int

_WORKER_ERROR_TAIL_LENGTH = 6000


def _run_worker(
    config: dense_runtime.DenseRuntimeConfig,
    model: str,
    index: Path,
    phase: str,
):
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_dense_runtime",
        "--worker",
        phase,
        "--model",
        model,
        "--index-root",
        str(index),
    ]
    revision = config.revision_for(model)
    if revision is not None:
        command.extend(["--revision", revision])
    options = {
        "corpus": config.corpus,
        "queries": config.queries,
        "manifest": config.manifest,
        "expected_documents": config.expected_documents,
        "device": config.device,
        "batch_size": config.batch_size,
        "top_k": config.top_k,
        "warmup": config.warmup,
        "repeats": config.repeats,
        "threads": config.threads,
    }
    for option, value in options.items():
        command.extend(["--" + option.replace("_", "-"), str(value)])
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(
            f"{phase} exited {process.returncode}: "
            f"{process.stderr[-_WORKER_ERROR_TAIL_LENGTH:]}"
        )
    return json.loads(process.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/processed/corpus.jsonl")
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("data/processed/queries_train.jsonl")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.json")
    )
    parser.add_argument("--expected-documents", type=_positive_int, default=10000)
    parser.add_argument(
        "--model", action="append", help="반복 지정 가능; 기본: 세 후보"
    )
    parser.add_argument(
        "--revision",
        action="append",
        help="고정 40자리 SHA; 지정 시 각 --model 순서에 맞춰 하나씩 반복 지정",
    )
    parser.add_argument(
        "--device", default=CUDA_DEVICE, help="명시적 장치; 기본 cuda:0"
    )
    parser.add_argument("--threads", type=_positive_int, default=DEFAULT_THREADS)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--top-k", type=_positive_int, default=EVALUATION_TOP_K)
    parser.add_argument("--warmup", type=_positive_int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--index-root", type=Path, default=Path("indexes/dense-runtime")
    )
    parser.add_argument(
        "--output", type=Path, help="결과 JSON 경로 (일반 실행에서는 필수)"
    )
    parser.add_argument(
        "--worker", choices=("build", "validate"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    try:
        config = dense_runtime.DenseRuntimeConfig(
            corpus=args.corpus,
            queries=args.queries,
            manifest=args.manifest,
            expected_documents=args.expected_documents,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            warmup=args.warmup,
            repeats=args.repeats,
            threads=args.threads,
            index_root=args.index_root,
            model=tuple(args.model) if args.model else None,
            revision=tuple(args.revision) if args.revision else None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.worker:
        from evaluation.runtime_config import configure_runtime

        if config.model is None or len(config.model) != 1:
            parser.error("--worker requires exactly one --model")
        configure_runtime(config.threads, cuda=config.device.startswith("cuda"))
        if args.worker == "build":
            result = dense_runtime.build(config, config.model[0], config.index_root)
        else:
            result = dense_runtime.validate(config, config.index_root)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required for benchmark runs")
    if args.output.exists():
        parser.error(f"output already exists; use a new --output: {args.output}")
    try:
        _, _, provenance = dense_runtime.check_inputs(config)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    report = dense_runtime.create_report(config, provenance)
    seen_indexes: dict[Path, str] = {}
    output_path = args.output.resolve()
    for row in report["models"]:
        index = Path(row["index"]).resolve()
        model = row["model_name"]
        if index in seen_indexes:
            parser.error(
                f"index path collision: {seen_indexes[index]!r} and {model!r} "
                f"both map to {index}"
            )
        if index.exists():
            parser.error(f"index already exists; use a new --index-root: {index}")
        if output_path.is_relative_to(index):
            parser.error(f"--output must not be inside an index directory: {index}")
        seen_indexes[index] = model
    write_json(args.output, report)
    for row in report["models"]:
        phase = "build"
        try:
            index = Path(row["index"])
            if index.exists():
                raise ValueError(
                    f"index already exists; use a new --index-root: {index}"
                )
            for phase in ("build", "validate"):
                print(f"{row['model_name']}: {phase}", file=sys.stderr, flush=True)
                row["status"] = phase + "_running"
                write_json(args.output, report)
                row.update(_run_worker(config, row["model_name"], index, phase))
            row["status"] = "completed"
        except (ValueError, OSError, RuntimeError) as exc:
            row.update(status="failed", failed_stage=phase, error=str(exc))
            print(str(exc), file=sys.stderr)
        write_json(args.output, report)
    report["finished_at"] = datetime.now(UTC).isoformat()
    write_json(args.output, report)
    if any(row["status"] != "completed" for row in report["models"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
