"""Select, execute, combine and verify local DAT on the BC Card primary protocol."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("select", "evaluate", "baseline", "combine", "verify")
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Prepared BC Card dataset directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New run directory, or policy JSON for select",
    )
    parser.add_argument(
        "--baseline", type=Path, help="Verified canonical bundle for combine"
    )
    parser.add_argument(
        "--live", type=Path, help="Verified DAT/Fixed bundle for combine"
    )
    parser.add_argument("--vector-cache", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--canonical-summary", type=Path)
    parser.add_argument("--encoder-model", type=Path)
    parser.add_argument("--judge-model", type=Path)
    parser.add_argument("--encoder-name", default="BAAI/bge-m3")
    parser.add_argument(
        "--encoder-revision", default="5617a9f61b028005a4858fdac845db406aefb181"
    )
    parser.add_argument("--judge-name", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument(
        "--judge-revision", default="e61197ed45024b0ed8a2d74b80b4d909f1255473"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="First pass quality/warmup, remaining passes latency",
    )
    parser.add_argument("--run-id", default="bccard-local-dat")
    args = parser.parse_args(argv)
    if args.stage == "evaluate" and args.repeats < 2:
        parser.error(
            "evaluate requires --repeats >= 2 to measure after the quality pass"
        )
    required = {
        "select": ["vector_cache"],
        "evaluate": ["vector_cache", "policy", "encoder_model", "judge_model"],
        "combine": ["baseline", "live"],
        "baseline": ["vector_cache", "canonical_summary"],
        "verify": [],
    }[args.stage]
    for name in required:
        if getattr(args, name) is None:
            parser.error(
                "--" + name.replace("_", "-") + " is required for " + args.stage
            )
    # NumPy가 BLAS를 로드한 뒤 제한해야 새 프로세스에서도 제한이 적용된다.
    from threadpoolctl import threadpool_limits

    from evaluation.contextual import audit_main, select_vector_policy
    from evaluation.contextual_report import combine_handoffs, verify_handoff

    with threadpool_limits(2):
        if args.stage == "verify":
            print(json.dumps(verify_handoff(args.data, args.output), indent=2))
        elif args.stage == "select":
            selected = select_vector_policy(args.data, args.vector_cache, args.output)
            print(json.dumps(selected, indent=2))
        elif args.stage == "combine":
            combine_handoffs(args.data, args.baseline, args.live, args.output)
        elif args.stage == "baseline":
            import faiss

            faiss.omp_set_num_threads(2)
            audit_main(
                args.data,
                args.vector_cache,
                args.canonical_summary,
                args.output,
                run_id=args.run_id,
            )
        else:
            import torch

            from evaluation.contextual_execution import live_evaluate

            torch.set_num_threads(2)
            live_evaluate(args)


if __name__ == "__main__":
    main()
