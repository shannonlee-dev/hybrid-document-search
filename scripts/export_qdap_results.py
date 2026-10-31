"""Export frozen QDAP per-query results and replay the main canonical baselines."""

import argparse
from pathlib import Path

from evaluation.qdap_handoff import audit_main, export_frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--vector-cache", type=Path, help="Used only by canonical audit"
    )
    parser.add_argument("--run-id", help="Identity for the QDAP JSONL export")
    parser.add_argument("--main-run-id", help="Identity for the main baseline audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--canonical-summary",
        type=Path,
        default=Path("docs/experiments/bccard/test-summary.json"),
    )
    parser.add_argument(
        "--stage", choices=["all", "export", "audit", "enterprise"], default="all"
    )
    parser.add_argument(
        "--per-query",
        type=Path,
        help="JSONL handoff to convert without model inference",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        help="Also align this enterprise report to the primary query IDs",
    )
    args = parser.parse_args(argv)
    if args.stage == "enterprise":
        from evaluation.json_io import read_json, write_json
        from evaluation.qdap_report import align_reference_report, enterprise_report

        source = args.per_query or args.output / "per-query.jsonl"
        report = enterprise_report(
            args.inputs, source, canonical_summary=args.canonical_summary
        )
        reference = None
        if args.reference_report:
            reference = align_reference_report(read_json(args.reference_report), report)
            reference["source_report"] = str(args.reference_report)
        from evaluation.qdap_artifacts import unused_target

        unused_target(args.output / "report.json")
        if reference is not None:
            unused_target(args.output / "reference-report.json")
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "report.json", report)
        if reference is not None:
            write_json(args.output / "reference-report.json", reference)
        return
    if args.run is None:
        parser.error("--run is required for all/export/audit stages")
    if args.stage in ("all", "export") and not (args.run_id and args.run_id.strip()):
        parser.error("--run-id is required for all/export stages")
    if args.stage in ("all", "audit") and not (
        args.main_run_id and args.main_run_id.strip()
    ):
        parser.error("--main-run-id is required for all/audit stages")
    if args.stage in ("all", "export"):
        export_frozen(args)
    if args.stage in ("all", "audit"):
        audit_main(args)


if __name__ == "__main__":
    main()
