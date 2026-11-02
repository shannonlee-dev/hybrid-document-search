"""Collect, analyze and check the Finance encoder service-path evidence.

Commands
  collect  copy result files from a local results directory into the evidence
           directory, replacing GPU UUIDs with GPU indexes (needs nvidia-smi)
  analyze  regression breakdown, summary tables and SVG figures from committed files
  check    recompute `analyze` in memory and verify MANIFEST.sha256
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from math import fsum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-service-eval"
RESULT_FILES = (
    "prepare.json", "parity.json", "accuracy.json", "bccard-per-query.jsonl",
    "latency-bge.json", "latency-fin.json",
    "http-bge.json", "http-fin.json", "http-bge-rep2.json", "http-fin-rep2.json",
    *(f"cold-start-{e}-run{i}.json" for e in ("bge", "fin") for i in (1, 2, 3)),
    "external.json", "miracl10k-per-query.jsonl", "strategyqa-per-query.jsonl",
)
DERIVED = ("analysis.json", "paired-differences.svg", "latency-stages.svg", "MANIFEST.sha256")
PAIRS = {"A": ("A-fin", "A-bge"), "B": ("B-fin", "B-bge"), "D": ("D-fin", "D-bge")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_by(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out.setdefault(row["condition"], {})[row["query_id"]] = row
    return out


# -------------------------------------------------------------------- collect
def collect(results_dir, evidence_dir=EVIDENCE):
    mapping = {}
    lines = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    for line in lines:
        index, uuid = (part.strip() for part in line.split(","))
        mapping[uuid] = f"GPU{index}"
    pattern = re.compile(r"GPU-[0-9a-f-]{36}")
    for name in RESULT_FILES:
        source = Path(results_dir) / name
        text = source.read_text(encoding="utf-8")
        text = pattern.sub(lambda m: mapping.get(m.group(0), "GPU?"), text)
        if "GPU?" in text:
            raise ValueError(f"unmapped GPU UUID in {name}")
        (Path(evidence_dir) / name).write_text(text, encoding="utf-8")
    return {"files": len(RESULT_FILES), "gpu_uuid_map": sorted(mapping.values())}


# -------------------------------------------------------------------- analyze
def regressions(rows, treated, control):
    """Classify per-query nDCG@10 losses of treated vs control (descriptive only)."""
    kinds = {"relevant_left_top10": 0, "relevant_moved_down": 0, "other": 0}
    by_prefix = {}
    magnitudes = []
    for query_id, control_row in rows[control].items():
        treated_row = rows[treated][query_id]
        delta = treated_row["ndcg@10"] - control_row["ndcg@10"]
        prefix = query_id.split("_")[0]
        bucket = by_prefix.setdefault(prefix, {"queries": 0, "losses": 0, "gains": 0})
        bucket["queries"] += 1
        if delta > 1e-12:
            bucket["gains"] += 1
        if delta < -1e-12:
            bucket["losses"] += 1
            magnitudes.append(delta)
            before, after = control_row["relevant_rank_top10"], treated_row["relevant_rank_top10"]
            if before is not None and after is None:
                kinds["relevant_left_top10"] += 1
            elif before is not None and after is not None and after > before:
                kinds["relevant_moved_down"] += 1
            else:
                kinds["other"] += 1
    magnitudes.sort()
    return {
        "losses": len(magnitudes),
        "loss_kinds": kinds,
        "mean_loss": fsum(magnitudes) / len(magnitudes) if magnitudes else 0.0,
        "largest_losses": magnitudes[:5],
        "by_query_id_prefix": by_prefix,
    }


def analyze(evidence_dir=EVIDENCE):
    evidence_dir = Path(evidence_dir)
    accuracy = read_json(evidence_dir / "accuracy.json")
    external = read_json(evidence_dir / "external.json")
    bccard = rows_by(evidence_dir / "bccard-per-query.jsonl")
    differences = []
    for dataset, comparisons in (
        ("BC Card Test", accuracy["comparisons"]),
        ("Ko-MIRACL dev", external["miracl10k"]["comparisons"]),
        ("Ko-StrategyQA", external["strategyqa"]["comparisons"]),
    ):
        for pair in ("A", "B", "D"):
            entry = comparisons[pair]["ndcg@10"]
            differences.append({
                "dataset": dataset, "pair": pair,
                "mean": entry["mean_difference"], "ci95": entry["ci95"],
                "verdict": entry["verdict"], "queries": entry["queries"],
                "wins_ties_losses": entry["wins_ties_losses"],
            })
    latency = {}
    for encoder in ("bge", "fin"):
        staged = read_json(evidence_dir / f"latency-{encoder}.json")
        http = [read_json(evidence_dir / f"http-{encoder}{tag}.json") for tag in ("", "-rep2")]
        cold = [read_json(evidence_dir / f"cold-start-{encoder}-run{i}.json") for i in (1, 2, 3)]
        latency[encoder] = {
            "stages_ms": {k: {q: v[q] for q in ("mean", "p50", "p95", "p99")} for k, v in staged["stages_ms"].items()},
            "staged_ranking_mismatches": staged["staged_ranking_mismatches"],
            "torch_cuda_max_memory_allocated_mib": staged["torch_cuda_max_memory_allocated_bytes"] / 2**20,
            "process_rss_mib": staged["process_memory"]["VmRSS"] / 2**20,
            "process_hwm_mib": staged["process_memory"]["VmHWM"] / 2**20,
            "http": [
                {
                    method: {
                        "sequential_mean_ms": run["results"][method]["sequential_ms"]["mean"],
                        "sequential_p50_ms": run["results"][method]["sequential_ms"]["p50"],
                        "sequential_p95_ms": run["results"][method]["sequential_ms"]["p95"],
                        "sequential_p99_ms": run["results"][method]["sequential_ms"]["p99"],
                        **{
                            f"rps_c{c}": run["results"][method]["throughput"][c]["requests_per_second"]
                            for c in ("1", "4")
                        },
                    }
                    for method in ("hybrid", "dense")
                }
                for run in http
            ],
            "server_rss_mib": [run["server_process_memory"]["VmRSS"] / 2**20 for run in http],
            "cold_start": [
                {"ready_seconds": c["service_ready_seconds"], "first_hybrid_ms": c["first_hybrid_query_ms"]}
                for c in cold
            ],
        }
    return {
        "schema_version": 1,
        "inputs_sha256": {name: sha256(evidence_dir / name) for name in RESULT_FILES},
        "paired_ndcg_differences": differences,
        "bccard_regressions": {
            pair: regressions(bccard, *PAIRS[pair]) for pair in ("A", "B", "D")
        },
        "latency": latency,
    }


# -------------------------------------------------------------------- figures
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834")
PAIR_LABEL = {"A": "서비스 Hybrid (.05)", "B": "Fixed α=.84", "D": "Dense-only"}


FONT = (
    "-apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', "
    "'Segoe UI', Helvetica, Arial, sans-serif"
)


def _svg(width, height, body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}" '
        f'font-family="{FONT}">\n'
        f"<title>{title}</title>\n"
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>\n'
        + "\n".join(body) + "\n</svg>\n"
    )


def differences_figure(analysis):
    rows = analysis["paired_ndcg_differences"]
    width, top, row_h = 760, 70, 26
    height = top + row_h * (len(rows) + 3) + 40
    x0, x1, lo, hi = 300, 720, -0.05, 0.05
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    body = [
        f'<text x="20" y="26" font-size="15" font-weight="600" fill="{INK}">Finance Seed43 − 원본 BGE, nDCG@10 query 단위 paired 차이</text>',
        f'<text x="20" y="44" font-size="12" fill="{MUTED}">점 = 평균 차이, 선 = bootstrap 95% CI (seed 20261102, 20,000 resamples). 0의 오른쪽이 Finance 우위.</text>',
    ]
    for t in (-0.04, -0.02, 0, 0.02, 0.04):
        body.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 6}" y2="{height - 40}" stroke="{GRID if t else MUTED}" stroke-width="1"/>')
        body.append(f'<text x="{sx(t):.1f}" y="{height - 24}" font-size="11" text-anchor="middle" fill="{MUTED}">{t:+.2f}</text>')
    y, current = top, None
    for row in rows:
        if row["dataset"] != current:
            current = row["dataset"]
            n = row["queries"]
            body.append(f'<text x="20" y="{y + 12}" font-size="12" font-weight="600" fill="{INK}">{current} (n={n})</text>')
            y += row_h
        low, high = row["ci95"]
        body.append(f'<text x="36" y="{y + 4}" font-size="12" fill="{INK}">{PAIR_LABEL[row["pair"]]}</text>')
        body.append(f'<text x="{x0 - 10}" y="{y + 4}" font-size="11" text-anchor="end" fill="{MUTED}">{row["mean"]:+.4f}</text>')
        body.append(f'<line x1="{sx(max(low, lo)):.1f}" x2="{sx(min(high, hi)):.1f}" y1="{y}" y2="{y}" stroke="{SERIES[0]}" stroke-width="2"/>')
        body.append(
            f'<circle cx="{sx(row["mean"]):.1f}" cy="{y}" r="5" fill="{SERIES[0]}" stroke="{SURFACE}" stroke-width="2">'
            f'<title>{current} {PAIR_LABEL[row["pair"]]}: {row["mean"]:+.4f} [{low:+.4f}, {high:+.4f}] {row["verdict"]}</title></circle>'
        )
        y += row_h
    return _svg(width, height, body, "Finance minus BGE paired nDCG@10 differences")


def latency_figure(analysis):
    stages = (
        ("query_encoding", "Query encoding"), ("dense_service_hybrid", "Dense scoring (Hybrid)"),
        ("dense_faiss", "Dense FAISS (Dense-only)"), ("bm25", "BM25 score_all"),
        ("fusion_service_hybrid", "Fusion (Hybrid .05)"), ("fusion_fixed", "Fusion (Fixed α)"),
        ("total_service_hybrid", "Total Hybrid .05"), ("total_fixed", "Total Fixed α"),
        ("total_dense", "Total Dense-only"),
    )
    width, top, row_h = 840, 80, 34
    height = top + row_h * len(stages) + 50
    x0, x1, hi = 230, 740, 20.0
    sx = lambda v: x0 + min(v, hi) / hi * (x1 - x0)  # noqa: E731
    body = [
        f'<text x="20" y="26" font-size="15" font-weight="600" fill="{INK}">BC Card Test 721 × 3, 단계별 latency (in-process, GPU0, ms)</text>',
        f'<text x="20" y="44" font-size="12" fill="{MUTED}">막대 = P50, 선 끝 = P95. 위 = 원본 BGE, 아래 = Finance Seed43.</text>',
        f'<line x1="24" x2="44" y1="60" y2="60" stroke="{SERIES[0]}" stroke-width="6"/><text x="50" y="64" font-size="11" fill="{INK}">원본 BGE</text>',
        f'<line x1="130" x2="150" y1="60" y2="60" stroke="{SERIES[1]}" stroke-width="6"/><text x="156" y="64" font-size="11" fill="{INK}">Finance Seed43</text>',
    ]
    for t in (0, 5, 10, 15, 20):
        body.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 8}" y2="{height - 44}" stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<text x="{sx(t):.1f}" y="{height - 28}" font-size="11" text-anchor="middle" fill="{MUTED}">{t}</text>')
    for i, (key, label) in enumerate(stages):
        y = top + i * row_h
        body.append(f'<text x="{x0 - 10}" y="{y + 12}" font-size="12" text-anchor="end" fill="{INK}">{label}</text>')
        for j, encoder in enumerate(("bge", "fin")):
            stats = analysis["latency"][encoder]["stages_ms"][key]
            yy = y + 4 + j * 11
            body.append(
                f'<rect x="{x0}" y="{yy - 4}" width="{sx(stats["p50"]) - x0:.1f}" height="8" rx="2" fill="{SERIES[j]}">'
                f'<title>{label} {encoder}: p50 {stats["p50"]:.2f} ms, p95 {stats["p95"]:.2f} ms</title></rect>'
            )
            body.append(f'<line x1="{sx(stats["p50"]):.1f}" x2="{sx(stats["p95"]):.1f}" y1="{yy}" y2="{yy}" stroke="{SERIES[j]}" stroke-width="2"/>')
        body.append(f'<text x="{x1 + 4}" y="{y + 12}" font-size="10" fill="{MUTED}">{analysis["latency"]["bge"]["stages_ms"][key]["p50"]:.1f}/{analysis["latency"]["fin"]["stages_ms"][key]["p50"]:.1f}</text>')
    return _svg(width, height, body, "Staged latency per encoder")


def write_manifest(evidence_dir=EVIDENCE):
    evidence_dir = Path(evidence_dir)
    lines = [
        f"{sha256(p)}  {p.name}"
        for p in sorted(evidence_dir.iterdir())
        if p.is_file() and p.name != "MANIFEST.sha256"
    ]
    (evidence_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def write_analysis(evidence_dir=EVIDENCE):
    evidence_dir = Path(evidence_dir)
    result = analyze(evidence_dir)
    (evidence_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (evidence_dir / "paired-differences.svg").write_text(differences_figure(result))
    (evidence_dir / "latency-stages.svg").write_text(latency_figure(result))
    write_manifest(evidence_dir)
    return result


def check(evidence_dir=EVIDENCE):
    evidence_dir = Path(evidence_dir)
    manifest = (evidence_dir / "MANIFEST.sha256").read_text().splitlines()
    listed = {}
    for line in manifest:
        digest, name = line.split("  ", 1)
        listed[name] = digest
        if sha256(evidence_dir / name) != digest:
            raise ValueError(f"manifest mismatch: {name}")
    present = {p.name for p in evidence_dir.iterdir() if p.is_file()} - {"MANIFEST.sha256"}
    if set(listed) != present:
        raise ValueError(f"unlisted or missing files: {sorted(set(listed) ^ present)}")
    recomputed = json.loads(json.dumps(analyze(evidence_dir)))
    if recomputed != read_json(evidence_dir / "analysis.json"):
        raise ValueError("analysis.json does not match recomputation")
    if (evidence_dir / "paired-differences.svg").read_text() != differences_figure(recomputed):
        raise ValueError("paired differences figure differs")
    if (evidence_dir / "latency-stages.svg").read_text() != latency_figure(recomputed):
        raise ValueError("latency figure differs")
    return {"verified": True, "files": len(listed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect").add_argument("--results-dir", type=Path, required=True)
    sub.add_parser("analyze")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    if args.command == "collect":
        result = collect(args.results_dir)
    elif args.command == "analyze":
        result = write_analysis()
        result = {"paired_ndcg_differences": result["paired_ndcg_differences"]}
    else:
        result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
