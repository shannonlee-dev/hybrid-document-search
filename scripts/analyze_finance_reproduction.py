"""Export, analyze and check the Finance encoder two-seed reproduction evidence.

Commands
  export-evidence  copy the compact run files byte-for-byte and derive small JSON
                   records from a local run directory (needs the local run only).
  analyze          compute the paired bootstrap, Validation alpha stability and SVG
                   figures from committed evidence only.
  check            recompute `analyze` in memory and verify MANIFEST.sha256.

The bootstrap and alpha analyses are post-hoc uncertainty and sensitivity
analyses of an already-frozen run. They select nothing and read no Test labels
beyond the committed per-query metrics.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-repro-blackwell"
COMPACT = ("frozen.json", "protocol.json", "per-query.jsonl", "per-query-manifest.json")
BOOTSTRAP_SEED = 20261102
BOOTSTRAP_RESAMPLES = 20000
TOLERANCES = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3)
STEPS = (0, 64, 128, 256, 512, 1024, 2048, 4096)

# Seed42 reference curve from the historical run linked by PR #40's documentation.
# Commit c95c3fb is on no branch; it is fetchable by SHA from the upstream remote.
SEED42_REFERENCE = {
    "source": "c95c3fb106287749fbec268fb52768e854e00c05:docs/validation/2026-11-01/finance/protocol.json",
    "source_sha256": "69caf8eda46520c190a7128e7108ff17141b010abf8ee5d57efc94d591882ef8",
    "selected_step": 4096,
    "checkpoint_sha256": "361e09ee8496f3227ffdc5a25ae27acb28d679ea123e72bcb8a10dff1476c245",
    "validation_alpha_dense": 0.82,
    "validation_fixed_ndcg": 0.8298404923658383,
    "history": {
        0: 0.7873389562551699,
        64: 0.7964060084596272,
        128: 0.7986196392857288,
        256: 0.7992135447781619,
        512: 0.7993330207602344,
        1024: 0.8039586792538053,
        2048: 0.8119843080445067,
        4096: 0.815199858454069,
    },
}
REFERENCE_RUN_REL = "docs/validation/2026-11-02/finance-retrain"
REFERENCE_RUN = ROOT / REFERENCE_RUN_REL


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def per_query(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.setdefault(row["method"], {})[row["query_id"]] = row
    return rows


# --------------------------------------------------------------------- export
def export_evidence(run_dir, output_dir):
    """Copy compact files byte-for-byte and derive path-free JSON records."""
    import numpy as np

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in COMPACT:
        shutil.copyfile(run_dir / "compact" / name, output_dir / name)
    for name in ("preregistration.md", "selection-freeze.json"):
        shutil.copyfile(run_dir / name, output_dir / name)

    quality_path = run_dir / "replay" / "validation-quality.npz"
    quality = np.load(quality_path, allow_pickle=False)
    frozen = read_json(run_dir / "replay" / "frozen.json")
    queries = read_json(run_dir / "scores" / "queries.json")
    validation_ids = [q["query_id"] for q in queries if q["split"] == "validation"]
    alphas = [round(a, 2) for a in frozen["alphas"]]
    reference_alpha = {"baseline": 0.84, "seed42": 0.82, "seed43": 0.83}
    curves, paired = {}, {}
    for name in quality.files:
        table = quality[name]
        if table.shape != (len(validation_ids), len(alphas), 3):
            raise ValueError("validation table does not match primary queries")
        curves[name] = [float(v) for v in table[:, :, 0].mean(0)]
        keep = sorted(
            {frozen["models"][name]["alpha_dense"], reference_alpha[name], 0.84}
        )
        paired[name] = {
            f"{alpha:.2f}": [float(v) for v in table[:, int(round(alpha * 100)), 0]]
            for alpha in keep
        }
    write_json(
        output_dir / "validation-alpha-curves.json",
        {
            "metric": "Validation primary binary nDCG@10 (selection metric), Fixed fusion",
            "queries": len(validation_ids),
            "query_ids_sha256": hashlib.sha256(
                "\n".join(validation_ids).encode()
            ).hexdigest(),
            "alphas": alphas,
            "mean_ndcg": curves,
            "per_query_ndcg_at_selected_reference_and_shared_alpha": paired,
            "source": {
                "file": "replay/validation-quality.npz (not committed)",
                "sha256": sha256(quality_path),
            },
        },
    )

    loss = {}
    for seed in (42, 43):
        log = run_dir / f"train-seed{seed}" / "progress-log.jsonl"
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        loss[f"seed{seed}"] = {
            "source_sha256": sha256(log),
            "note": "single-batch training loss written by the CLI every 64 steps",
            "points": [
                {"step": e["step"], "loss": e["loss"], "seconds": e["seconds"]}
                for e in entries
            ],
        }
    write_json(output_dir / "training-loss.json", loss)

    runtime = read_json(run_dir / "env" / "runtime.json")
    smi = (run_dir / "env" / "nvidia-smi.csv").read_text().strip().splitlines()
    header = [h.strip() for h in smi[0].split(",")]
    gpus = [dict(zip(header, (c.strip() for c in line.split(",")))) for line in smi[1:]]
    sources = {}
    for line in (run_dir / "env" / "source-sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sources[name] = digest
    write_json(
        output_dir / "environment.json",
        {
            "runtime": {k: v for k, v in runtime.items() if k not in ("cpu_count",)},
            "cpu_threads": runtime["cpu_count"],
            "gpus": [
                {k: g[k] for k in header if not k.startswith("pci")} for g in gpus
            ],
            "placement": {"seed42": "GPU0", "seed43": "GPU1", "concurrent": True},
            "source_sha256": sources,
            "git_head": (run_dir / "env" / "git-head.txt").read_text().strip(),
            "git_status_porcelain_bytes": len(
                (run_dir / "env" / "git-status.txt").read_text()
            ),
            "launcher": {
                "file": "tools/train_with_loss_log.py (local)",
                "sha256": (run_dir / "env" / "preregistration.sha256")
                .read_text()
                .splitlines()[1]
                .split()[0],
                "behavior": "calls scripts.train_query_encoder.main unchanged; also appends each progress.json write to progress-log.jsonl",
            },
        },
    )
    return sorted(p.name for p in output_dir.iterdir())


# -------------------------------------------------------------------- analyze
def paired_bootstrap(differences, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    import numpy as np

    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(values.mean()),
        "ci95_percentile": [float(low), float(high)],
        "bootstrap_standard_error": float(means.std(ddof=1)),
        "fraction_resamples_le_zero": float((means <= 0).mean()),
        "wins_ties_losses": [
            int((values > 1e-12).sum()),
            int((abs(values) <= 1e-12).sum()),
            int((values < -1e-12).sum()),
        ],
        "queries": int(len(values)),
    }


def _intervals(alphas, mask):
    runs, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(mask) - 1):
            end = i if flag else i - 1
            runs.append([alphas[start], alphas[end]])
            start = None
    return runs


def analyze(evidence_dir):
    import numpy as np

    evidence_dir = Path(evidence_dir)
    rows = per_query(evidence_dir / "per-query.jsonl")
    reference_rows = per_query(REFERENCE_RUN / "per-query.jsonl")
    manifest = read_json(evidence_dir / "per-query-manifest.json")
    order = list(next(iter(rows.values())).keys())

    def diff(table, method, base="baseline + Fixed alpha=0.84"):
        return [table[method][q]["ndcg@10"] - table[base][q]["ndcg@10"] for q in order]

    bootstrap = {
        "kind": "post-hoc paired bootstrap over the 721 frozen Test queries; uncertainty only, not a new held-out experiment",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "statistic": "mean per-query nDCG@10(encoder) - nDCG@10(matched BF16 baseline, alpha .84)",
        "primary": {
            "comparison": "reproduced seed43 alpha .84 vs matched baseline alpha .84",
            **paired_bootstrap(diff(rows, "seed43 + Fixed alpha=0.84")),
        },
        "secondary": {
            "reproduced seed43 validation-selected alpha .89": paired_bootstrap(
                diff(rows, "seed43 + Fixed")
            ),
            "reproduced seed42 alpha .84": paired_bootstrap(
                diff(rows, "seed42 + Fixed alpha=0.84")
            ),
            "reproduced seed42 validation-selected alpha .88": paired_bootstrap(
                diff(rows, "seed42 + Fixed")
            ),
            "PR #40 reference seed43 alpha .84 (same method on committed reference rows)": paired_bootstrap(
                diff(reference_rows, "seed43 + Fixed alpha=0.84")
            ),
            "reproduced minus reference seed43 alpha .84 (cross-run, same queries)": paired_bootstrap(
                [
                    rows["seed43 + Fixed alpha=0.84"][q]["ndcg@10"]
                    - reference_rows["seed43 + Fixed alpha=0.84"][q]["ndcg@10"]
                    for q in order
                ]
            ),
        },
    }

    curves = read_json(evidence_dir / "validation-alpha-curves.json")
    frozen = read_json(evidence_dir / "frozen.json")
    reference_frozen = read_json(REFERENCE_RUN / "frozen.json")
    reference_alpha = {
        "baseline": reference_frozen["models"]["baseline"]["alpha_dense"],
        "seed42": SEED42_REFERENCE["validation_alpha_dense"],
        "seed43": reference_frozen["models"]["seed43"]["alpha_dense"],
    }
    alphas = curves["alphas"]
    stability = {
        "kind": "Validation-only sensitivity of the frozen alpha choice; selections are unchanged",
        "tolerances": list(TOLERANCES),
        "encoders": {},
    }
    for name, mean in curves["mean_ndcg"].items():
        mean = np.asarray(mean)
        best = float(mean.max())
        selected = frozen["models"][name]["alpha_dense"]
        ref = round(reference_alpha[name], 2)
        paired = curves["per_query_ndcg_at_selected_reference_and_shared_alpha"][name]
        sel_values = np.asarray(paired[f"{selected:.2f}"])
        entry = {
            "selected_alpha": round(selected, 2),
            "best_mean_ndcg": best,
            "reference_alpha": ref,
            "mean_at_reference_alpha": float(mean[int(round(ref * 100))]),
            "gap_reference_alpha_to_best": best - float(mean[int(round(ref * 100))]),
            "near_optimal": {},
        }
        for tol in TOLERANCES:
            mask = mean >= best - tol
            entry["near_optimal"][f"{tol:g}"] = {
                "count": int(mask.sum()),
                "intervals": _intervals(alphas, mask),
            }
        if f"{ref:.2f}" in paired and ref != round(selected, 2):
            entry["validation_paired_bootstrap_selected_minus_reference_alpha"] = (
                paired_bootstrap(sel_values - np.asarray(paired[f"{ref:.2f}"]))
            )
        stability["encoders"][name] = entry

    protocol = read_json(evidence_dir / "protocol.json")
    reference_protocol = read_json(REFERENCE_RUN / "protocol.json")
    references = {
        "seed42": SEED42_REFERENCE["history"],
        "seed43": {
            h["step"]: h["validation_ndcg"]
            for h in reference_protocol["encoders"]["seed43"]["history"]
        },
    }
    training = {}
    for name in ("seed42", "seed43"):
        encoder = protocol["encoders"][name]
        history = {h["step"]: h["validation_ndcg"] for h in encoder["history"]}
        deltas = {s: history[s] - references[name][s] for s in STEPS}
        reference_step = (
            SEED42_REFERENCE["selected_step"]
            if name == "seed42"
            else reference_protocol["encoders"]["seed43"]["selected_step"]
        )
        training[name] = {
            "selected_step": encoder["selected_step"],
            "reference_selected_step": reference_step,
            "checkpoint_sha256": encoder["checkpoint_sha256"],
            "reference_checkpoint_sha256": (
                SEED42_REFERENCE["checkpoint_sha256"]
                if name == "seed42"
                else reference_protocol["encoders"]["seed43"]["checkpoint_sha256"]
            ),
            "validation_dense_ndcg": {str(s): history[s] for s in STEPS},
            "reference_validation_dense_ndcg": {
                str(s): references[name][s] for s in STEPS
            },
            "delta": {str(s): deltas[s] for s in STEPS},
            "max_abs_delta": max(abs(v) for v in deltas.values()),
            "margin_selected_over_runner_up": history[encoder["selected_step"]]
            - max(v for s, v in history.items() if s != encoder["selected_step"]),
            "reference_source": (
                SEED42_REFERENCE["source"]
                if name == "seed42"
                else f"{REFERENCE_RUN_REL}/protocol.json"
            ),
        }

    aggregates = manifest["aggregates"]
    reference_aggregates = read_json(REFERENCE_RUN / "per-query-manifest.json")[
        "aggregates"
    ]
    test = {
        "methods": aggregates,
        "differences_vs_pr40": {
            method: {
                k: aggregates[method][k] - reference_aggregates[method][k]
                for k in aggregates[method]
            }
            for method in aggregates
            if method in reference_aggregates
        },
        "note": "'seed43 + Fixed' is Validation-selected in both runs, at alpha .89 (new) vs .83 (PR #40)",
    }
    return {
        "schema_version": 1,
        "inputs_sha256": {
            name: sha256(evidence_dir / name)
            for name in (*COMPACT, "validation-alpha-curves.json")
        },
        "reference_inputs_sha256": {
            name: sha256(REFERENCE_RUN / name)
            for name in (
                "protocol.json",
                "frozen.json",
                "per-query.jsonl",
                "per-query-manifest.json",
            )
        },
        "training": training,
        "alpha_stability": stability,
        "bootstrap": bootstrap,
        "test": test,
    }


# -------------------------------------------------------------------- figures
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def _svg(width, height, body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}" '
        f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">\n'
        f"<title>{title}</title>\n"
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _panel(x0, y0, w, h, xs, xlabels, ylo, yhi, yticks, series, title, xname):
    """Draw one line panel. series: list of (label, color, xs, ys, dash, marks)."""
    body = []
    sx = lambda v: x0 + (v - xs[0]) / (xs[-1] - xs[0]) * w  # noqa: E731
    sy = lambda v: y0 + h - (v - ylo) / (yhi - ylo) * h  # noqa: E731
    body.append(
        f'<text x="{x0}" y="{y0 - 14}" font-size="13" font-weight="600" fill="{INK}">{title}</text>'
    )
    for t in yticks:
        body.append(
            f'<line x1="{x0}" x2="{x0 + w}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{x0 - 6}" y="{sy(t) + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{t:.3f}</text>'
        )
    for v, label in zip(xs, xlabels):
        if label:
            body.append(
                f'<text x="{sx(v):.1f}" y="{y0 + h + 16}" font-size="11" text-anchor="middle" fill="{MUTED}">{label}</text>'
            )
    body.append(
        f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="{MUTED}" stroke-width="1"/>'
    )
    body.append(
        f'<text x="{x0 + w / 2}" y="{y0 + h + 34}" font-size="11" text-anchor="middle" fill="{MUTED}">{xname}</text>'
    )
    clip = f"clip{int(x0)}"
    body.append(
        f'<clipPath id="{clip}"><rect x="{x0}" y="{y0 - 8}" width="{w}" height="{h + 8}"/></clipPath>'
    )
    for label, color, px, py, dash, marks in series:
        points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(px, py))
        dash_attr = ' stroke-dasharray="6 4"' if dash else ""
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash_attr} stroke-linejoin="round" clip-path="url(#{clip})"/>'
        )
        for a, b, kind, tip in marks:
            fill = color if kind == "filled" else SURFACE
            body.append(
                f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="5" fill="{fill}" stroke="{color}" stroke-width="2">'
                f"<title>{tip}</title></circle>"
            )
    return body, sx, sy


def checkpoint_figure(analysis):
    width, height = 860, 360
    body = [
        f'<text x="24" y="28" font-size="15" font-weight="600" fill="{INK}">Validation Dense nDCG@10 by checkpoint (745 primary queries)</text>',
        f'<text x="24" y="46" font-size="12" fill="{MUTED}">Filled marker = selected checkpoint. Steps are evenly spaced on the axis.</text>',
    ]
    positions = list(range(len(STEPS)))
    for index, name in enumerate(("seed42", "seed43")):
        t = analysis["training"][name]
        new = [t["validation_dense_ndcg"][str(s)] for s in STEPS]
        ref = [t["reference_validation_dense_ndcg"][str(s)] for s in STEPS]
        sel_new = STEPS.index(t["selected_step"])
        sel_ref = STEPS.index(t["reference_selected_step"])
        series = [
            (
                "Reproduction",
                SERIES[0],
                positions,
                new,
                False,
                [
                    (
                        p,
                        v,
                        "filled" if p == sel_new else "hollow",
                        f"Reproduction step {STEPS[p]}: {v:.6f}",
                    )
                    for p, v in enumerate(new)
                ],
            ),
            (
                "Reference",
                SERIES[1],
                positions,
                ref,
                True,
                [
                    (
                        p,
                        v,
                        "filled" if p == sel_ref else "hollow",
                        f"Reference step {STEPS[p]}: {v:.6f}",
                    )
                    for p, v in enumerate(ref)
                ],
            ),
        ]
        x0 = 76 + index * 400
        ref_name = "c95c3fb" if name == "seed42" else "PR #40"
        panel, sx, sy = _panel(
            x0,
            90,
            310,
            200,
            positions,
            [str(s) for s in STEPS],
            0.784,
            0.826,
            [0.79, 0.80, 0.81, 0.82],
            series,
            f"{name.capitalize()} (selected {t['selected_step']} vs reference {t['reference_selected_step']})",
            "training step",
        )
        body += panel
        y_new, y_ref = sy(new[-1]) + 4, sy(ref[-1]) + 4
        if abs(y_new - y_ref) < 14:
            # Separate collided end labels; the higher value keeps the upper slot.
            mid = (y_new + y_ref) / 2
            y_new, y_ref = (
                (mid + 8, mid - 8) if new[-1] < ref[-1] else (mid - 8, mid + 8)
            )
        body.append(
            f'<text x="{sx(positions[-1]) + 10:.1f}" y="{y_new:.1f}" font-size="11" fill="{INK}">Repro</text>'
        )
        body.append(
            f'<text x="{sx(positions[-1]) + 10:.1f}" y="{y_ref:.1f}" font-size="11" fill="{INK}">{ref_name}</text>'
        )
    body.append(
        f'<line x1="24" x2="44" y1="340" y2="340" stroke="{SERIES[0]}" stroke-width="2"/><text x="50" y="344" font-size="11" fill="{INK}">Repro: reproduction (Blackwell, original cache)</text>'
    )
    body.append(
        f'<line x1="340" x2="360" y1="340" y2="340" stroke="{SERIES[1]}" stroke-width="2" stroke-dasharray="6 4"/><text x="366" y="344" font-size="11" fill="{INK}">Reference (Seed42: c95c3fb; Seed43: PR #40)</text>'
    )
    return _svg(
        width, height, body, "Validation checkpoint curves, reproduction vs reference"
    )


def alpha_figure(analysis, curves):
    width, height = 900, 380
    alphas = curves["alphas"]
    import math

    lo = alphas.index(0.7)
    xs = alphas[lo:]
    body = [
        f'<text x="24" y="28" font-size="15" font-weight="600" fill="{INK}">Validation Fixed-fusion nDCG@10 by Dense weight α (745 primary queries, α from 0.70)</text>',
        f'<text x="24" y="46" font-size="12" fill="{MUTED}">Shaded: α within 1e-3 of the best. Filled marker = selected α; hollow = PR #40/c95c3fb α. Each panel has its own y-range.</text>',
    ]
    titles = {"baseline": "Original BGE", "seed42": "Seed42", "seed43": "Seed43"}
    for index, name in enumerate(("baseline", "seed42", "seed43")):
        mean = curves["mean_ndcg"][name][lo:]
        s = analysis["alpha_stability"]["encoders"][name]
        ylo = math.floor(min(mean) / 0.005) * 0.005
        yhi = math.ceil(max(mean) / 0.005) * 0.005
        ticks = [
            round(ylo + 0.005 * k, 3)
            for k in range(int(round((yhi - ylo) / 0.005)) + 1)
        ]
        sel, ref = s["selected_alpha"], s["reference_alpha"]
        marks = [
            (
                sel,
                mean[xs.index(sel)],
                "filled",
                f"selected α {sel:.2f}: {mean[xs.index(sel)]:.6f}",
            )
        ]
        if ref != sel:
            marks.append(
                (
                    ref,
                    mean[xs.index(ref)],
                    "hollow",
                    f"reference α {ref:.2f}: {mean[xs.index(ref)]:.6f}",
                )
            )
        x0 = 70 + index * 285
        panel, sx, sy = _panel(
            x0,
            90,
            220,
            220,
            xs,
            [f"{a:.1f}" if round(a * 10) == a * 10 else "" for a in xs],
            ylo,
            yhi,
            ticks,
            [(name, SERIES[index], xs, mean, False, marks)],
            f"{titles[name]} (selected {sel:.2f}, ref {ref:.2f})",
            "Dense weight α",
        )
        for a0, a1 in s["near_optimal"]["0.001"]["intervals"]:
            body.append(
                f'<rect x="{sx(max(a0, xs[0])) - 2:.1f}" y="90" width="{sx(a1) - sx(max(a0, xs[0])) + 4:.1f}" height="220" fill="{GRID}" opacity="0.7"/>'
            )
        body += panel
    return _svg(width, height, body, "Validation alpha stability per encoder")


def write_analysis(evidence_dir):
    evidence_dir = Path(evidence_dir)
    result = analyze(evidence_dir)
    write_json(evidence_dir / "analysis.json", result)
    curves = read_json(evidence_dir / "validation-alpha-curves.json")
    (evidence_dir / "checkpoint-curves.svg").write_text(checkpoint_figure(result))
    (evidence_dir / "alpha-stability.svg").write_text(alpha_figure(result, curves))
    write_manifest(evidence_dir)
    return result


def write_manifest(evidence_dir):
    evidence_dir = Path(evidence_dir)
    lines = [
        f"{sha256(path)}  {path.name}"
        for path in sorted(evidence_dir.iterdir())
        if path.is_file() and path.name != "MANIFEST.sha256"
    ]
    (evidence_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def check(evidence_dir):
    evidence_dir = Path(evidence_dir)
    for line in (evidence_dir / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256(evidence_dir / name) != digest:
            raise ValueError(f"manifest mismatch: {name}")
    listed = {
        line.split("  ", 1)[1]
        for line in (evidence_dir / "MANIFEST.sha256").read_text().splitlines()
    }
    present = {p.name for p in evidence_dir.iterdir() if p.is_file()} - {
        "MANIFEST.sha256"
    }
    if listed != present:
        raise ValueError(f"unlisted or missing files: {sorted(listed ^ present)}")
    recomputed = json.loads(json.dumps(analyze(evidence_dir)))
    if recomputed != read_json(evidence_dir / "analysis.json"):
        raise ValueError("analysis.json does not match recomputation")
    curves = read_json(evidence_dir / "validation-alpha-curves.json")
    if (evidence_dir / "checkpoint-curves.svg").read_text() != checkpoint_figure(
        recomputed
    ):
        raise ValueError("checkpoint figure differs")
    if (evidence_dir / "alpha-stability.svg").read_text() != alpha_figure(
        recomputed, curves
    ):
        raise ValueError("alpha figure differs")
    return {"verified": True, "files": len(listed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export-evidence")
    exp.add_argument("--run-dir", type=Path, required=True)
    exp.add_argument("--output-dir", type=Path, default=EVIDENCE)
    for name in ("analyze", "check"):
        sub.add_parser(name).add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args(argv)
    if args.command == "export-evidence":
        result = export_evidence(args.run_dir, args.output_dir)
    elif args.command == "analyze":
        result = write_analysis(args.evidence_dir)
        result = {
            "bootstrap_primary": result["bootstrap"]["primary"],
            "alpha_stability": {
                k: {
                    kk: v[kk]
                    for kk in ("selected_alpha", "reference_alpha", "near_optimal")
                }
                for k, v in result["alpha_stability"]["encoders"].items()
            },
        }
    else:
        result = check(args.evidence_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
