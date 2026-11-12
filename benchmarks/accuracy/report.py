# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""Summary tables (CSV and Markdown) and an optional plot from the evaluation CSVs.

Every paired comparison uses the pairs where both engines finished: the per-pair value of each
engine (for example the mean Dice over a structure group), the difference FastWarp minus ANTs
with a 95 % percentile bootstrap CI of its mean, and the Wilcoxon signed-rank p-value.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .stats import paired_summary

#: Per-pair metrics compared between the engines, if present, and which way is better.
METRIC_PREFIXES = {
    "dice_": "higher",
    "hd95_mm_": "lower",
    "assd_mm_": "lower",
    "landmark_mean_mm": "lower",
    "landmark_median_mm": "lower",
    "jac_nonpositive_frac": "lower",
    "wall_s": "lower",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _metrics(columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        for prefix in METRIC_PREFIXES:
            if c == prefix or (prefix.endswith("_") and c.startswith(prefix)):
                out.append(c)
                break
    return out


def paired_table(pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """One row per dataset and metric: :func:`paired_summary` of FastWarp against ANTs."""
    by: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    columns: list[str] = []
    for r in pairs:
        by[(r["dataset"], r["pair"])][r["engine"]] = r
        for c in r:
            if c not in columns:
                columns.append(c)
    rows = []
    for dataset in sorted({d for d, _ in by}):
        both = [v for (d, _), v in sorted(by.items()) if d == dataset and len(v) == 2]
        for metric in _metrics(columns):
            a = [_float(v["fastwarp"].get(metric)) for v in both]
            b = [_float(v["ants"].get(metric)) for v in both]
            if not any(not math.isnan(x) for x in a):
                continue
            s = paired_summary(a, b)
            better = next(v for k, v in METRIC_PREFIXES.items() if metric.startswith(k))
            rows.append({"dataset": dataset, "metric": metric, "better": better, **s})
    return rows


def agreement_table(agreement: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Mean, SD and max over pairs of each FastWarp-ANTs agreement column per dataset."""
    rows = []
    for dataset in sorted({r["dataset"] for r in agreement}):
        sel = [r for r in agreement if r["dataset"] == dataset]
        for col in sel[0]:
            if col in ("dataset", "recipe", "pair", "fixed", "moving"):
                continue
            v = np.array([_float(r.get(col)) for r in sel])
            v = v[~np.isnan(v)]
            if not v.size:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "metric": col,
                    "n": int(v.size),
                    "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if v.size > 1 else math.nan,
                    "max": float(v.max()),
                }
            )
    return rows


def per_structure_table(labels: list[dict[str, str]], metric: str = "dice") -> list[dict]:
    """Mean of ``metric`` per dataset, structure and engine over the pairs."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    for r in labels:
        acc[(r["dataset"], r["group"], int(r["label"]), r.get("name", ""), r["engine"])].append(
            _float(r[metric])
        )
    keys = sorted({k[:4] for k in acc})
    rows = []
    for k in keys:
        row = {"dataset": k[0], "group": k[1], "label": k[2], "name": k[3]}
        for engine in ("fastwarp", "ants"):
            v = np.array(acc.get((*k, engine), []))
            v = v[~np.isnan(v)]
            row[f"{metric}_{engine}_mean"] = float(v.mean()) if v.size else math.nan
            row[f"{metric}_{engine}_n"] = int(v.size)
        rows.append(row)
    return rows


def _fmt(v: Any, digits: int = 3) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    return str(v)


def _md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(t for _, t in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(r.get(k)) for k, _ in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _plot(pairs: list[dict[str, str]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    panels = []
    for dataset in sorted({r["dataset"] for r in pairs}):
        sel = [r for r in pairs if r["dataset"] == dataset]
        for metric in _metrics(list(sel[0])):
            if metric.startswith("dice_") or metric == "landmark_mean_mm":
                panels.append((dataset, metric, sel))
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6), squeeze=False)
    for ax, (dataset, metric, sel) in zip(axes[0], panels, strict=True):
        data = [
            [
                v
                for v in (_float(r.get(metric)) for r in sel if r["engine"] == e)
                if not math.isnan(v)
            ]
            for e in ("fastwarp", "ants")
        ]
        ax.boxplot(data)
        ax.set_xticks([1, 2], ["FastWarp", "ANTs"])
        ax.set_title(f"{dataset}\n{metric}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def report(*, work: Path, recipe_name: str, plot: bool = True) -> Path:
    """Write ``summary.md``, ``paired.csv``, ``agreement_summary.csv``,
    ``structures.csv`` (and ``summary.png``) next to the evaluation CSVs; returns the
    Markdown path."""
    folder = work / "results" / recipe_name
    pairs = _read_csv(folder / "pairs.csv")
    if not pairs:
        raise RuntimeError(f"no evaluated pairs in {folder}; run evaluate first")
    paired = paired_table(pairs)
    agreement = agreement_table(_read_csv(folder / "agreement.csv"))
    structures = per_structure_table(_read_csv(folder / "labels.csv"))
    _write_csv(folder / "paired.csv", paired)
    _write_csv(folder / "agreement_summary.csv", agreement)
    _write_csv(folder / "structures.csv", structures)
    env_path = work / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
    gpu = env.get("gpu") or {}
    lines = [
        f"# FastWarp accuracy benchmark: recipe `{recipe_name}`",
        "",
        f"- fastwarp {env.get('fastwarp')} (engine {env.get('fastwarp_engine_source')}), "
        f"ANTsPy {env.get('antspyx')}, ANTs program: {env.get('ants_program')}",
        f"- GPU: {gpu.get('name', gpu.get('error'))} ({gpu.get('backend')}); "
        f"CPU: {env.get('cpu')}, {env.get('cpu_count')} logical cores, "
        f"ITK threads {env.get('itk_threads')}",
        f"- {env.get('platform')}, Python {env.get('python')}, benchmark commit "
        f"{env.get('benchmark_commit')}, recorded {env.get('recorded')}",
        "",
        "## FastWarp against ANTs, paired over the same pairs",
        "",
        "Difference = FastWarp minus ANTs; CI = 95 % percentile bootstrap of the mean "
        "difference; p = Wilcoxon signed-rank (pairs share subjects, so the p-values and CIs "
        "treat dependent pairs as independent).",
        "",
        _md_table(
            paired,
            [
                ("dataset", "Dataset"),
                ("metric", "Metric"),
                ("better", "Better"),
                ("n", "Pairs"),
                ("mean_a", "FastWarp mean"),
                ("sd_a", "SD"),
                ("mean_b", "ANTs mean"),
                ("sd_b", "SD"),
                ("mean_diff", "Mean diff."),
                ("ci_low", "CI low"),
                ("ci_high", "CI high"),
                ("wilcoxon_p", "p"),
            ],
        ),
        "",
        "## Agreement between the two engines",
        "",
        "Field differences (mm) inside the fixed brain mask; Dice between the label maps the "
        "two engines warp.",
        "",
        _md_table(
            agreement,
            [
                ("dataset", "Dataset"),
                ("metric", "Metric"),
                ("n", "Pairs"),
                ("mean", "Mean"),
                ("sd", "SD"),
                ("max", "Max"),
            ],
        ),
    ]
    small = [r for r in structures if r["group"] != "cortical"]
    if small:
        lines += [
            "",
            "## Mean Dice per structure (non-cortical)",
            "",
            _md_table(
                small,
                [
                    ("dataset", "Dataset"),
                    ("label", "Label"),
                    ("name", "Structure"),
                    ("dice_fastwarp_mean", "FastWarp"),
                    ("dice_ants_mean", "ANTs"),
                    ("dice_fastwarp_n", "Pairs"),
                ],
            ),
        ]
    if plot and _plot(pairs, folder / "summary.png"):
        lines += ["", "![per-pair results](summary.png)"]
    md = folder / "summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md
