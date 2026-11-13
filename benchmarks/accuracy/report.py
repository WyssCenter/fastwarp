# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""Summary tables (CSV and Markdown) and an optional plot from the evaluation CSVs and the
timing records.

Every paired comparison uses the pairs where both engines finished: the per-pair value of each
engine (for example the mean Dice over a structure group), the difference FastWarp minus ANTs
with a 95 % percentile bootstrap CI of its mean, and the Wilcoxon signed-rank p-value. The
timing tables give the median and quartiles over pairs of the registration, linear-fit and
post-processing times of both engines, with the machine and software that produced them.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import linear, post, timing
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
    columns: list[str] = []
    for r in rows:
        columns += [k for k in r if k not in columns]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
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


#: Order of the linear fits and post-processing steps in the tables.
LINEAR_FITS = ("prepare", "rigid", "affine", "multistart")
POST_STEPS = ("forward_field", "inverse_field", "lanczos_warp")
_FIT_NAMES = {
    "prepare": "session preparation",
    "rigid": "rigid",
    "affine": "affine (from the rigid fit)",
    "multistart": "rigid multi-start",
}
_STEP_NAMES = {
    "forward_field": "forward field (compose)",
    "inverse_field": "inverse field (compose)",
    "lanczos_warp": "Lanczos re-warp",
}


def _summaries(rows: list[dict[str, Any]], keys: tuple[str, ...], fields: tuple[str, ...]):
    return {tuple(r[k] for k in keys): r for r in timing.summary_rows(rows, keys, fields)}


def _cell(stats: dict[str, Any], rows: int, digits: int = 1) -> str:
    text = timing.spread(stats, digits)
    if stats.get("n") and stats["n"] != rows:
        text += f" [n={stats['n']}]"
    return text


def paired_ratios(rows: list[dict[str, Any]], dataset: str, field: str) -> dict[str, Any]:
    """:func:`timing.describe` of ANTs / FastWarp per pair of ``dataset``, over the pairs where
    both engines have ``field``."""
    by: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["dataset"] == dataset:
            by[r["pair"]][r["engine"]] = _float(r.get(field))
    return timing.describe(
        timing.ratio(v["ants"], v["fastwarp"]) for v in by.values() if len(v) == 2
    )


def registration_timing_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per dataset and engine: registrations and ``median (q1–q3)`` of the linear, SyN, engine
    total and wall seconds (``[n=k]`` when only ``k`` registrations have the value), and the
    median over pairs of ANTs / FastWarp, taken on the pairs where both engines have it."""
    fields = ("linear_s", "syn_s", "total_s", "wall_s")
    s = _summaries(rows, ("dataset", "engine"), fields)
    out = []
    for dataset in sorted({k[0] for k in s}):
        for engine in ("fastwarp", "ants"):
            r = s.get((dataset, engine))
            if r is None:
                continue
            row: dict[str, Any] = {"dataset": dataset, "engine": engine, "n": r["rows"]}
            for f in fields:
                row[f] = _cell(timing.stats_of(r, f), r["rows"])
            out.append(row)
        if s.get((dataset, "fastwarp")) and s.get((dataset, "ants")):
            row = {"dataset": dataset, "engine": "ANTs / FastWarp, per pair", "n": ""}
            for f in fields:
                d = paired_ratios(rows, dataset, f)
                row[f] = "n/a" if not d["n"] else f"{d['median']:.1f}x ({d['n']} pairs)"
            out.append(row)
    return out


def _ratio_text(a: Any, b: Any) -> str:
    r = timing.ratio(_float(a), _float(b))
    return "n/a" if math.isnan(r) else f"{r:.1f}x"


def linear_timing_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per dataset and fit: FastWarp and ANTs seconds (``median (q1–q3)``), their ratio, each
    engine's median landmark error and the median distance between the engines' fits."""
    fields = ("seconds", "landmark_mean_mm", "agreement_mean_mm", "agreement_max_mm")
    s = _summaries(rows, ("dataset", "fit", "engine"), fields)
    out = []
    for dataset in sorted({k[0] for k in s}):
        for fit in LINEAR_FITS:
            fw, an = s.get((dataset, fit, "fastwarp")), s.get((dataset, fit, "ants"))
            if fw is None and an is None:
                continue

            def cell(r: dict[str, Any] | None, f: str, digits: int = 1) -> str:
                return "n/a" if r is None else timing.spread(timing.stats_of(r, f), digits)

            out.append(
                {
                    "dataset": dataset,
                    "fit": _FIT_NAMES[fit],
                    "n": max(r["rows"] for r in (fw, an) if r is not None),
                    "fastwarp_s": cell(fw, "seconds", 2),
                    "ants_s": cell(an, "seconds", 2),
                    "ratio": _ratio_text(
                        an and an.get("seconds_median"), fw and fw.get("seconds_median")
                    ),
                    "landmark_fastwarp": cell(fw, "landmark_mean_mm", 2),
                    "landmark_ants": cell(an, "landmark_mean_mm", 2),
                    "agreement_mean": cell(fw, "agreement_mean_mm", 2),
                    "agreement_max": cell(fw, "agreement_max_mm", 2),
                }
            )
    return out


def post_timing_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per dataset and step: voxels, ANTs and FastWarp seconds (``median (q1–q3)``), their
    ratio and the median difference between the two outputs."""
    fields = ("voxels", "fastwarp_s", "ants_s", "field_diff_max_mm", "image_diff_rms_relative")
    s = _summaries(rows, ("dataset", "step"), fields)
    out = []
    for dataset in sorted({k[0] for k in s}):
        for step in POST_STEPS:
            r = s.get((dataset, step))
            if r is None:
                continue
            if step == "lanczos_warp":
                diff = _fmt(_float(r.get("image_diff_rms_relative_median")))
                diff = f"relative RMS {diff}"
            else:
                diff = f"max {_fmt(_float(r.get('field_diff_max_mm_median')))} mm"
            voxels = _float(r.get("voxels_median"))
            out.append(
                {
                    "dataset": dataset,
                    "step": _STEP_NAMES[step],
                    "n": r["rows"],
                    "voxels": "n/a" if math.isnan(voxels) else f"{voxels / 1e6:.1f} M",
                    "ants_s": timing.spread(timing.stats_of(r, "ants_s"), 2),
                    "fastwarp_s": timing.spread(timing.stats_of(r, "fastwarp_s"), 2),
                    "ratio": _ratio_text(r.get("ants_s_median"), r.get("fastwarp_s_median")),
                    "difference": diff,
                }
            )
    return out


def machine_lines(env: dict[str, Any]) -> list[str]:
    """The hardware and software of an ``environment`` record, as Markdown list items."""
    gpu = env.get("gpu") or {}
    fw_env = env.get("fastwarp_env") or {}
    return [
        f"- CPU: {env.get('cpu')}, {env.get('cpu_count')} logical cores; ITK threads (ANTs) "
        f"{env.get('itk_threads')}; RAYON_NUM_THREADS {env.get('rayon_threads') or 'not set'}",
        f"- GPU: {gpu.get('name', gpu.get('error'))}, {gpu.get('device_type')}, "
        f"{gpu.get('backend')}, driver {gpu.get('driver')}",
        f"- OS: {env.get('platform')}; Python {env.get('python')}, NumPy {env.get('numpy')}, "
        f"SciPy {env.get('scipy')}",
        f"- fastwarp {env.get('fastwarp')} (engine {env.get('fastwarp_engine_source')}); "
        f"ANTs {env.get('ants_version')} ({env.get('ants_program')}); ANTsPy "
        f"{env.get('antspyx')}"
        + ("; " + ", ".join(f"{k}={v}" for k, v in sorted(fw_env.items())) if fw_env else ""),
        f"- benchmark commit {env.get('benchmark_commit')}"
        + (" with local changes" if env.get("benchmark_tree_dirty") else "")
        + f", recorded {env.get('recorded')}",
    ]


def machine_table(fingerprints: list[tuple[dict[str, Any], int]]) -> list[dict[str, Any]]:
    """One row per distinct machine fingerprint with the number of records."""
    return [{**fp, "records": n} for fp, n in fingerprints]


def _timing_sections(reg: list, lin: list, pst: list, records: list) -> list[str]:
    lines: list[str] = []
    if reg:
        lines += [
            "",
            "## Registration time",
            "",
            "Seconds per registration, median (quartiles). Linear: the rigid and affine stages; "
            "SyN: the SyN stage; total: the engine's own total (FastWarp: GPU set-up, "
            "resampling, both stages and writing the transforms; ANTs: `Total elapsed time`, "
            "the stages and their set-up without reading the images or writing the "
            "transforms), so the totals do not cover the same work; wall: FastWarp's "
            "`register_syn` call (images already read), ANTs' whole process (start, reading, "
            "writing; with ANTsPy also starting Python). ANTs' split comes from its "
            "`--verbose 1` log. The ratio row is the median over pairs of ANTs / FastWarp, on "
            "the pairs where both engines have the value.",
            "",
            _md_table(
                registration_timing_table(reg),
                [
                    ("dataset", "Dataset"),
                    ("engine", "Engine"),
                    ("n", "Registrations"),
                    ("linear_s", "Linear (s)"),
                    ("syn_s", "SyN (s)"),
                    ("total_s", "Total (s)"),
                    ("wall_s", "Wall (s)"),
                ],
            ),
        ]
    if lin:
        lines += [
            "",
            "## Linear fits",
            "",
            "Seconds per fit, median (quartiles). FastWarp: one `linear_session` per pair "
            "(preparation once, its own row; the fits exclude it), rigid from the centres of "
            "mass, affine from the rigid result, then a rigid multi-start; ANTs: one "
            "`antsRegistration` stage time each, which includes the stage's own image "
            "preparation (no multi-start). ANTs' threads are capped, FastWarp's CPU work is "
            "not. Landmark error: mean over the landmarks of a pair, mm; "
            "FastWarp - ANTs: mean and max distance between the two fits over the fixed brain "
            "mask, mm (the multi-start against ANTs' rigid fit).",
            "",
            _md_table(
                linear_timing_table(lin),
                [
                    ("dataset", "Dataset"),
                    ("fit", "Fit"),
                    ("n", "Pairs"),
                    ("fastwarp_s", "FastWarp (s)"),
                    ("ants_s", "ANTs (s)"),
                    ("ratio", "ANTs / FastWarp"),
                    ("landmark_fastwarp", "Landmarks FastWarp (mm)"),
                    ("landmark_ants", "Landmarks ANTs (mm)"),
                    ("agreement_mean", "FastWarp - ANTs mean (mm)"),
                    ("agreement_max", "FastWarp - ANTs max (mm)"),
                ],
            ),
        ]
    if pst:
        lines += [
            "",
            "## Post-processing",
            "",
            "Seconds per step, median (quartiles), each including reading the inputs and "
            "writing the `.nii.gz` output; ANTs: `antsApplyTransforms --float 1`. Difference: "
            "median over pairs of the max field difference (mm) or of the relative RMS "
            "difference of the re-warped images.",
            "",
            _md_table(
                post_timing_table(pst),
                [
                    ("dataset", "Dataset"),
                    ("step", "Step"),
                    ("n", "Pairs"),
                    ("voxels", "Voxels"),
                    ("ants_s", "ANTs (s)"),
                    ("fastwarp_s", "FastWarp (s)"),
                    ("ratio", "ANTs / FastWarp"),
                    ("difference", "Difference"),
                ],
            ),
        ]
    machines = timing.machines(records)
    if len(machines) > 1:
        lines += [
            "",
            "## Machines",
            "",
            "The timing records come from more than one machine or software version:",
            "",
            _md_table(
                machine_table(machines),
                [
                    ("records", "Records"),
                    ("gpu", "GPU"),
                    ("gpu_driver", "Driver"),
                    ("gpu_backend", "Backend"),
                    ("cpu", "CPU"),
                    ("itk_threads", "ITK threads"),
                    ("fastwarp", "fastwarp"),
                    ("ants", "ANTs"),
                ],
            ),
        ]
    return lines


def _timing_records(work: Path, recipe_name: str) -> list[dict[str, Any]]:
    paths = [
        *work.glob(f"*/{recipe_name}/*/*/run.json"),
        *work.glob(f"*/{linear.FOLDER}/*/*/run.json"),
        *work.glob(f"*/{recipe_name}/*/post.json"),
    ]
    return [rec for _, rec in timing.read_records(paths) if rec.get("status") == "ok"]


def report(*, work: Path, recipe_name: str, plot: bool = True) -> Path:
    """Write ``summary.md`` with the accuracy tables (when ``evaluate`` has run) and the timing
    tables, and the CSVs behind them (``paired.csv``, ``agreement_summary.csv``,
    ``structures.csv``, ``timings.csv``, ``linear.csv``, ``post.csv``; with matplotlib
    ``summary.png``) under ``<work>/results/<recipe>/``; returns the Markdown path."""
    folder = work / "results" / recipe_name
    pairs = _read_csv(folder / "pairs.csv")
    reg = timing.registration_rows(work, recipe_name)
    lin = linear.rows(work, recipe_name)
    pst = post.rows(work, recipe_name)
    if not (pairs or reg or lin or pst):
        raise RuntimeError(f"nothing to report in {work} for {recipe_name}; run (and evaluate)")
    folder.mkdir(parents=True, exist_ok=True)
    env_path = work / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
    source = "the latest `run` (`environment.json`)"
    if not env:
        records = _timing_records(work, recipe_name)
        env = next((r["environment"] for r in records if r.get("environment")), {})
        source = "the first timing record"
    lines = [
        f"# FastWarp benchmark: recipe `{recipe_name}`",
        "",
        f"Machine and software of {source}:",
        "",
        *machine_lines(env),
    ]
    if pairs:
        paired = paired_table(pairs)
        agreement = agreement_table(_read_csv(folder / "agreement.csv"))
        structures = per_structure_table(_read_csv(folder / "labels.csv"))
        _write_csv(folder / "paired.csv", paired)
        _write_csv(folder / "agreement_summary.csv", agreement)
        _write_csv(folder / "structures.csv", structures)
        lines += [
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
    _write_csv(folder / "timings.csv", reg)
    _write_csv(folder / "linear.csv", lin)
    _write_csv(folder / "post.csv", pst)
    lines += _timing_sections(reg, lin, pst, _timing_records(work, recipe_name))
    if pairs and plot and _plot(pairs, folder / "summary.png"):
        lines += ["", "![per-pair results](summary.png)"]
    md = folder / "summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md
