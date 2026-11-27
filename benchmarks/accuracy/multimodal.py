# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""The multimodal check on IXI, where no labels exist.

Each subject's T2w image, aligned rigidly to its T1w first, is registered onto the MNI 2009c
T1w template with Mattes mutual information in SyN (dataset ``ixi_t2``, recipe ``cohort-mi``),
and the same subject's T1w with CC (``ixi_t1``, recipe ``cohort``). Both warps take the template
into the subject's T1w space, so the difference of the two displacement fields inside the
template's brain mask is a proxy for the error of the MI warp. For each engine's MI warp it is
taken against the T1w warp of the same engine (``own``) and against ANTs' T1w warp (``ants``).
Each subject's result is cached in ``multimodal.json`` in its ``ixi_t2`` pair folder; the
tables go to ``<work>/results/multimodal/``.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import datasets, metrics, recipes
from .evaluate import _field
from .run import ENGINES, composite, engine_done, pair_id, select_pairs
from .stats import paired_summary

MOVING = ("ixi_t2", "cohort-mi")
REFERENCE = ("ixi_t1", "cohort")
FIELD_KEYS = ("field_diff_rms_mm", "field_diff_mean_mm", "field_diff_p95_mm", "field_diff_max_mm")


def _done(folder: Path, recipe_name: str, iso_mm: Any) -> list[str]:
    rec = recipes.recipe(recipe_name)
    return [e for e in ENGINES if engine_done(folder / e, rec, iso_mm)]


def compare_pair(moving_folder: Path, reference_folder: Path) -> dict[str, Any]:
    """The field differences of one subject: per engine of the MI warp in ``moving_folder``,
    against each engine's CC warp in ``reference_folder``, inside the template's brain mask."""
    import ants

    pair = json.loads((moving_folder / "pair.json").read_text(encoding="utf-8"))
    files = pair["fixed_prep"]["files"]
    ref = ants.image_read(files["reference"])
    mask = ants.image_read(files["mask"]).numpy() > 0
    iso_mm = pair.get("iso_mm")
    reference = {
        e: _field(ref, composite(reference_folder / e), reference_folder / e / "mm_")
        for e in _done(reference_folder, REFERENCE[1], iso_mm)
    }
    out: dict[str, Any] = {"pair": moving_folder.name, "moving": pair["moving"], "engines": {}}
    for e in _done(moving_folder, MOVING[1], iso_mm):
        field = _field(ref, composite(moving_folder / e), moving_folder / e / "mm_")
        out["engines"][e] = {
            r: metrics.field_difference(field, f, mask) for r, f in reference.items()
        }
    return out


def run_multimodal(
    *,
    work: Path,
    pairs: int | None = None,
    quick: bool = False,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    """Compare every selected subject whose MI and CC warps finished (cached unless a warp is
    newer or ``force``), then write the tables; returns the results folder."""
    moving_ds, reference_ds = datasets.get(MOVING[0]), datasets.get(REFERENCE[0])
    n = (
        pairs
        if pairs is not None
        else (moving_ds.quick_pairs if quick else moving_ds.default_pairs)
    )
    results = []
    for fixed_id, moving_id in select_pairs(moving_ds, n):
        pid = pair_id(fixed_id, moving_id)
        a = work / moving_ds.key / MOVING[1] / pid
        b = work / reference_ds.key / REFERENCE[1] / pid
        runs = [
            p / e / "run.json" for p in (a, b) for e in ENGINES if (p / e / "run.json").is_file()
        ]
        if not (a / "pair.json").is_file() or not runs:
            continue
        target = a / "multimodal.json"
        newest = max(r.stat().st_mtime for r in runs)
        if target.is_file() and target.stat().st_mtime > newest and not force:
            results.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        log(f"comparing {moving_ds.key}/{pid}")
        try:
            res = compare_pair(a, b)
        except Exception as e:
            log(f"    failed: {type(e).__name__}: {e}")
            continue
        target.write_text(json.dumps(res, indent=1), encoding="utf-8")
        results.append(res)
    return write_tables(work / "results" / "multimodal", results)


def rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per subject and engine of the MI warp: its field differences against each
    engine's CC warp (``vs_<engine>_*``) and against the same engine's (``vs_own_*``)."""
    out = []
    for res in results:
        for engine, by_ref in res["engines"].items():
            row: dict[str, Any] = {"pair": res["pair"], "moving": res["moving"], "engine": engine}
            for ref_engine, diff in by_ref.items():
                row.update({f"vs_{ref_engine}_{k}": v for k, v in diff.items()})
            row.update({f"vs_own_{k}": v for k, v in by_ref.get(engine, {}).items()})
            out.append(row)
    return out


def _both(table: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in table:
        by.setdefault(r["pair"], {})[r["engine"]] = r
    return [v for _, v in sorted(by.items()) if set(ENGINES) <= set(v)]


def paired(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """:func:`~.stats.paired_summary` of FastWarp against ANTs for each ``vs_own_*`` column (each
    engine's MI warp against its own CC warp), over the subjects both engines finished. Only these
    are a fair paired test: against one engine's CC warp, that engine's MI warp has the head start
    of sharing its implementation."""
    both = _both(table)
    out = []
    for c in (f"vs_own_{k}" for k in FIELD_KEYS):
        a = [float(v["fastwarp"].get(c, math.nan)) for v in both]
        b = [float(v["ants"].get(c, math.nan)) for v in both]
        if a and any(not math.isnan(x) for x in a):
            out.append({"metric": c, "better": "lower", **paired_summary(a, b)})
    return out


def descriptive(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and SD per engine of the ``vs_ants_*`` and ``vs_fastwarp_*`` columns, over the
    subjects both engines finished: descriptive only, no test (see :func:`paired`)."""
    both = _both(table)
    out = []
    for ref in ("ants", "fastwarp"):
        for k in FIELD_KEYS:
            c = f"vs_{ref}_{k}"
            row: dict[str, Any] = {"metric": c, "n": len(both)}
            for e in ENGINES:
                v = [float(x[e].get(c, math.nan)) for x in both]
                v = [x for x in v if not math.isnan(x)]
                row[f"{e}_mean"] = statistics.fmean(v) if v else math.nan
                row[f"{e}_sd"] = statistics.stdev(v) if len(v) > 1 else math.nan
            out.append(row)
    return out


def _write_csv(path: Path, table: list[dict[str, Any]]) -> None:
    columns: list[str] = list(dict.fromkeys(k for r in table for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(table)


def _cell(v: Any) -> str:
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.3g}"
    return str(v)


def write_tables(folder: Path, results: list[dict[str, Any]]) -> Path:
    """``multimodal.csv`` (one row per subject and engine), ``multimodal_paired.csv``,
    ``multimodal_descriptive.csv`` and ``summary.md`` in ``folder``; returns ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    table = rows(results)
    summary = paired(table)
    desc = descriptive(table)
    _write_csv(folder / "multimodal.csv", table)
    _write_csv(folder / "multimodal_paired.csv", summary)
    _write_csv(folder / "multimodal_descriptive.csv", desc)
    head = ("metric", "n", "mean_a", "sd_a", "mean_b", "sd_b", "mean_diff", "ci_low", "ci_high")
    names = ("Metric", "Subjects", "FastWarp", "SD", "ANTs", "SD", "Diff.", "CI low", "CI high")
    lines = [
        "# Multimodal check: IXI T2w with MI against the T1w with CC",
        "",
        "Field differences (mm) inside the template's brain mask between each engine's T2w MI "
        "warp and the same engine's T1w CC warp of the same subject. Diff. = FastWarp minus "
        "ANTs; CI = 95 % percentile bootstrap of the mean difference.",
        "",
        "| " + " | ".join(names) + " |",
        "|" + "---|" * len(names),
        *("| " + " | ".join(_cell(r.get(k)) for k in head) + " |" for r in summary),
        "",
        "Against one engine's T1w CC warp (descriptive only: that engine's own MI warp shares its "
        "implementation, so the comparison is not fair to the other):",
        "",
        "| Metric | Subjects | FastWarp mean | SD | ANTs mean | SD |",
        "|---|---|---|---|---|---|",
        *(
            "| "
            + " | ".join(
                _cell(r.get(k))
                for k in ("metric", "n", "fastwarp_mean", "fastwarp_sd", "ants_mean", "ants_sd")
            )
            + " |"
            for r in desc
        ),
    ]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
