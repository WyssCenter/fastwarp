# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""Post-processing of finished registrations: FastWarp's ``compose_field`` and
``apply_transforms`` against ``antsApplyTransforms`` on the same composite transforms.

Per pair, three steps, each run once by FastWarp (in this process) and once by
``antsApplyTransforms`` (a child process, one at a time, never at the same time as FastWarp):

- ``forward_field``: the forward composite as one displacement field on the fixed image's
  evaluation grid (``antsApplyTransforms --output [<file>,1]``);
- ``inverse_field``: the inverse composite as one displacement field on the moving image's own
  grid;
- ``lanczos_warp``: the moving image resampled onto the fixed evaluation grid through the forward
  composite with windowed-sinc (Lanczos) interpolation.

Each time covers reading the inputs and writing the output (``.nii.gz``), as the ANTs command
does. ANTs runs with ``--float 1``, so both write ``float32``. The composites are those of the
FastWarp registration of the pair (ANTs' when FastWarp has none), the same file for both tools.
FastWarp's CPU threads are capped by ``RAYON_NUM_THREADS``, ANTs' by
``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS``, both to ``--threads``. The outputs are compared
(field differences in mm, the warped images' relative RMS difference) and then deleted; only
``post.json`` stays in the pair folder.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from . import datasets, metrics, recipes, timing
from .run import (
    RunLock,
    _now,
    _read_json,
    _write_json,
    ants_tool,
    child_env,
    composite,
    engine_done,
    environment,
    inverse_composite,
    itk_threads,
    pair_id,
    run_child,
    select_pairs,
)

STEPS = ("forward_field", "inverse_field", "lanczos_warp")
INTERPOLATION = "lanczosWindowedSinc"
_MARK = "FASTWARP_BENCH_SECONDS"
ANTSPY_APPLY = (
    "import sys, time, ants; t = time.perf_counter(); "
    "r = ants.lib.antsApplyTransforms(sys.argv[1:]); "
    f"print('{_MARK}', repr(time.perf_counter() - t)); sys.exit(int(r or 0))"
)
_SECONDS = re.compile(rf"^{_MARK} (\S+)\s*$", re.MULTILINE)


def config(rec: dict[str, Any], iso_mm: float) -> dict[str, Any]:
    """What a post-processing record depends on: the registrations' recipe and grid, the
    interpolation and the output format."""
    return {
        "recipe": rec,
        "iso_mm": iso_mm,
        "interpolation": INTERPOLATION,
        "output": ".nii.gz",
        "ants_float": True,
    }


def source_stamp(folder: Path, engine: str) -> str | None:
    """When the registration whose composites are post-processed finished (its ``run.json``)."""
    return _read_json(folder / engine / "run.json").get("finished")


def post_done(folder: Path, cfg: dict[str, Any]) -> bool:
    """Whether a pair folder holds a successful ``post.json`` of ``cfg``, made from the
    registration that is there now (the same engine, finished at the same time)."""
    rec = _read_json(folder / "post.json")
    if rec.get("status") != "ok" or rec.get("config") != cfg:
        return False
    src = source_engine(folder, cfg["recipe"], cfg["iso_mm"])
    return src == rec.get("source_engine") and source_stamp(folder, src) == rec.get(
        "source_finished"
    )


def source_engine(folder: Path, rec: dict[str, Any], iso_mm: float) -> str | None:
    """The engine whose composites are post-processed: FastWarp's, else ANTs', else ``None``."""
    for engine in ("fastwarp", "ants"):
        if engine_done(folder / engine, rec, iso_mm):
            return engine
    return None


@contextmanager
def rayon_threads(threads: int) -> Iterator[None]:
    """Sets ``RAYON_NUM_THREADS`` and restores it afterwards. FastWarp's CPU thread pool reads it
    when the pool starts, so it takes effect only if nothing in this process used the pool
    before (true for the ``post`` command, a fresh process)."""
    old = os.environ.get("RAYON_NUM_THREADS")
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = old


def ants_args(
    step: str, *, reference: str, moving: str, forward: str, inverse: str, out: str
) -> list[str]:
    """The ``antsApplyTransforms`` arguments of one step."""
    paths = [recipes._check_path(p) for p in (reference, moving, forward, inverse, out)]
    reference, moving, forward, inverse, out = paths
    head = ["--dimensionality", "3", "--float", "1"]
    if step == "forward_field":
        body = ["--reference-image", reference, "--transform", forward, "--output", f"[{out},1]"]
    elif step == "inverse_field":
        body = ["--reference-image", moving, "--transform", inverse, "--output", f"[{out},1]"]
    elif step == "lanczos_warp":
        body = ["--input", moving, "--reference-image", reference, "--transform", forward]
        body += ["--interpolation", "LanczosWindowedSinc", "--output", out]
    else:
        raise ValueError(f"unknown step {step!r}")
    return [*head, *body]


def child_seconds(log_text: str) -> float | None:
    """The time the ANTsPy child measured around its ``antsApplyTransforms`` call."""
    m = _SECONDS.search(log_text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def run_fastwarp_steps(
    reference: Path, moving: Path, forward: Path, inverse: Path, tmp: Path
) -> dict[str, Any]:
    """FastWarp's three steps; returns ``{"steps": {step: {"seconds", "voxels"}}, "wall_s"}``."""
    import ants
    import fastwarp

    steps: dict[str, Any] = {}
    t0 = time.perf_counter()
    t = time.perf_counter()
    ref = ants.image_read(str(reference))
    fastwarp.compose_field([str(forward)], ref, tmp / "fastwarp_forward_field.nii.gz")
    steps["forward_field"] = {"seconds": time.perf_counter() - t, "voxels": int(np.prod(ref.shape))}
    t = time.perf_counter()
    mov = ants.image_read(str(moving))
    fastwarp.compose_field([str(inverse)], mov, tmp / "fastwarp_inverse_field.nii.gz")
    steps["inverse_field"] = {"seconds": time.perf_counter() - t, "voxels": int(np.prod(mov.shape))}
    t = time.perf_counter()
    ref = ants.image_read(str(reference))
    mov = ants.image_read(str(moving))
    warped = fastwarp.apply_transforms(mov, ref, [str(forward)], interpolation=INTERPOLATION)
    if not hasattr(warped, "new_image_like"):
        warped = ref.new_image_like(np.asarray(warped, dtype=np.float32))
    ants.image_write(warped.clone("float"), str(tmp / "fastwarp_lanczos_warp.nii.gz"))
    steps["lanczos_warp"] = {"seconds": time.perf_counter() - t, "voxels": int(np.prod(ref.shape))}
    return {"steps": steps, "wall_s": time.perf_counter() - t0}


def run_ants_steps(
    reference: Path,
    moving: Path,
    forward: Path,
    inverse: Path,
    tmp: Path,
    *,
    threads: int,
    ants_dir: Path | None,
    timeout_s: float | None,
) -> dict[str, Any]:
    """``antsApplyTransforms`` for the three steps, one process after the other. A step's
    ``seconds`` is the process wall time for an ANTs program, and the time around the library
    call for ANTsPy (whose process start and import are not part of the step)."""
    cmd = ants_tool(ants_dir, "antsApplyTransforms", ANTSPY_APPLY)
    env = child_env(threads)
    out: dict[str, Any] = {
        "program": "ANTsPy" if ants_dir is None else cmd[0],
        "itk_threads": threads,
        "steps": {},
    }
    t0 = time.perf_counter()
    for step in STEPS:
        args = ants_args(
            step,
            reference=reference.as_posix(),
            moving=moving.as_posix(),
            forward=forward.as_posix(),
            inverse=inverse.as_posix(),
            out=(tmp / f"ants_{step}.nii.gz").as_posix(),
        )
        log = tmp / f"ants_{step}.log"
        code, wall = run_child([*cmd, *args], log, env, timeout_s)
        seconds = wall
        entry: dict[str, Any] = {"wall_s": wall, "exit_code": code, "args": args}
        if ants_dir is None:
            seconds = child_seconds(log.read_text(encoding="utf-8", errors="replace"))
            if seconds is None:
                seconds = wall
                entry["timing_note"] = "no time from the ANTsPy child; process wall time used"
        out["steps"][step] = {"seconds": seconds, **entry}
        if code != 0 or not (tmp / f"ants_{step}.nii.gz").is_file():
            why = "timed out" if code is None else f"exited with {code}"
            text = log.read_text(encoding="utf-8", errors="replace")[-2000:]
            out["error"] = f"antsApplyTransforms ({step}) {why}: {text}"
            break
    out["wall_s"] = time.perf_counter() - t0
    return out


def image_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """RMS and max of ``a - b``, and the RMS relative to the RMS of ``b``."""
    if a.shape != b.shape:
        raise ValueError(f"images differ in shape: {a.shape} vs {b.shape}")
    d = a.astype(np.float64) - b.astype(np.float64)
    rms = float(np.sqrt(np.mean(d**2))) if d.size else math.nan
    ref = float(np.sqrt(np.mean(b.astype(np.float64) ** 2))) if b.size else math.nan
    return {
        "rms": rms,
        "max_abs": float(np.abs(d).max()) if d.size else math.nan,
        "rms_relative": rms / ref if ref else math.nan,
    }


def agreement(tmp: Path) -> dict[str, Any]:
    """FastWarp's outputs against ANTs' in ``tmp``."""
    import ants

    out: dict[str, Any] = {}
    for step in ("forward_field", "inverse_field"):
        a = ants.image_read(str(tmp / f"fastwarp_{step}.nii.gz")).numpy()
        b = ants.image_read(str(tmp / f"ants_{step}.nii.gz")).numpy()
        out[step] = metrics.field_difference(a, b)
    a = ants.image_read(str(tmp / "fastwarp_lanczos_warp.nii.gz")).numpy()
    b = ants.image_read(str(tmp / "ants_lanczos_warp.nii.gz")).numpy()
    out["lanczos_warp"] = image_difference(a, b)
    return out


def post_pair(
    ds: datasets.Dataset,
    folder: Path,
    cfg: dict[str, Any],
    *,
    data: Path,
    threads: int,
    ants_dir: Path | None,
    timeout_s: float | None,
    fastwarp_steps: Callable[..., dict[str, Any]] = run_fastwarp_steps,
    ants_steps: Callable[..., dict[str, Any]] = run_ants_steps,
    compare: Callable[[Path], dict[str, Any]] = agreement,
) -> dict[str, Any]:
    """Post-process one pair folder; returns the ``post.json`` content."""
    pair = json.loads((folder / "pair.json").read_text(encoding="utf-8"))
    record: dict[str, Any] = {"kind": "post", "started": _now(), "config": cfg}
    src = source_engine(folder, cfg["recipe"], cfg["iso_mm"])
    if src is None:
        record.update(status="skipped", reason="no finished registration in this pair")
        return record
    reference = Path(pair["fixed_prep"]["files"]["reference"])
    moving = data / ds.key / ds.subject(pair["moving"]).image
    forward, inverse = composite(folder / src), inverse_composite(folder / src)
    record.update(
        source_engine=src,
        source_finished=source_stamp(folder, src),
        reference=str(reference),
        moving=str(moving),
        forward=str(forward),
        inverse=str(inverse),
    )
    tmp = folder / "post_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        try:
            with rayon_threads(threads), itk_threads(threads):
                record["fastwarp"] = fastwarp_steps(reference, moving, forward, inverse, tmp)
        except Exception as e:
            record.update(status="failed", error=f"FastWarp: {type(e).__name__}: {e}")
            return record
        record["ants"] = ants_steps(
            reference,
            moving,
            forward,
            inverse,
            tmp,
            threads=threads,
            ants_dir=ants_dir,
            timeout_s=timeout_s,
        )
        if "error" in record["ants"]:
            record.update(status="failed", error=record["ants"]["error"])
            return record
        try:
            with itk_threads(threads):
                record["agreement"] = compare(tmp)
        except Exception as e:
            record.update(status="failed", error=f"comparing the outputs: {type(e).__name__}: {e}")
            return record
        record.update(status="ok", finished=_now(), rayon_threads_requested=threads)
        return record
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_post(
    *,
    data: Path,
    work: Path,
    keys: list[str],
    recipe_name: str = "cohort",
    pairs: int | None = None,
    quick: bool = False,
    iso_mm: float = 1.0,
    threads: int = 8,
    ants_dir: Path | None = None,
    retry_failed: bool = False,
    ants_timeout_s: float | None = 3600.0,
    log: Callable[[str], None] = print,
    **step_overrides: Any,
) -> list[dict[str, Any]]:
    """Post-process every selected pair whose registration finished and that is not done yet
    (resumable); returns one row per pair with its status."""
    rec = recipes.recipe(recipe_name)
    cfg = config(rec, iso_mm)
    rows: list[dict[str, Any]] = []
    with RunLock(work):
        env = environment(ants_dir=ants_dir, threads=threads, adapter=None)
        for key in keys:
            ds = datasets.get(key)
            n = pairs if pairs is not None else (ds.quick_pairs if quick else ds.default_pairs)
            selected = select_pairs(ds, n)
            log(f"{ds.key}: post-processing of {len(selected)} pairs")
            for i, (fixed_id, moving_id) in enumerate(selected, 1):
                folder = work / ds.key / recipe_name / pair_id(fixed_id, moving_id)
                old = _read_json(folder / "post.json")
                same = old.get("config") == cfg
                if post_done(folder, cfg) or (
                    same and old.get("status") in ("failed", "skipped") and not retry_failed
                ):
                    rows.append(_row(ds.key, folder.name, old.get("status"), False))
                    continue
                if (
                    not (folder / "pair.json").is_file()
                    or source_engine(folder, rec, iso_mm) is None
                ):
                    log(f"[{i}/{len(selected)}] {folder.name}: no finished registration; run first")
                    rows.append(_row(ds.key, folder.name, None, False))
                    continue
                log(f"[{i}/{len(selected)}] {folder.name} post-processing ...")
                try:
                    record = post_pair(
                        ds,
                        folder,
                        cfg,
                        data=data,
                        threads=threads,
                        ants_dir=ants_dir,
                        timeout_s=ants_timeout_s,
                        **step_overrides,
                    )
                except Exception as e:
                    record = {
                        "kind": "post",
                        "config": cfg,
                        "status": "failed",
                        "error": f"{type(e).__name__}: {e}",
                    }
                record["environment"] = env
                _write_json(folder / "post.json", record)
                log(f"    {record['status']}")
                rows.append(_row(ds.key, folder.name, record["status"], True))
    return rows


def _row(dataset: str, pair: str, status: str | None, ran: bool) -> dict[str, Any]:
    return {"dataset": dataset, "pair": pair, "status": status, "ran": ran}


def rows(work: Path, recipe_name: str) -> list[dict[str, Any]]:
    """One row per dataset, pair and step of the finished post-processing records: voxels,
    FastWarp and ANTs seconds, the outputs' difference and the machine fingerprint."""
    out = []
    for path, rec in timing.read_records(work.glob(f"*/{recipe_name}/*/post.json")):
        if rec.get("status") != "ok":
            continue
        fp = timing.fingerprint(rec.get("environment"))
        for step in STEPS:
            fw = ((rec.get("fastwarp") or {}).get("steps") or {}).get(step) or {}
            an = ((rec.get("ants") or {}).get("steps") or {}).get(step) or {}
            ag = (rec.get("agreement") or {}).get(step) or {}
            out.append(
                {
                    "dataset": path.parents[2].name,
                    "pair": path.parent.name,
                    "step": step,
                    "voxels": fw.get("voxels"),
                    "fastwarp_s": fw.get("seconds"),
                    "ants_s": an.get("seconds"),
                    "field_diff_rms_mm": ag.get("field_diff_rms_mm"),
                    "field_diff_max_mm": ag.get("field_diff_max_mm"),
                    "image_diff_rms_relative": ag.get("rms_relative"),
                    **fp,
                }
            )
    return out
