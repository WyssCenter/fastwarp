"""GPU SyN registration from Wyss DBS Localizer (Richard Köhler, Wyss Center), for Python.

The engine is a wgpu port of ANTs/ITK's SyN with the CC and Mattes MI metrics (Vulkan, Metal,
DX12). :func:`register_syn` runs the dbs-localizer cohort recipe (centre of mass, Rigid,
Affine, SyN on an isotropic grid) and writes ANTs-compatible composite transforms.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np

from ._core import ENGINE_SOURCE, gpu_info
from ._core import register_syn as _register_syn

__all__ = ["ENGINE_SOURCE", "gpu_available", "gpu_info", "register_syn"]


def _image_args(img: Any) -> tuple:
    """(flat x-fastest float64 data, size, spacing, origin, direction) in ITK (LPS) geometry.

    Accepts an ``ANTsImage`` (or anything with ``numpy()``, ``spacing``, ``origin``,
    ``direction``) or a tuple ``(array, spacing, origin, direction)`` with ``array[i, j, k]``
    the voxel at ITK index (i, j, k).
    """
    if isinstance(img, tuple):
        arr, spacing, origin, direction = img
    else:
        arr, spacing, origin, direction = img.numpy(), img.spacing, img.origin, img.direction
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D image, got shape {arr.shape}")
    flat = np.ascontiguousarray(arr.ravel(order="F"))
    d = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    return (
        flat,
        tuple(int(n) for n in arr.shape),
        tuple(float(s) for s in spacing),
        tuple(float(o) for o in origin),
        tuple(tuple(float(v) for v in row) for row in d),
    )


def register_syn(
    fixed: Any | list[Any],
    moving: Any | list[Any],
    out_prefix: str | PathLike[str],
    *,
    iso_mm: float = 1.0,
    write_warped: bool = False,
    adapter: str | None = None,
) -> dict[str, Any]:
    """Register ``moving`` to ``fixed`` on the GPU (dbs-localizer cohort recipe).

    ``fixed`` / ``moving``: one image, or a list of two (T1w + T2w) for the multispectral SyN
    stage, the moving T2w already in the moving T1w's space. Writes ``<out_prefix>Composite.h5`` and
    ``<out_prefix>InverseComposite.h5`` (ANTs layout: ``[Affine, Warp]`` and
    ``[InverseWarp, Affine^-1]``). Returns timings and per-level diagnostics.
    ``adapter`` picks a GPU by a substring of its name; the GIL is released while it runs.
    """
    fixed_l = fixed if isinstance(fixed, list) else [fixed]
    moving_l = moving if isinstance(moving, list) else [moving]
    return _register_syn(
        [_image_args(i) for i in fixed_l],
        [_image_args(i) for i in moving_l],
        str(out_prefix),
        iso_mm=float(iso_mm),
        write_warped=write_warped,
        adapter=adapter,
    )


def gpu_available(adapter: str | None = None) -> bool:
    """True when a usable hardware GPU adapter opens (no software fallback)."""
    try:
        gpu_info(adapter)
    except RuntimeError:
        return False
    return True
