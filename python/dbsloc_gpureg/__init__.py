"""GPU SyN registration from Wyss DBS Localizer (Richard Köhler, Wyss Center), for Python.

The engine is a wgpu port of ANTs/ITK's SyN with the CC and Mattes MI metrics (Vulkan, Metal,
DX12). :func:`register_syn` runs the dbs-localizer cohort recipe (centre of mass, Rigid,
Affine, SyN on an isotropic grid) and writes ANTs-compatible composite transforms.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from typing import Any

import numpy as np

from ._core import (
    ENGINE_SOURCE,
    CancelToken,
    GpuCapacityError,
    GpuUnavailableError,
    RegistrationCancelled,
    RegistrationError,
    gpu_info,
)
from ._core import fits_gpu as _fits_gpu
from ._core import register_syn as _register_syn

try:
    __version__ = version("dbsloc-gpureg")
except PackageNotFoundError:  # pragma: no cover - source tree without metadata
    __version__ = "0+unknown"

__all__ = [
    "CITATION",
    "ENGINE_SOURCE",
    "CancelToken",
    "GpuCapacityError",
    "GpuUnavailableError",
    "RegistrationCancelled",
    "RegistrationError",
    "__version__",
    "gpu_available",
    "gpu_info",
    "iso_grid_dims",
    "register_syn",
    "fits_gpu",
]

#: Citation for the engine, for method logs and reference lists.
CITATION = (
    "Köhler R. Wyss DBS Localizer: GPU (wgpu) port of the ANTs SyN registration. "
    "Wyss Center for Bio and Neuroengineering, Geneva; 2026."
)


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


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else [x]


def register_syn(
    fixed: Any | list[Any],
    moving: Any | list[Any],
    out_prefix: str | PathLike[str],
    *,
    iso_mm: float = 1.0,
    write_warped: bool = False,
    adapter: str | None = None,
    cancel: CancelToken | None = None,
) -> dict[str, Any]:
    """Register ``moving`` to ``fixed`` on the GPU (dbs-localizer cohort recipe).

    ``fixed`` / ``moving``: one image, or a list of two (T1w + T2w) for the multispectral SyN
    stage, the moving T2w already in the moving T1w's space. Writes
    ``<out_prefix>Composite.h5`` and ``<out_prefix>InverseComposite.h5`` (ANTs layout:
    ``[Affine, Warp]`` and ``[InverseWarp, Affine^-1]``) plus ``<out_prefix>0GenericAffine.mat``.
    Returns timings and per-level diagnostics.

    ``adapter`` picks a GPU by a substring of its name. ``cancel`` stops the run at the next
    iteration when another thread calls ``cancel.cancel()``; the GIL is released while it runs.

    Raises :class:`GpuUnavailableError` (no hardware GPU), :class:`GpuCapacityError` (the
    ``iso_mm`` grid does not fit; checked before any work), :class:`RegistrationCancelled` and
    :class:`RegistrationError`. The first two mean "use another engine".
    """
    return _register_syn(
        [_image_args(i) for i in _as_list(fixed)],
        [_image_args(i) for i in _as_list(moving)],
        str(out_prefix),
        iso_mm=float(iso_mm),
        write_warped=write_warped,
        adapter=adapter,
        cancel=cancel,
    )


def iso_grid_dims(img: Any, iso_mm: float = 1.0) -> tuple[int, int, int]:
    """Voxels per axis of the registration grid :func:`register_syn` builds from ``img``."""
    shape = img.shape if not isinstance(img, tuple) else np.shape(img[0])
    spacing = img.spacing if not isinstance(img, tuple) else img[1]
    out = []
    for n, sp in zip(shape, spacing, strict=True):
        if abs(sp - iso_mm) < 1e-6:
            out.append(int(n))
        else:
            out.append(max(int(np.floor((n - 1) * sp / iso_mm + 1e-9)) + 1, 1))
    return (out[0], out[1], out[2])


def fits_gpu(
    fixed: Any, *, iso_mm: float = 1.0, channels: int = 1, adapter: str | None = None
) -> bool:
    """Whether :func:`register_syn` of ``fixed`` at ``iso_mm`` fits the GPU (False without one)."""
    try:
        return _fits_gpu(iso_grid_dims(fixed, iso_mm), channels, adapter)
    except GpuUnavailableError:
        return False


def gpu_available(adapter: str | None = None) -> bool:
    """True when a usable hardware GPU adapter opens (software adapters only with
    ``DBSLOC_GPU_ALLOW_SOFTWARE=1``)."""
    try:
        gpu_info(adapter)
    except GpuUnavailableError:
        return False
    return True
