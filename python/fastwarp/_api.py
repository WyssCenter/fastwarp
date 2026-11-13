# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""Implementation of the public FastWarp API; import the names from ``fastwarp``."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from os import PathLike, fspath
from pathlib import Path
from typing import Any

import numpy as np

from ._core import (
    CancelToken,
    GpuUnavailableError,
    gpu_info,
)
from ._core import LinearSession as _LinearSession
from ._core import apply_transforms as _apply_transforms
from ._core import check_linear_stage as _check_linear_stage
from ._core import cohort_recipe as _cohort_recipe
from ._core import compose_field as _compose_field
from ._core import fits_gpu as _fits_gpu
from ._core import register_syn as _register_syn

#: Citation for the engine, for method logs and reference lists.
CITATION = (
    "Köhler RM, Payonk JP, Neumann WJ. FastWarp: the GPU registration engine of "
    "Wyss DBS Localizer. https://github.com/WyssCenter/fastwarp; 2026."
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
    arr = np.asarray(arr)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D image, got shape {arr.shape}")
    # A view for the x-fastest (Fortran-ordered) arrays ANTs returns; copied only otherwise.
    flat = arr.ravel(order="F")
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


def _prefix_folder(prefix: str) -> Path:
    return Path(prefix + "Composite.h5").parent


def _prepared_prefix(out_prefix: str | PathLike[str]) -> str:
    """``out_prefix`` as a string, a trailing separator kept, its folder created (checked before
    the registration runs)."""
    prefix = fspath(out_prefix)
    _prefix_folder(prefix).mkdir(parents=True, exist_ok=True)
    return prefix


#: Output layouts :func:`register_syn` writes.
OUTPUT_LAYOUTS = ("composite", "split", "both")


def register_syn(
    fixed: Any | list[Any],
    moving: Any | list[Any],
    out_prefix: str | PathLike[str],
    *,
    iso_mm: float = 1.0,
    write_warped: bool = False,
    adapter: str | None = None,
    cancel: CancelToken | None = None,
    recipe: Mapping[str, Any] | None = None,
    outputs: str = "composite",
) -> dict[str, Any]:
    """Register ``moving`` to ``fixed`` on the GPU. By default: Rigid and Affine (Mattes MI,
    first channel), then SyN (CC, every channel), from a centre-of-mass start.

    ``fixed`` / ``moving``: one image, or a list of two (T1w + T2w) for the multispectral SyN
    stage, the moving T2w already in the moving T1w's space. A prefix string ending in a path
    separator, such as ``"out/sub/"``, is a folder, as in ANTs (a ``pathlib.Path`` drops a
    trailing separator). Returns timings and per-level diagnostics.

    ``outputs`` chooses the transform files (ANTs names, after ``out_prefix``):
    ``"composite"`` (default) writes ``Composite.h5`` and ``InverseComposite.h5`` (``[Affine,
    Warp]`` and ``[InverseWarp, Affine^-1]``; ``[Affine]`` and ``[Affine^-1]`` without a SyN
    stage) plus ``0GenericAffine.mat``; ``"split"`` writes what ``antsRegistration`` writes without
    ``--write-composite-transform``: ``0GenericAffine.mat`` and, after a SyN stage,
    ``1Warp.nii.gz`` / ``1InverseWarp.nii.gz`` (``float32`` fields; ANTs writes ``float64``);
    ``"both"`` writes all. The result's ``"files"``
    lists the paths written.

    ``recipe`` sets the stages and their schedule, in one of two forms. The cohort form
    ``{"rigid": {...}, "affine": {...}, "syn": {...}}`` takes the fields :func:`cohort_recipe`
    returns. The stage-list form ``{"initial": ..., "stages": [...]}`` takes ``initial``:
    ``"com"`` (default, centres of intensity mass), ``"identity"``, a transform file as
    ``pathlib.Path`` or ``{"path": ..., "invert": bool}`` (one affine, ``.mat``/``.txt``), or
    ``(parameters, center)`` / ``{"parameters": [...], "center": [...]}`` (a relative path is
    read from the current folder); and ``stages``: a list
    of dicts with ``"transform"`` ``"rigid"``, ``"affine"`` or ``"syn"`` plus that stage's fields,
    any number of linear stages followed by at most one SyN stage (SyN alone is allowed; default:
    the cohort's three). Both forms take ``"winsorize"``: ``[lower, upper]`` intensity quantiles
    (default ``[0.005, 0.995]``) or ``None`` for ANTs' plain rescale to [0, 1] (what
    ``antsRegistration`` does without ``--winsorize-image-intensities``). A stage or field left
    out keeps the dbs-localizer cohort value; an unknown key or a value the GPU kernels do not
    support (MI bins other than 32, a CC radius above 8) raises ``ValueError`` before any work.
    The result's ``"recipe"`` is the full recipe that ran, in the form given.

    ``adapter`` picks a GPU by a substring of its name. ``cancel`` stops the run at the next
    iteration when another thread calls ``cancel.cancel()``; the GIL is released while it runs.

    Raises :class:`GpuUnavailableError` (no hardware GPU), :class:`GpuCapacityError` (the
    ``iso_mm`` grid does not fit; checked for every stage before any work),
    :class:`RegistrationCancelled` and :class:`RegistrationError`. The first two mean "use another
    engine".
    """
    return _register_syn(
        [_image_args(i) for i in _as_list(fixed)],
        [_image_args(i) for i in _as_list(moving)],
        _prepared_prefix(out_prefix),
        iso_mm=float(iso_mm),
        write_warped=write_warped,
        adapter=adapter,
        cancel=cancel,
        recipe=_recipe_arg(recipe),
        outputs=outputs,
    )


def _as_dict(stage: Any) -> Any:
    return dict(stage) if isinstance(stage, Mapping) else stage


def _initial_arg(initial: Any) -> Any:
    if isinstance(initial, PathLike):
        return {"path": fspath(initial)}
    if isinstance(initial, Mapping):
        d = dict(initial)
        if isinstance(d.get("path"), PathLike):
            d["path"] = fspath(d["path"])
        return d
    if isinstance(initial, (tuple, list)) and len(initial) == 2:
        params, center = _initial((initial[0], initial[1]))
        return {"parameters": list(params), "center": list(center)}
    return initial


def _recipe_arg(recipe: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``recipe`` as plain dicts and lists for the extension: mappings copied, the stage list
    made a list of dicts, a path start made ``{"path": str}``."""
    if recipe is None:
        return None
    out: dict[str, Any] = {}
    for k, v in recipe.items():
        if k == "stages" and isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            out[k] = [_as_dict(stage) for stage in v]
        elif k == "initial":
            out[k] = _initial_arg(v)
        elif k == "winsorize" and isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = _as_dict(v)
    return out


def cohort_recipe(channels: int = 1) -> dict[str, dict[str, Any]]:
    """The dbs-localizer cohort recipe for 1 or 2 channels as :func:`register_syn` takes it:
    ``MI[32, Regular 0.25]`` Rigid and Affine at ``1000x500x250x100``, then ``SyN[0.1,3,0]``
    with ``CC`` radius 4 at ``100x70x50x20`` (weights 0.5 each for two channels); shrink
    ``8x4x2x1``, smoothing ``3x2x1x0`` voxels."""
    return _cohort_recipe(channels)


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
    fixed: Any,
    moving: Any | None = None,
    *,
    iso_mm: float = 1.0,
    channels: int | None = None,
    adapter: str | None = None,
    recipe: Mapping[str, Any] | None = None,
) -> bool:
    """Whether :func:`register_syn` of ``fixed`` at ``iso_mm`` with ``recipe`` (default: the cohort
    chain) fits the GPU (False without one).

    ``fixed`` / ``moving`` are what :func:`register_syn` takes: an image or a list of 1 or 2 of
    them (the first fixed image sets the grid; the moving grid is the largest of the moving
    images, and there are as many moving images as fixed ones). ``channels`` defaults to the number
    of fixed images; give it to ask about a second channel when passing one image. Give ``moving``
    to get exactly the answer of the capacity check ``register_syn`` runs; without it the moving
    image is assumed to have the fixed grid, which undercounts when it is larger. ``recipe`` is
    what :func:`register_syn` takes; a chain without a SyN stage needs far less memory. Invalid
    arguments raise ``ValueError`` as in :func:`register_syn`.
    """
    fixed_list = _as_list(fixed)
    if not 1 <= len(fixed_list) <= 2:
        raise ValueError("give 1 or 2 fixed images")
    moving_list = None if moving is None else _as_list(moving)
    if moving_list is not None and len(moving_list) != len(fixed_list):
        raise ValueError("give as many moving images as fixed images")
    if channels is None:
        channels = len(fixed_list)
    elif isinstance(fixed, list) and channels != len(fixed_list):
        raise ValueError(f"channels={channels} does not match {len(fixed_list)} fixed images")
    elif not 1 <= channels <= 2:
        raise ValueError("channels must be 1 or 2")
    iso_mm = float(iso_mm)
    if not (math.isfinite(iso_mm) and iso_mm > 0.0):
        raise ValueError("iso_mm must be a positive number")
    fixed_img = fixed_list[0]
    moving_dims = None
    if moving_list is not None:
        grids = [iso_grid_dims(m, iso_mm) for m in moving_list]
        moving_dims = max(grids, key=lambda d: d[0] * d[1] * d[2])
    try:
        return _fits_gpu(
            iso_grid_dims(fixed_img, iso_mm), channels, adapter, moving_dims, _recipe_arg(recipe)
        )
    except GpuUnavailableError:
        return False


def gpu_available(adapter: str | None = None) -> bool:
    """True when a usable hardware GPU adapter opens (software adapters only with
    ``FASTWARP_GPU_ALLOW_SOFTWARE=1``)."""
    try:
        gpu_info(adapter)
    except GpuUnavailableError:
        return False
    return True


#: Interpolators :func:`apply_transforms` runs (ANTs names).
INTERPOLATORS = ("linear", "nearestNeighbor", "lanczosWindowedSinc")


def _grid_args(ref: Any) -> tuple:
    """(size, spacing, origin, direction) of an ``ANTsImage`` or of a tuple
    ``(shape, spacing, origin, direction)``."""
    if isinstance(ref, tuple):
        shape, spacing, origin, direction = ref
    else:
        shape, spacing, origin, direction = ref.shape, ref.spacing, ref.origin, ref.direction
    if len(shape) != 3:
        raise ValueError(f"expected a 3-D reference, got shape {tuple(shape)}")
    d = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    return (
        tuple(int(n) for n in shape),
        tuple(float(v) for v in spacing),
        tuple(float(v) for v in origin),
        tuple(tuple(float(v) for v in row) for row in d),
    )


def _transform_paths(transforms: Any) -> list[str]:
    items = [transforms] if isinstance(transforms, (str, PathLike)) else list(transforms)
    for t in items:
        if not Path(t).is_file():
            raise FileNotFoundError(f"transform file {t} does not exist")
    return [str(t) for t in items]


def _invert_flags(invert: Any, count: int) -> list[bool] | None:
    """``invert`` as one ``bool`` per transform (``None`` stays ``None``: nothing inverted)."""
    if invert is None:
        return None
    if isinstance(invert, np.ndarray) and invert.ndim == 1 and invert.dtype == bool:
        invert = invert.tolist()
    if isinstance(invert, (bool, np.bool_)) or not isinstance(invert, Sequence):
        raise ValueError("invert must be a list with one bool per transform")
    flags = list(invert)
    if len(flags) != count:
        raise ValueError(f"invert has {len(flags)} flags for {count} transforms")
    if not all(isinstance(f, (bool, np.bool_)) for f in flags):
        raise ValueError("invert must hold bools (True inverts that transform)")
    return [bool(f) for f in flags]


def compose_field(
    transforms: str | PathLike[str] | list[str | PathLike[str]],
    reference: Any,
    out_path: str | PathLike[str],
    *,
    compression_level: int | None = None,
    invert: Sequence[bool] | None = None,
) -> Path:
    """Write the transforms as one displacement field on ``reference``'s grid.

    The file ``antsApplyTransforms --compose`` / ``ants.apply_transforms(compose=...)`` writes
    (vectors in LPS mm, ``float32``), computed in parallel and written with parallel gzip when
    ``out_path`` ends in ``.nii.gz``. ``transforms`` is a list as ``ants.apply_transforms`` takes
    it, with the same meaning; ``invert`` as in :func:`apply_transforms`.
    ``compression_level`` (0 to 9) sets the gzip level; 0 stores the data in valid gzip without
    compressing it, which suits displacement fields (they barely compress).
    """
    paths = _transform_paths(transforms)
    flags = _invert_flags(invert, len(paths))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _compose_field(
        paths,
        _grid_args(reference),
        str(out),
        compression_level=compression_level,
        invert=flags,
    )
    return out


def apply_transforms(
    moving: Any,
    reference: Any,
    transforms: str | PathLike[str] | list[str | PathLike[str]],
    *,
    interpolation: str = "linear",
    default_value: float = 0.0,
    invert: Sequence[bool] | None = None,
) -> Any:
    """Resample ``moving`` onto ``reference`` through ``transforms`` (``antsApplyTransforms``).

    ``transforms`` is a list as ``ants.apply_transforms`` takes it, with the same meaning.
    ``invert`` is ANTsPy's ``whichtoinvert``: one ``bool`` per transform (a list, tuple or 1-D
    boolean array; ``0`` / ``1`` are refused), ``True`` to use that transform's inverse in its
    place (``antsApplyTransforms -t [file,1]``). Only linear transforms
    (``.mat``, ``.txt``, or a composite holding only affines) can be inverted; a displacement
    field raises ``ValueError`` (pass the matching ``InverseWarp`` / ``InverseComposite`` file).
    ``None`` inverts nothing; unlike ANTsPy there is no automatic default for an affine followed
    by a field. ``interpolation`` is one of :data:`INTERPOLATORS`. Returns an ``ANTsImage`` like
    ``reference`` when it is one, else a ``float32`` array of the reference's shape. Points that
    map outside ``moving`` take ``default_value``.
    """
    size, spacing, origin, direction = _grid_args(reference)
    paths = _transform_paths(transforms)
    flat = _apply_transforms(
        _image_args(moving),
        (size, spacing, origin, direction),
        paths,
        interpolation=interpolation,
        default_value=float(default_value),
        invert=_invert_flags(invert, len(paths)),
    )
    arr = flat.reshape(size, order="F")
    if hasattr(reference, "new_image_like"):
        return reference.new_image_like(arr)
    return arr


def _initial(initial: tuple[Any, Any] | None) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if initial is None:
        return None
    params, center = initial
    if np.size(params) != 12 or np.size(center) != 3:
        raise ValueError("initial must be (12 affine parameters, 3-element centre)")
    return (
        tuple(float(v) for v in np.asarray(params, dtype=np.float64).ravel()),
        tuple(float(v) for v in np.asarray(center, dtype=np.float64).ravel()),
    )


class LinearSession:
    """Linear (rigid or affine) fits of one image pair on the GPU, the pair prepared once.

    Made by :func:`linear_session`. Each :meth:`register` runs one ``Rigid`` or ``Affine`` stage
    on Mattes MI with regular sampling from a given start; the images are converted, winsorized
    and uploaded once, and their smoothed copies are kept between fits, so a multi-start, or a
    rigid fit followed by an affine one, pays the preparation once. Use it in a ``with`` block, or
    :meth:`close` it, to release the GPU and the kept images.
    """

    def __init__(
        self,
        fixed: Any,
        moving: Any,
        *,
        fixed_mask: Any | None = None,
        winsorize: tuple[float, float] | None = None,
        adapter: str | None = None,
    ) -> None:
        self._core = _LinearSession(
            _image_args(fixed),
            _image_args(moving),
            fixed_mask=None if fixed_mask is None else _image_args(fixed_mask),
            winsorize=None if winsorize is None else (float(winsorize[0]), float(winsorize[1])),
            adapter=adapter,
        )

    @property
    def adapter(self) -> str:
        """The GPU in use."""
        return self._core.adapter

    @property
    def com_initial(self) -> tuple[tuple[float, ...], tuple[float, float, float]]:
        """The centre-of-mass start ``(parameters, center)``, what ``initial=None`` uses."""
        return self._core.com_initial

    def register(
        self,
        stage: Mapping[str, Any] | None = None,
        initial: tuple[Any, Any] | None = None,
        use_mask: bool = True,
        out_prefix: str | PathLike[str] | None = None,
        cancel: CancelToken | None = None,
        *,
        transform: str = "rigid",
        mask: Any | None = None,
    ) -> dict[str, Any]:
        """One ``transform`` (``"rigid"`` or ``"affine"``) fit from ``initial``.

        ``stage``: the schedule, the fields of ``cohort_recipe()["rigid"]`` (left out = that
        value). ``initial``: ``(parameters, center)`` of an ITK affine mapping fixed points to
        moving ones (12 parameters: the row-major matrix, then the translation); ``None`` starts
        from the centres of intensity mass, like ``--initial-moving-transform [f,m,1]``. The
        fitted transform starts as the identity, centred where ``initial`` is. With ``use_mask``
        the metric samples only the fixed points inside ``mask`` (an image on any grid, its
        non-zero voxels; ``None`` = the session's ``fixed_mask``, if any).

        Returns the collapsed affine (start then fit) as ``parameters`` / ``center`` (what
        ``0GenericAffine.mat`` holds, written as ``<out_prefix>0GenericAffine.mat`` when
        ``out_prefix`` is given, ``"out/sub/"`` being a folder; its folder is created), the final
        MI value ``metric`` (lower is better), the start, iterations per level and the time.
        """
        return self._core.register(
            stage=None if stage is None else dict(stage),
            initial=_initial(initial),
            use_mask=use_mask,
            out_prefix=None if out_prefix is None else _prepared_prefix(out_prefix),
            cancel=cancel,
            transform=transform,
            mask=None if mask is None else _image_args(mask),
        )

    def close(self) -> None:
        """Release the GPU and the kept images."""
        self._core.close()

    def __enter__(self) -> LinearSession:
        return self

    def __exit__(self, *args: object) -> bool:
        self.close()
        return False


#: Deprecated alias of :class:`LinearSession`; removed in fastwarp 0.3.0.
RigidSession = LinearSession


def linear_session(
    fixed: Any,
    moving: Any,
    *,
    fixed_mask: Any | None = None,
    winsorize: tuple[float, float] | None = None,
    adapter: str | None = None,
) -> LinearSession:
    """A :class:`LinearSession` for ``fixed`` / ``moving`` (``ANTsImage`` or tuples, as
    :func:`register_syn` takes them): the pair prepared once for several fits.

    ``fixed_mask``: an image on any grid whose non-zero voxels are where the metric samples (in
    fits with ``use_mask`` that bring no mask of their own). ``winsorize``: ``(lower, upper)``
    intensity quantiles, as ``--winsorize-image-intensities``; ``(0.0, 1.0)`` is exactly what
    ``antsRegistration`` does without that option (a rescale to [0, 1]). ``None`` (default) feeds
    the intensities as they are. Mattes MI is invariant to that affine rescale in exact arithmetic,
    but the ``float32`` GPU kernels are not, so the two can take different optimiser paths and
    end at slightly different transforms; pass ``(0.0, 1.0)`` to preprocess as ANTs does.
    """
    return LinearSession(fixed, moving, fixed_mask=fixed_mask, winsorize=winsorize, adapter=adapter)


#: Deprecated alias of :func:`linear_session`; removed in fastwarp 0.3.0.
rigid_session = linear_session


def register_rigid(
    fixed: Any,
    moving: Any,
    *,
    stage: Mapping[str, Any] | None = None,
    initial: tuple[Any, Any] | None = None,
    fixed_mask: Any | None = None,
    winsorize: tuple[float, float] | None = None,
    out_prefix: str | PathLike[str] | None = None,
    adapter: str | None = None,
    cancel: CancelToken | None = None,
) -> dict[str, Any]:
    """One rigid fit of ``moving`` onto ``fixed`` on the GPU (Mattes MI, regular sampling).

    ``stage``: the schedule, the fields of ``cohort_recipe()["rigid"]`` (left out = that value).
    ``initial``: ``(parameters, center)`` of an ITK affine mapping fixed points to moving ones
    (12 parameters: the row-major matrix, then the translation); ``None`` starts from the
    centres of intensity mass, like ``antsRegistration --initial-moving-transform [f,m,1]``.
    ``fixed_mask`` / ``winsorize``: as :func:`linear_session`. For several fits of one pair (a
    multi-start), a :func:`linear_session` prepares the images once.

    Returns the collapsed affine as ``parameters`` / ``center`` (what ``0GenericAffine.mat`` holds,
    written as ``<out_prefix>0GenericAffine.mat`` when ``out_prefix`` is given), the final MI
    value ``metric`` (lower is better), the start, iterations per level and the time.
    """
    # Everything a caller can get wrong is refused before the GPU is opened.
    stage_dict = _check_linear_stage(None if stage is None else dict(stage), "rigid")
    start = _initial(initial)
    prefix = None if out_prefix is None else _prepared_prefix(out_prefix)
    if prefix is not None and not _prefix_folder(prefix).is_dir():
        raise ValueError(f"output folder {_prefix_folder(prefix)} does not exist")
    if winsorize is not None and not 0.0 <= winsorize[0] < winsorize[1] <= 1.0:
        raise ValueError("winsorize must be (lower, upper) quantiles with 0 <= lower < upper <= 1")
    with linear_session(
        fixed, moving, fixed_mask=fixed_mask, winsorize=winsorize, adapter=adapter
    ) as session:
        return session.register(
            stage=stage_dict,
            initial=start,
            use_mask=fixed_mask is not None,
            out_prefix=prefix,
            cancel=cancel,
        )
