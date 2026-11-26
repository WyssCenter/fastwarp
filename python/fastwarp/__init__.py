# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""FastWarp: GPU SyN registration for Python, developed by Richard Köhler and Jan Philipp Payonk
as the registration engine of Wyss DBS Localizer.

The engine is a wgpu port of ANTs/ITK's SyN with the CC and Mattes MI metrics (Vulkan, Metal,
DX12). :func:`register_syn` runs the dbs-localizer cohort recipe (centre of mass, Rigid,
Affine, SyN on an isotropic grid), or any start, linear stages and SyN stages a stage-list
recipe names, and writes ANTs-compatible transforms.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._api import (
    CITATION,
    INTERPOLATORS,
    OUTPUT_LAYOUTS,
    LinearSession,
    apply_transforms,
    cohort_recipe,
    compose_field,
    fits_gpu,
    gpu_available,
    iso_grid_dims,
    joint_label_fusion,
    linear_session,
    register_atlases,
    register_syn,
)
from ._core import (
    ENGINE_SOURCE,
    CancelToken,
    GpuCapacityError,
    GpuUnavailableError,
    RegistrationCancelled,
    RegistrationError,
    gpu_info,
    release_idle_gpus,
)

try:
    __version__ = version("fastwarp")
except PackageNotFoundError:  # pragma: no cover - source tree without metadata
    __version__ = "0+unknown"

__all__ = [
    "CITATION",
    "ENGINE_SOURCE",
    "CancelToken",
    "GpuCapacityError",
    "INTERPOLATORS",
    "OUTPUT_LAYOUTS",
    "GpuUnavailableError",
    "RegistrationCancelled",
    "RegistrationError",
    "LinearSession",
    "__version__",
    "apply_transforms",
    "cohort_recipe",
    "compose_field",
    "gpu_available",
    "gpu_info",
    "iso_grid_dims",
    "joint_label_fusion",
    "linear_session",
    "register_atlases",
    "register_syn",
    "release_idle_gpus",
    "fits_gpu",
]

for _obj in (
    apply_transforms,
    cohort_recipe,
    compose_field,
    fits_gpu,
    gpu_available,
    gpu_info,
    iso_grid_dims,
    joint_label_fusion,
    linear_session,
    register_atlases,
    register_syn,
    release_idle_gpus,
):
    _obj.__module__ = "fastwarp"
del _obj


def __dir__() -> list[str]:
    return sorted(__all__)
