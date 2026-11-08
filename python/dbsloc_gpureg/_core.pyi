from typing import Any

import numpy as np
import numpy.typing as npt

ENGINE_SOURCE: str
VERSION: str

class GpuUnavailableError(RuntimeError): ...
class GpuCapacityError(RuntimeError): ...
class RegistrationError(RuntimeError): ...
class RegistrationCancelled(Exception): ...

class CancelToken:
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    @property
    def cancelled(self) -> bool: ...

_Image = tuple[
    npt.NDArray[np.float32] | npt.NDArray[np.float64],
    tuple[int, int, int],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
]

def register_syn(
    fixed: list[_Image],
    moving: list[_Image],
    out_prefix: str,
    iso_mm: float = 1.0,
    write_warped: bool = False,
    adapter: str | None = None,
    cancel: CancelToken | None = None,
    recipe: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]: ...
def gpu_info(adapter: str | None = None) -> dict[str, Any]: ...
def fits_gpu(dims: tuple[int, int, int], channels: int, adapter: str | None = None) -> bool: ...
def compose_field(
    transforms: list[str],
    reference: tuple[
        tuple[int, int, int],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    ],
    out_path: str,
    compression_level: int | None = None,
) -> None: ...
def apply_transforms(
    moving: _Image,
    reference: tuple[
        tuple[int, int, int],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    ],
    transforms: list[str],
    interpolation: str = "linear",
    default_value: float = 0.0,
) -> npt.NDArray[np.float32]: ...
def cohort_recipe(channels: int = 1) -> dict[str, dict[str, Any]]: ...

class RigidSession:
    def __init__(
        self,
        fixed: _Image,
        moving: _Image,
        fixed_mask: _Image | None = None,
        winsorize: tuple[float, float] | None = None,
        adapter: str | None = None,
    ) -> None: ...
    @property
    def adapter(self) -> str: ...
    @property
    def com_initial(self) -> tuple[tuple[float, ...], tuple[float, float, float]]: ...
    def register(
        self,
        stage: dict[str, Any] | None = None,
        initial: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        use_mask: bool = True,
        out_prefix: str | None = None,
        cancel: CancelToken | None = None,
    ) -> dict[str, Any]: ...
    def close(self) -> None: ...
    def __enter__(self) -> RigidSession: ...
    def __exit__(self, *args: object) -> bool: ...

def check_rigid_stage(stage: dict[str, Any] | None = None) -> dict[str, Any]: ...
