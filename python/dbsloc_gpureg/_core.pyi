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
) -> dict[str, Any]: ...
def gpu_info(adapter: str | None = None) -> dict[str, Any]: ...
def fits_gpu(dims: tuple[int, int, int], channels: int, adapter: str | None = None) -> bool: ...
