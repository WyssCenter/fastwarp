# Normalization

{func}`~fastwarp.register_syn` aligns a moving image to a fixed image with a rigid, an affine
and a deformable (SyN) stage, in one call, and writes the result as ANTs transform files.

```{eval-rst}
.. py:function:: fastwarp.register_syn(fixed, moving, out_prefix, *, iso_mm=1.0, write_warped=False, adapter=None, cancel=None, recipe=None)
   :no-index:
```

| Argument | Meaning |
| --- | --- |
| `fixed`, `moving` | One [image](images.md) each, or a list of two images each (see [Two channels](#two-channels)). |
| `out_prefix` | Where the files go: a file-name prefix, or a folder when it ends in `/` (see [Output prefix](#output-prefix)). |
| `iso_mm` | Voxel size of the registration grid, in millimetres. |
| `write_warped` | Also write the warped moving image. |
| `adapter` | Use the GPU whose name contains this text (case-insensitive). `None` uses `FASTWARP_GPU_ADAPTER` if it is set, and otherwise lets FastWarp choose. |
| `cancel` | A {class}`~fastwarp.CancelToken` to stop the run from another thread. |
| `recipe` | The registration schedule; `None` uses the default. See [Recipes](recipes.md). |

## What happens in a run

1. The arguments and the recipe are checked; a mistake raises `ValueError`.
2. The GPU is opened and the [capacity check](gpu.md#what-fits) runs. If there is no usable
   GPU, or the grid does not fit, the call raises {class}`~fastwarp.GpuUnavailableError` or
   {class}`~fastwarp.GpuCapacityError`. No transform has been computed or written at this
   point; only the output folder may have been created.
3. Every image is resampled to the spacing of the registration grid (see below).
4. The intensities of every image are winsorized to its 0.005 and 0.995 quantiles and rescaled
   to [0, 1]. There is no histogram matching.
5. The centres of intensity mass of the first fixed and the first moving image give the starting
   translation, as `antsRegistration --initial-moving-transform [fixed,moving,1]` does. They are
   computed on the resampled images before winsorizing.
6. **Rigid**, then **Affine**: Mattes mutual information on the first channel.
7. **SyN**: neighbourhood cross-correlation on every channel, starting from the affine.
8. The first moving image is warped onto the registration grid, and the files are written.

The [recipe](recipes.md) sets the iterations, resolution levels and metric settings of the
three stages. Steps 3 to 5 are fixed.

## The registration grid

The registration runs on an isotropic grid with `iso_mm` spacing, built from the first fixed
image: it has the fixed image's origin and direction, and along each axis
`floor((n - 1) * spacing / iso_mm) + 1` voxels, where `n` and `spacing` are the fixed image's
size and spacing along that axis. An axis whose spacing already equals `iso_mm` keeps its size.
{func}`~fastwarp.iso_grid_dims` returns the size:

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
print(fastwarp.iso_grid_dims(fixed, 1.0), fastwarp.iso_grid_dims(fixed, 2.0))
```

```text
(182, 218, 182) (91, 109, 91)
```

The moving images are resampled to the same spacing on their own grids. The displacement fields
in the output files live on the registration grid. A coarser grid (a larger `iso_mm`) needs less
GPU memory and time; a finer one resolves finer detail. See [GPU and memory](gpu.md) for the
memory a grid needs.

## Output prefix

`out_prefix` is joined to the file names as text, as in ANTs. With the prefix `out/sub_`:

| File | Content |
| --- | --- |
| `out/sub_Composite.h5` | Forward transform, `[Affine, Warp]`. |
| `out/sub_InverseComposite.h5` | Inverse transform, `[InverseWarp, Affine⁻¹]`. |
| `out/sub_0GenericAffine.mat` | The affine part alone. |
| `out/sub__Warped.nii.gz` | Only with `write_warped=True`: the first moving image warped onto the registration grid. The name has two underscores because the prefix ends in one. |

[Output files](outputs.md) explains what each file maps and how to use it.

- A prefix that ends in a path separator names a folder: `out/sub/` gives `out/sub/Composite.h5`
  and so on. The separator is `/`, and on Windows also `\`.
- Pass a folder prefix as a string. `pathlib.Path("out/sub/")` drops the trailing separator and
  becomes the prefix `out/sub`, which gives `out/subComposite.h5`.
- `.` and `..` are text as well: the prefix `.` gives `.Composite.h5`. Use `./` for the current
  folder.
- The folder the files go to is created when it does not exist.

## Result

{func}`~fastwarp.register_syn` returns a dictionary:

| Key | Type | Content |
| --- | --- | --- |
| `adapter` | `str` | The GPU, its graphics API and type, for example `"Intel(R) Arc(TM) 140V GPU (16GB) (Vulkan, IntegratedGpu)"`. |
| `grid` | `list[int]` | Voxels per axis of the registration grid. |
| `total_s` | `float` | Seconds from opening the GPU to the end of writing the files, including preparing and uploading the images. |
| `resample_s` | `float` | Seconds to resample the images to the registration grid. |
| `preprocess_s` | `float` | Seconds to winsorize and rescale the images. |
| `linear_s` | `float` | Seconds for the centre-of-mass start and the Rigid and Affine stages. |
| `syn_s` | `float` | Seconds for the SyN stage. |
| `warp_s` | `float` | Seconds to warp the first moving image onto the registration grid. |
| `write_s` | `float` | Seconds to write the output files. |
| `levels` | `list[tuple]` | One tuple per SyN level, coarsest first (see below). |
| `affine_parameters` | `list[float]` | The 12 parameters of the affine (row-major 3 × 3 matrix, then the translation), as in `0GenericAffine.mat`. |
| `affine_center` | `list[float]` | The centre of rotation of that affine, in LPS millimetres. |
| `peak_gpu_bytes` | `int` | The most GPU memory the engine's buffers held at one time during the run. |
| `memory_budget_bytes` | `int` | The memory budget the capacity check used. |
| `recipe` | `dict` | The full recipe that ran, every field present. |

Each entry of `levels` is a tuple `(level, grid_size, iterations, converged, energy)`:

| Item | Type | Meaning |
| --- | --- | --- |
| `level` | `int` | Index of the level, from 0 (the coarsest). |
| `grid_size` | `list[int]` | Voxels per axis of the level's grid. |
| `iterations` | `int` | Iterations the level ran. |
| `converged` | `bool` | `True` when the convergence test stopped the level before its iteration limit. |
| `energy` | `float` | The metric value of the last iteration, the negative cross-correlation of the first channel (lower is better); 0.0 when the level ran no iteration. |

## Two channels

Pass two images on each side to use two channels in the SyN stage, for example a T1-weighted
and a T2-weighted scan:

```python
info = fastwarp.register_syn(
    [template_t1, template_t2], [subject_t1, subject_t2], "out/sub_"
)
```

The names are placeholders for your own images. The Rigid and Affine stages use only the first
channel; SyN uses both, each with the weight in `channel_weights` (0.5 each by default). The
second moving image must already be aligned to the first one, for example by a rigid fit with
{func}`~fastwarp.linear_session`. Two channels need more GPU memory; pass `channels=2` to
{func}`~fastwarp.fits_gpu`.

## Cancelling a run

A {class}`~fastwarp.CancelToken` stops a run from another thread. The run stops at its next
iteration and raises {class}`~fastwarp.RegistrationCancelled`; no files are written. Cancelling
the token before the run starts also raises it.

```python
import threading

import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
moving = ants.image_read(ants.get_ants_data("ch2"))
token = fastwarp.CancelToken()
outcome = {}


def run():
    try:
        outcome["info"] = fastwarp.register_syn(fixed, moving, "out/cancel_", cancel=token)
    except fastwarp.RegistrationCancelled:
        outcome["cancelled"] = True


worker = threading.Thread(target=run)
worker.start()
threading.Timer(2.0, token.cancel).start()
worker.join()
print(outcome)
```

```text
{'cancelled': True}
```

A token stays cancelled; use a new one for the next run. `token.cancelled` tells whether
`cancel()` has been called.

## Threads and repeat runs

The call releases Python's global interpreter lock while the engine runs, so other Python threads
keep running. Two runs with the same inputs on the same GPU give the same transforms
(`tests/test_register.py::test_runs_are_deterministic`).
