# Recipes

A recipe is the schedule of a registration: for each stage, the iterations per resolution level,
the shrink factors and smoothing, and the metric settings. It is a plain dictionary with up to
three stages:

```python
{"rigid": {...}, "affine": {...}, "syn": {...}}
```

{func}`~fastwarp.cohort_recipe` returns the default recipe in full. It is the recipe of Wyss DBS
Localizer, which calls it the cohort recipe. {func}`~fastwarp.register_syn` takes a recipe through `recipe=`; a stage or field that you leave
out keeps its default value. The result's `"recipe"` entry is the full recipe that ran.

## Change a few fields

Only the fields you name change. This recipe halves the SyN iterations and uses a smaller
cross-correlation window:

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
moving = ants.image_read(ants.get_ants_data("ch2"))

recipe = {"syn": {"iterations": [50, 35, 25, 10], "cc_radius": 2}}
info = fastwarp.register_syn(fixed, moving, "out/short_", recipe=recipe)
print(info["recipe"]["syn"]["iterations"], info["recipe"]["syn"]["cc_radius"])
```

```text
[50, 35, 25, 10] 2
```

To start from the full default and edit it:

```python
import fastwarp

recipe = fastwarp.cohort_recipe(channels=1)
recipe["rigid"]["iterations"] = [1000, 500, 250]
recipe["rigid"]["shrink_factors"] = [8, 4, 2]
recipe["rigid"]["smoothing_sigmas"] = [3.0, 2.0, 1.0]
```

Each stage has its own number of resolution levels: `iterations`, `shrink_factors` and
`smoothing_sigmas` of one stage must have the same length, which is that stage's number of
levels. The lists run from the coarsest level to the finest.

## Rigid and affine stages

The `rigid` and `affine` stages use Mattes mutual information on the first channel, with samples
on a regular grid. They have the same fields and the same defaults.

| Field | Type | Default | Meaning and limits |
| --- | --- | --- | --- |
| `gradient_step` | `float` | `0.1` | Largest step of an iteration, in millimetres. Greater than 0. |
| `iterations` | `list[int]` | `[1000, 500, 250, 100]` | Maximum iterations per level. At least one level. |
| `convergence_threshold` | `float` | `1e-6` | A level stops early when the slope of a line fitted to the last `convergence_window` metric values falls below this. Finite. |
| `convergence_window` | `int` | `10` | Number of iterations the convergence test looks at. At least 2. |
| `shrink_factors` | `list[int]` | `[8, 4, 2, 1]` | Downsampling factor of the images per level. At least 1. |
| `smoothing_sigmas` | `list[float]` | `[3.0, 2.0, 1.0, 0.0]` | Gaussian smoothing per level, in `sigma_units`. 0 or more. |
| `sigma_units` | `str` | `"vox"` | `"vox"` (voxels) or `"mm"` (millimetres). |
| `bins` | `int` | `32` | Histogram bins of the metric. Must be 32: the GPU kernels are built for 32 bins. |
| `sampling_percentage` | `float` | `0.25` | Fraction of the voxels used as samples, between 0 (excluded) and 1. `0.25` means 25 %. |
| `seed` | `int` | `20260910` | Seed of the random numbers used to place the samples. |

## SyN stage

The `syn` stage uses neighbourhood cross-correlation on every channel.

| Field | Type | Default | Meaning and limits |
| --- | --- | --- | --- |
| `gradient_step` | `float` | `0.1` | Step size of each update of the deformation. Greater than 0. |
| `update_field_variance` | `float` | `3.0` | Variance of the Gaussian that smooths each update, in voxels². 0 or more. |
| `total_field_variance` | `float` | `0.0` | Variance of the Gaussian that smooths the total deformation, in voxels². 0 (the default) means no smoothing. |
| `cc_radius` | `int` | `4` | Radius of the cross-correlation window, in voxels: the window is `2 * cc_radius + 1` voxels wide. 1 to 8: the GPU kernels support no larger radius. |
| `channel_weights` | `list[float]` | `[1.0]`, or `[0.5, 0.5]` for two channels | Weight of each channel's metric. One weight per channel, each 0 or more, with a positive sum. |
| `iterations` | `list[int]` | `[100, 70, 50, 20]` | As for the linear stages. |
| `convergence_threshold` | `float` | `1e-6` | As for the linear stages. |
| `convergence_window` | `int` | `10` | As for the linear stages. |
| `shrink_factors` | `list[int]` | `[8, 4, 2, 1]` | As for the linear stages. |
| `smoothing_sigmas` | `list[float]` | `[3.0, 2.0, 1.0, 0.0]` | As for the linear stages. |
| `sigma_units` | `str` | `"vox"` | As for the linear stages. |

`cohort_recipe(channels=2)` gives `channel_weights` `[0.5, 0.5]`. A recipe for a two-channel run
must have two weights.

## Mistakes are caught before any work

An unknown stage or field, a value of the wrong type, lists of different lengths in one stage, or
a value outside the limits above raises `ValueError` before the GPU is opened. The message names
the field:

```python
import numpy as np
import fastwarp

image = (np.ones((8, 8, 8), np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3))
try:
    fastwarp.register_syn(image, image, "out/x_", recipe={"syn": {"cc_radius": 12}})
except ValueError as error:
    print(error)
```

```text
recipe['syn']['cc_radius']: the GPU kernels support 1 to 8, not 12
```

## The same schedule in ANTs

The default recipe for one channel is this `antsRegistration` command. The
[accuracy benchmark](../benchmarks.md#recipe) runs ANTs with it, and its tests check that the
command and {func}`~fastwarp.cohort_recipe` agree:

```text
antsRegistration --dimensionality 3 --output [<out>/reg_]
  --winsorize-image-intensities [0.005,0.995] --use-histogram-matching 0
  --initial-moving-transform [<fixed>,<moving>,1]
  --transform Rigid[0.1]  --metric MI[<fixed>,<moving>,1,32,Regular,0.25]
    --convergence [1000x500x250x100,1e-06,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox
  --transform Affine[0.1] --metric MI[<fixed>,<moving>,1,32,Regular,0.25]
    --convergence [1000x500x250x100,1e-06,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox
  --transform SyN[0.1,3,0] --metric CC[<fixed>,<moving>,1,4]
    --convergence [100x70x50x20,1e-06,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox
  --write-composite-transform 1 --random-seed 20260910
```

The recipe fields map onto it as follows: `gradient_step` is the first argument of `Rigid[...]`,
`Affine[...]` and `SyN[...]`; `update_field_variance` and `total_field_variance` are the second
and third arguments of `SyN[...]`; `bins` and `sampling_percentage` are the fourth and sixth
arguments of `MI[...]`; `channel_weights` and `cc_radius` are the third and fourth arguments of
`CC[...]`; and `iterations`, `convergence_threshold` and `convergence_window` form
`--convergence`.
