# Recipes

A {term}`recipe` is the schedule of a registration: where it starts, which stages run, and for
each stage the iterations per resolution level, the shrink factors and smoothing, and the metric
settings. {func}`~fastwarp.register_syn` takes it through `recipe=`, in one of two forms:

The cohort form
: `{"rigid": {...}, "affine": {...}, "syn": {...}}`: the three default stages, each with the
  fields you want to change. {func}`~fastwarp.cohort_recipe` returns it in full. It is the
  recipe of [Wyss DBS Localizer](https://github.com/WyssCenter/dbs-localizer), which calls it
  the cohort recipe.

The stage-list form
: `{"initial": ..., "stages": [...]}`: any start, and any number of rigid and affine stages
  followed by any number of SyN stages, each stage with switches for the masks of the call (see
  [Stage lists](#stage-lists)).

Both forms also take `"winsorize"` (see [Winsorizing](#winsorizing)); the two forms' other keys do
not mix. A stage or field that you leave out keeps its default value. The result's `"recipe"`
entry is the full recipe that ran, in the form you gave. The same recipes, as JSON files, work
with `fastwarp register --recipe` on the [command line](../cli.md#register).

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

```{code-block} text
:class: fw-output
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

## Stage lists

The stage-list form names the start and the stages. This one runs only the rigid and the affine
stage, from the centres of mass, and writes both file layouts (see
[Output files](outputs.md#split-files)):

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
moving = ants.image_read(ants.get_ants_data("ch2"))

recipe = {"initial": "com", "stages": [{"transform": "rigid"}, {"transform": "affine"}]}
info = fastwarp.register_syn(fixed, moving, "out/linear_", recipe=recipe, outputs="both")
print(info["deformable"], [stage["transform"] for stage in info["linear_stages"]])
print(info["files"])
```

```{code-block} text
:class: fw-output
False ['rigid', 'affine']
['out/linear_Composite.h5', 'out/linear_InverseComposite.h5', 'out/linear_0GenericAffine.mat']
```

`initial`
: Where the chain starts. `"com"` (the default, also for `None`): the translation between the
  {term}`centres of mass <centre of mass>`, as `antsRegistration -r [fixed,moving,1]`. Note that
  `antsRegistration` without `-r` starts from the identity instead. `"identity"`: no start, as
  `-r Identity`; a first linear stage then rotates about the origin. A transform file with one
  affine (`.mat`, `.txt`, or an `.h5` that holds one affine), as a `pathlib.Path` or as
  `{"path": ..., "invert": True}` for its inverse (`-r [file,1]`); a relative path is read from
  the current folder. Or the affine itself, as `(parameters, center)` or
  `{"parameters": [12 values], "center": [3 values]}`, in the form described under
  [Linear fits](linear.md#one-fit).

`stages`
: A list of dicts, each with `"transform"` set to `"rigid"`, `"affine"` or `"syn"` and any of
  that stage's fields (see the tables below). Any number of rigid and affine stages, in any
  order, then any number of SyN stages, which must come last (see
  [Several SyN stages](#several-syn-stages)). SyN alone is allowed. `None` (or leaving `stages`
  out) runs the default three stages. Several {term}`channels <channel>` need a SyN stage,
  because only SyN uses the further images. Each stage may also carry the mask switches
  `"fixed_mask"` and, on a SyN stage, `"moving_mask"` (see [Mask switches](#mask-switches)).

Each linear stage starts as the identity, centred where the previous linear transform is
centred, and is applied before everything earlier, as ANTs chains its stages; all of them are
combined into one affine. The result reports each linear stage in `linear_stages` and whether a
SyN stage ran in `deformable` (see [Normalization](normalization.md#result)). With the default
start, stages and winsorizing, a stage list gives the same output, bit for bit, as the cohort
form.

A recipe without a SyN stage needs much less GPU memory, because the linear stages hold only
their sample buffers: on the laptop of the other examples, the default recipe at `iso_mm=0.5`
does not fit, while the rigid-and-affine recipe above does. Pass the recipe to
{func}`~fastwarp.fits_gpu` to check (see [What fits](gpu.md#what-fits)).

### Several SyN stages

A stage list may end with several SyN stages, as `antsRegistration` runs several `-t SyN[...]`
stages. Each SyN stage starts from everything before it: it samples the moving image (and the
moving mask) through its own moving-side field, then through the field of the earlier SyN stages,
then through the affine, with one interpolation of the image. After each SyN stage its field is
composed with the earlier one, as `antsRegistration --collapse-output-transforms 1` collapses
consecutive displacement fields, so the output is still one affine and one field per direction:
`Composite.h5` holds `[Affine, Warp]`, and the [split files](outputs.md#split-files) are
`0GenericAffine.mat`, `1Warp.nii.gz` and `1InverseWarp.nii.gz`.

Every SyN stage has its own fields (iterations, levels, window radius, gradient step, ...), and
all SyN stages register the same channels. They must end with the same shrink factor (usually 1),
so that their fields share the grid of the last level. The result reports the levels of every
SyN stage, one stage after the other, in `levels`, and the same levels grouped by stage in
`syn_stages` (see [Normalization](normalization.md#result)). Each SyN stage needs the GPU memory of
a single SyN stage, and each SyN stage after the first 12 bytes per voxel more for the field of
the earlier stages (see [GPU memory](gpu.md#what-fits)).

With three or more SyN stages, a stage sees the earlier SyN stages through their collapsed field
(the field the output holds), while `antsRegistration` chains their fields one after the other;
the two differ only where the collapsed field is interpolated between its voxels (by up to
0.07 mm in a test with synthetic fields of 2 mm that change by up to 0.4 mm per mm; smoother
fields differ less). With two SyN stages there is no difference.

### Mask switches

The masks of a call (`fixed_mask=` and `moving_mask=` of {func}`~fastwarp.register_syn`, or
`--mask` and `--moving-mask` of `fastwarp register`) apply to every stage they are allowed on: the
fixed mask to every stage, the moving mask to every SyN stage (see
[Masks](normalization.md#masks)). In a stage list each stage can switch them off for itself:

`"fixed_mask"`
: `True` (the default) or `False`: whether this stage uses the fixed mask.

`"moving_mask"`
: `True` (the default) or `False`: whether this SyN stage uses the moving mask. Rigid and affine
  stages never use the moving mask, so they have no such switch, and giving one is an error.

A switch that is on while the call gives no mask does nothing. A mask that every stage switches
off is refused, as is a moving mask for a recipe without a SyN stage. The result's `"recipe"`
reports both switches of every stage. This is how `antsRegistration` takes one `--masks` per
stage, with each stage's masks being either the call's masks or `NULL`.

A "subcortical refine": a whole-brain SyN, then a second SyN that only the region of a
subcortical mask of the template drives, with a mask of the resection cavity in the subject left
out of both SyN stages. The file names are placeholders for your own images:

```python
import ants
import fastwarp

template = ants.image_read("template_T1w.nii.gz")
subject = ants.image_read("sub-01_T1w.nii.gz")
subcortical = ants.image_read("template_subcortical_mask.nii.gz")
cavity = ants.image_read("sub-01_cavity.nii.gz")

recipe = {
    "initial": "com",
    "stages": [
        {"transform": "rigid", "fixed_mask": False},
        {"transform": "affine", "fixed_mask": False},
        {"transform": "syn", "fixed_mask": False},
        {
            "transform": "syn",
            "iterations": [40, 20],
            "shrink_factors": [2, 1],
            "smoothing_sigmas": [1.0, 0.0],
        },
    ],
}
info = fastwarp.register_syn(
    template,
    subject,
    "out/sub-01_",
    recipe=recipe,
    fixed_mask=subcortical,
    moving_mask=cavity == 0,
)
print([len(levels) for levels in info["syn_stages"]])
```

```{code-block} text
:class: fw-output
[4, 2]
```

The linear stages and the first SyN stage switch the subcortical mask off; the second SyN stage
keeps both masks, so it only updates the deformation in and around the subcortical region. The
same run with `antsRegistration` gives each stage its masks:
`-x [NULL,NULL]` for the rigid and the affine stage, `-x [NULL,cavity_mask]` for the first SyN
stage and `-x [subcortical_mask,cavity_mask]` for the second; `fastwarp ants-registration` takes
exactly that (see [ANTs compatibility](ants.md)). As a JSON file for
`fastwarp register --recipe`, write `false` for `False`.

## Winsorizing

The key `"winsorize"` sets how the intensities are prepared before the stages run:

- `[lower, upper]` quantiles, by default `[0.005, 0.995]`: the intensities are
  {term}`winsorized <winsorize>` to these quantiles and rescaled to [0, 1], as
  `antsRegistration --winsorize-image-intensities [0.005,0.995]`.
- `None`: the intensities are only rescaled to [0, 1], which is what `antsRegistration` (and every
  ANTsPy registration) does without `--winsorize-image-intensities`.

This SyN-only recipe starts from the identity and does not winsorize:

```python
recipe = {"initial": "identity", "stages": [{"transform": "syn", "cc_radius": 2}], "winsorize": None}
```

## Rigid and affine stages

The `rigid` and `affine` stages use {term}`Mattes mutual information` on the first channel, with
samples on a regular grid. They have the same fields and the same defaults.

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

The `syn` stage uses neighbourhood {term}`cross-correlation` on every channel.

| Field | Type | Default | Meaning and limits |
| --- | --- | --- | --- |
| `gradient_step` | `float` | `0.1` | Step size of each update of the deformation. Greater than 0. |
| `update_field_variance` | `float` | `3.0` | Variance of the Gaussian that smooths each update, in voxels². 0 or more. |
| `total_field_variance` | `float` | `0.0` | Variance of the Gaussian that smooths the total deformation, in voxels². 0 (the default) means no smoothing. |
| `cc_radius` | `int` | `4` | Radius of the cross-correlation window, in voxels: the window is `2 * cc_radius + 1` voxels wide. 1 to 8: the GPU kernels support no larger radius. |
| `channel_weights` | `list[float]` | `[1.0]`, or `1 / channels` each (`[0.5, 0.5]` for two) | Weight of each channel's metric. One weight per channel, each 0 or more, with a positive sum. |
| `iterations` | `list[int]` | `[100, 70, 50, 20]` | As for the linear stages. |
| `convergence_threshold` | `float` | `1e-6` | As for the linear stages. |
| `convergence_window` | `int` | `10` | As for the linear stages. |
| `shrink_factors` | `list[int]` | `[8, 4, 2, 1]` | As for the linear stages. |
| `smoothing_sigmas` | `list[float]` | `[3.0, 2.0, 1.0, 0.0]` | As for the linear stages. |
| `sigma_units` | `str` | `"vox"` | As for the linear stages. |

`cohort_recipe(channels=2)` gives `channel_weights` `[0.5, 0.5]`, and `channels=4` gives four
weights of `0.25`. A recipe for a run of several channels must have one weight per channel.

## Mistakes are caught before any work

An unknown stage or field, a value of the wrong type, lists of different lengths in one stage, a
value outside the limits above, an empty stage list, a linear stage after a SyN stage, SyN
stages that end with different shrink factors, a mask switch that is not `True` or `False` or a
`"moving_mask"` switch on a linear stage, keys of both forms in one recipe, a start file that does
not exist, a start affine that cannot be inverted, or several channels without a SyN stage raises
`ValueError` before the GPU is opened. The message
names the field:

```python
import numpy as np
import fastwarp

image = (np.ones((8, 8, 8), np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3))
try:
    fastwarp.register_syn(image, image, "out/x_", recipe={"syn": {"cc_radius": 12}})
except ValueError as error:
    print(error)
```

```{code-block} text
:class: fw-output
recipe['syn']['cc_radius']: the GPU kernels support 1 to 8, not 12
```

## The same schedule in ANTs

The default recipe for one channel is this `antsRegistration` command. The
[accuracy benchmark](../benchmarks.md#recipe) runs ANTs with it, and its tests check that the
command and {func}`~fastwarp.cohort_recipe` agree. `--verbose 1` only makes ANTs print its
progress and stage times; it does not change the registration.

```{literalinclude} ../_includes/ants-cohort-command.txt
:language: text
```

`initial` corresponds to `--initial-moving-transform` (`-r`): `"com"` to `[fixed,moving,1]`,
`"identity"` to `Identity`, a file to the file or `[file,1]`. The recipe fields map onto it as
follows: `gradient_step` is the first argument of `Rigid[...]`,
`Affine[...]` and `SyN[...]`; `update_field_variance` and `total_field_variance` are the second
and third arguments of `SyN[...]`; `bins` and `sampling_percentage` are the fourth and sixth
arguments of `MI[...]`; `channel_weights` and `cc_radius` are the third and fourth arguments of
`CC[...]`; and `iterations`, `convergence_threshold` and `convergence_window` form
`--convergence`.
