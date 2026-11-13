# User guide

FastWarp has three groups of functions:

- **Registration on the GPU.** {func}`~fastwarp.register_syn` runs a full normalization (Rigid,
  Affine and SyN) and writes ANTs transform files. {func}`~fastwarp.linear_session` runs rigid or
  affine fits of one image pair.
- **Resampling on the CPU.** {func}`~fastwarp.apply_transforms` and
  {func}`~fastwarp.compose_field` apply ANTs transform files, like `antsApplyTransforms`.
- **GPU checks.** {func}`~fastwarp.gpu_available`, {func}`~fastwarp.gpu_info` and
  {func}`~fastwarp.fits_gpu` tell, before any work, whether a registration can run on this
  machine.

The pages below explain each group. They use the words of ANTs: the **fixed** image is the
reference (for example a template), and the **moving** image is the one that is aligned to it
(for example a subject's scan).

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Images
:link: guide/images
:link-type: doc

What an image is: an `ANTsImage` or a NumPy array with its geometry.
:::

:::{grid-item-card} Normalization
:link: guide/normalization
:link-type: doc

`register_syn`: the stages, the registration grid, the result and cancelling a run.
:::

:::{grid-item-card} Output files
:link: guide/outputs
:link-type: doc

The transform files, which direction each one maps, and the `antsApplyTransforms` commands.
:::

:::{grid-item-card} Recipes
:link: guide/recipes
:link-type: doc

The registration schedule: every field, its type, unit, default and limits.
:::

:::{grid-item-card} Linear fits
:link: guide/linear
:link-type: doc

`linear_session`: rigid and affine fits, multi-start searches and masks.
:::

:::{grid-item-card} Resampling
:link: guide/resampling
:link-type: doc

`apply_transforms` and `compose_field` on the CPU.
:::

:::{grid-item-card} GPU and memory
:link: guide/gpu
:link-type: doc

Which GPU is used, the memory budget, what fits, and falling back to ANTs.
:::
::::

```{toctree}
:hidden:

guide/images
guide/normalization
guide/outputs
guide/recipes
guide/linear
guide/resampling
guide/gpu
```
