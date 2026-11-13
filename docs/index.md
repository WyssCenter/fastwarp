---
html_theme.sidebar_secondary.remove: true
---

# FastWarp

**FastWarp registers 3-D images on the GPU.** It runs the rigid, affine and SyN registration of
[ANTs](https://github.com/ANTsX/ANTs) on a graphics card and writes the same transform files as
ANTs, so the rest of an ANTs-based pipeline can stay as it is.

FastWarp is a Python package with a compiled Rust engine. It is meant for people who build or run
neuroimaging pipelines, for example to normalize brain MRI onto a template, and who use ANTs for
that today.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} What it does
:class-card: sd-shadow-none

- A full normalization in one call: centre of mass, Rigid, Affine, then SyN.
- Rigid or affine fits of one image pair, several in a row without uploading the images again.
- Resampling and field composition like ANTs' `antsApplyTransforms`, on the CPU.
:::

:::{grid-item-card} What it works with
:class-card: sd-shadow-none

- GPUs through [wgpu](https://wgpu.rs): Vulkan (Linux, Windows), Metal (macOS) and DirectX 12
  (Windows).
- Python 3.11 or later and NumPy 2.0 or later. [ANTsPy](https://github.com/ANTsX/ANTsPy) is
  optional.
- Transform files that ANTs and ANTsPy read, such as `Composite.h5`.
:::
::::

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
moving = ants.image_read(ants.get_ants_data("ch2"))

info = fastwarp.register_syn(fixed, moving, "out/ch2_to_mni_")
warped = ants.apply_transforms(fixed, moving, ["out/ch2_to_mni_Composite.h5"])
```

## Where to go next

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {fas}`download` Installation
:link: installation
:link-type: doc

Install the wheel from PyPI, check that FastWarp finds your GPU, or build from source.
:::

:::{grid-item-card} {fas}`play` Quickstart
:link: quickstart
:link-type: doc

Register one brain image onto a template with public sample data.
:::

:::{grid-item-card} {fas}`book-open` User guide
:link: user-guide
:link-type: doc

Images, normalization, output files, recipes, linear fits, resampling and GPU memory.
:::

:::{grid-item-card} {fas}`code` API reference
:link: api
:link-type: doc

Every public function, class, constant and exception, with its signature.
:::

:::{grid-item-card} {fas}`gauge-high` Performance and validation
:link: validation
:link-type: doc

Measured run times, the accuracy benchmark against ANTs, and how the code is tested.
:::

:::{grid-item-card} {fas}`users` Project
:link: project
:link-type: doc

Changelog, contributing, citing, licence and related projects.
:::
::::

## Background

FastWarp was developed by Richard Köhler (Wyss Center for Bio and Neuroengineering) and Jan
Philipp Payonk (University of Rostock) as the registration engine of
[Wyss DBS Localizer](https://github.com/WyssCenter/dbs-localizer), a tool for localizing deep
brain stimulation (DBS) electrodes in brain images. The engine
ports the registration code of ANTs and [ITK](https://itk.org) to GPU compute shaders: the Mattes
mutual information and cross-correlation metrics, the conjugate-gradient linear stages, and SyN
with Gaussian field regularisation and field inversion. Before a normalization starts, FastWarp
checks whether a GPU is present and has enough memory, so that a pipeline can fall back to ANTs
when it does not.

FastWarp is research software, licensed under the [Apache License 2.0](licence.md). If you use
it in your work, please [cite it](citing.md).

```{toctree}
:hidden:

getting-started
user-guide
reference
validation
project
```
