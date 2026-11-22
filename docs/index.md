---
html_theme.sidebar_secondary.remove: true
---

# FastWarp

**FastWarp registers 3D images on the GPU.** It runs the rigid, affine and {term}`SyN`
registration of [ANTs](https://github.com/ANTsX/ANTs) on a graphics card and writes the same
transform files, so the rest of an ANTs-based pipeline can stay as it is.

It ships as both a [Python package](installation.md#from-pypi) and a [command-line interface](installation.md#command-line) and is meant for people who build or run neuroimaging pipelines, for example to normalize brain MRI onto a template.

**FastWarp** is shipped with [Lead-DBS v4.0](https://www.lead-dbs.org), the de-facto standard for localizing deep brain stimulation electrodes, and is also the registration engine of
[Wyss DBS Localizer](https://github.com/WyssCenter/dbs-localizer).

```{button-ref} browser
:color: primary
:align: center
:class: fw-cta

Try FastWarp in your browser now
```

```{raw} html
<figure class="fw-hero">
  <img src="_static/figures/registration-checkerboard.png" width="990" height="396"
       alt="Three axial brain slices: the MNI template; a checkerboard of the template and a subject's scan before registration, with edges that do not line up; and the same checkerboard after registration, with continuous edges.">
  <figcaption><span>Template (fixed image)</span><span>Before: template and subject in a checkerboard</span><span>After FastWarp</span></figcaption>
</figure>
```

:::{div} fw-caption
A 7 T scan of one subject of the public [SNSX dataset](https://openneuro.org/datasets/ds004470)
registered onto the MNI 2009c template with the default settings
([how the figure is made](guide/normalization.md#the-figure-on-the-home-page)).
:::

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} What it does
:class-card: sd-shadow-none

- A full normalization in one call: {term}`centre of mass`, Rigid, Affine, then SyN.
- Rigid or affine fits of one image pair, several in a row on images prepared once.
- Resampling and field composition like ANTs' `antsApplyTransforms`, on the CPU.
:::

:::{grid-item-card} What it works with
:class-card: sd-shadow-none

- GPUs through [wgpu](https://wgpu.rs): Vulkan (Linux, Windows), Metal (macOS) and DirectX 12
  (Windows).
- Python 3.11 or later.
- Transform files that ANTs and [ANTsPy](https://github.com/ANTsX/ANTsPy) read, such as `Composite.h5`.
:::

:::{grid-item-card} How to use it
:class-card: sd-shadow-none

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
moving = ants.image_read(ants.get_ants_data("ch2"))

info = fastwarp.register_syn(fixed, moving, "out/ch2_to_mni_")
warped = ants.apply_transforms(fixed, moving, ["out/ch2_to_mni_Composite.h5"])
```

:::
::::

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

:::{grid-item-card} {fas}`globe` Try it in your browser
:link: browser
:link-type: doc

Run a registration on your GPU through WebGPU, without installing anything or uploading images.
:::

:::{grid-item-card} {fas}`book-open` User guide
:link: user-guide
:link-type: doc

How it works, images, output files, recipes, linear fits, resampling and GPU memory.
:::

:::{grid-item-card} {fas}`code` API reference
:link: api
:link-type: doc

Every public function, class, constant and exception, with its parameters.
:::

:::{grid-item-card} {fas}`gauge-high` Performance and validation
:link: validation
:link-type: doc

How speed and accuracy are measured, and what the tests check today.
:::

:::{grid-item-card} {fas}`life-ring` Troubleshooting
:link: troubleshooting
:link-type: doc

No GPU found, too little GPU memory, and why results differ slightly from ANTs.
:::
::::

FastWarp is research software, licensed under the [Apache License 2.0](licence.md). If you use it in your work, please
[cite it](citing.md).

```{toctree}
:hidden:

getting-started
user-guide
reference
validation
project
```
