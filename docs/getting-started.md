# Getting started

Three steps: install the wheel, check that FastWarp finds a GPU, and run a first registration,
from Python or from the command line.

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
pip install fastwarp antspyx
python -c "import fastwarp; print(fastwarp.gpu_info())"
```
:::

:::{tab-item} uv
:sync: uv

```bash
uv add fastwarp antspyx
uv run python -c "import fastwarp; print(fastwarp.gpu_info())"
```
:::

::::

ANTsPy (`antspyx`) publishes no wheels for Python 3.14 yet, on Windows or elsewhere: use Python
3.11, 3.12 or 3.13 if you want it. FastWarp itself also runs on 3.14, and its command line reads
NIfTI files without ANTsPy. If the second command raises `GpuUnavailableError`, see
[Troubleshooting](troubleshooting.md#no-gpu-is-found).

## A first registration

::::{tab-set}

:::{tab-item} Python

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
moving = ants.image_read(ants.get_ants_data("ch2"))

fastwarp.register_syn(fixed, moving, "out/sub_")
warped = fastwarp.apply_transforms(moving, fixed, ["out/sub_Composite.h5"])
```

The [quickstart](quickstart.md) explains each step.
:::

:::{tab-item} Command line

```bash
fastwarp register fixed.nii.gz moving.nii.gz -o out/sub_
fastwarp apply -r fixed.nii.gz -i moving.nii.gz -o warped.nii.gz -t out/sub_Composite.h5
```

`fixed.nii.gz` and `moving.nii.gz` stand for your two images. The first command writes
`out/sub_Composite.h5` and the other [output files](guide/outputs.md); the second writes the
moving image resampled onto the fixed image's grid. [Command line](cli.md) lists every option.
The command line registers on a grid with the fixed image's smallest voxel spacing, where the
Python function uses 1 mm unless told otherwise.
:::

::::

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`download` Installation
:link: installation
:link-type: doc

Requirements, wheels for each platform, building from source and checking the GPU.
:::

:::{grid-item-card} {fas}`play` Quickstart
:link: quickstart
:link-type: doc

A full normalization of a sample brain onto a template, resampling and a rigid fit.
:::

:::{grid-item-card} {fas}`life-ring` Troubleshooting
:link: troubleshooting
:link-type: doc

No GPU found, too little GPU memory, timeouts, and differences from ANTs.
:::
::::

```{toctree}
:hidden:

installation
quickstart
troubleshooting
```
