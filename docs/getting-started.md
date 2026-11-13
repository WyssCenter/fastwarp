# Getting started

Three steps: install the wheel, check that FastWarp finds a GPU, and run a first registration.

```bash
pip install fastwarp antspyx
python -c "import fastwarp; print(fastwarp.gpu_info())"
```

Then follow the [quickstart](quickstart.md). If the second command raises
`GpuUnavailableError`, see [Troubleshooting](troubleshooting.md#no-gpu-is-found).

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
