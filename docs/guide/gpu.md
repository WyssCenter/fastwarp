# GPU and memory

FastWarp runs its registrations on the GPU through [wgpu](https://wgpu.rs), which talks to the
graphics driver through Vulkan, Metal or DirectX 12. This page explains which GPU it uses, how
much memory a registration needs, and how to fall back to ANTs when the GPU is missing or too
small.

## Requirements

- A hardware GPU with a driver for one of the graphics APIs below. Discrete and integrated GPUs
  both work; the measured examples in [Performance](../performance.md) include an NVIDIA
  RTX A4000, an Apple M5 Pro, an Intel Lunar Lake integrated GPU, an AMD Radeon Pro 5500 XT and
  an NVIDIA Quadro K620.
- Enough memory for the registration grid (see [What fits](#what-fits)).
- Software adapters, which emulate a GPU on the CPU (Mesa's lavapipe, Microsoft WARP), are
  refused unless `FASTWARP_GPU_ALLOW_SOFTWARE=1` is set. They work but are slow; the tests use
  them on machines without a GPU.

The repository does not record minimum driver versions. If {func}`~fastwarp.gpu_info` raises
{class}`~fastwarp.GpuUnavailableError` on a machine with a GPU, update the graphics driver first;
the error message lists each graphics API tried, how many adapters it found, and the filters
that excluded them.

## Which GPU is used

| Operating system | Graphics APIs tried, in order |
| --- | --- |
| Linux | Vulkan |
| Windows | Vulkan, then DirectX 12 |
| macOS | Metal |

Among the adapters of an API, FastWarp prefers a discrete GPU, then an integrated one, then a
virtual one. If creating the device fails on one API, the next one in the list is tried.

To choose yourself:

- `FASTWARP_GPU_BACKEND` selects the API: `vulkan`, `dx12`, `metal` or `auto` (the default,
  the order above). An unknown value raises {class}`~fastwarp.GpuUnavailableError`.
- `FASTWARP_GPU_ADAPTER`, or the `adapter=` argument of every GPU function, selects the adapter
  whose name contains the given text, ignoring case. The argument takes precedence over the
  variable.

For example, on a machine with an Intel Arc GPU:

```python
import fastwarp

print(fastwarp.gpu_info(adapter="arc")["name"])
```

```text
Intel(R) Arc(TM) 140V GPU (16GB)
```

With a name that matches no adapter, the call raises {class}`~fastwarp.GpuUnavailableError`.

All environment variables are listed under [Environment variables](../environment.md).

## The memory budget

Before a registration starts, FastWarp compares the memory it will need with a budget:

1. The graphics API reports how much device memory is free: the memory budget of
   `VK_EXT_memory_budget` on Vulkan (or, without that extension, the size of the device-local
   heap), the local video memory budget on DirectX 12, and the recommended working set on Metal.
2. On an integrated GPU, which shares its memory with the CPU (for example Apple silicon or
   Intel Lunar Lake), and on a software adapter, this figure is halved.
3. 1 GiB is subtracted and kept free for the desktop and other programs.

`FASTWARP_GPU_MEMORY_BUDGET_MB` replaces this budget with a number of mebibytes; nothing is
subtracted from it. When the API reports no memory figure and the variable is unset, the budget
is the device's largest buffer size, at most 4 GiB, minus 1 GiB, and FastWarp prints a note.

{func}`~fastwarp.gpu_info` reports the budget as `memory_budget_bytes`.

## What fits

The memory a {func}`~fastwarp.register_syn` run needs depends on the number of voxels of the
registration grid ({func}`~fastwarp.iso_grid_dims`) and on the number of channels:

| Channels | Bytes per voxel of the registration grid |
| --- | --- |
| 1 | 168 |
| 2 | 188 |

These figures come from the engine's capacity check (`crates/fastwarp-gpu/src/capacity.rs`),
which bounds the measured peak allocation of a SyN run. If a moving image has more voxels than
the fixed image at the same spacing, each extra voxel adds 4 bytes per channel. In addition, the
largest single buffer, the displacement field at 12 bytes per voxel, must fit the device's limit
on one storage buffer.

For the 1 mm MNI template that ANTsPy ships (182 × 218 × 182 voxels), at three grid spacings:

| `iso_mm` | Grid | Voxels | 1 channel | 2 channels |
| --- | --- | --- | --- | --- |
| 2.0 | 91 × 109 × 91 | 0.9 million | 0.14 GiB | 0.16 GiB |
| 1.0 | 182 × 218 × 182 | 7.2 million | 1.13 GiB | 1.26 GiB |
| 0.5 | 363 × 435 × 363 | 57.3 million | 8.97 GiB | 10.04 GiB |

With the 1 GiB reserve, a 1 mm registration of one channel on this template needs a discrete
GPU that reports a little over 2 GiB of free memory. The measurements on the
[Performance](../performance.md) page used a different template, MNI 2009b, on a 1 mm grid of
197 × 233 × 189 voxels (8.7 million): there, one channel needs 1.4 GB and two channels 1.6 GB.
On a 2 GB NVIDIA Quadro K620, the two-channel registration was refused, and the one-channel one
ran only with `FASTWARP_GPU_MEMORY_BUDGET_MB=1900`.

## Check before you run

{func}`~fastwarp.fits_gpu` answers whether a registration of `fixed` at `iso_mm` with `channels`
channels fits the GPU, without running it. It returns `False` when there is no usable GPU.

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
for iso_mm in (2.0, 1.0, 0.5):
    print(iso_mm, fastwarp.fits_gpu(fixed, iso_mm=iso_mm, channels=1))
```

On the laptop of the other examples (Intel Arc 140V, memory budget 7.5 GiB) this printed:

```text
2.0 True
1.0 True
0.5 False
```

{func}`~fastwarp.fits_gpu` checks the fixed image's grid only. {func}`~fastwarp.register_syn`
repeats the check with the moving images as well, so it can still raise
{class}`~fastwarp.GpuCapacityError` when a moving image has many more voxels than the fixed one.

The [linear fits](linear.md) have no capacity check: a session that does not fit fails with
{class}`~fastwarp.RegistrationError` when the GPU runs out of memory.

## Fall back to ANTs

{class}`~fastwarp.GpuUnavailableError` and {class}`~fastwarp.GpuCapacityError` are raised before
any work, so a pipeline can catch them and run ANTs instead:

```python
import fastwarp


def normalize(fixed, moving, out_prefix, run_ants):
    try:
        return fastwarp.register_syn(fixed, moving, out_prefix)
    except (fastwarp.GpuUnavailableError, fastwarp.GpuCapacityError) as error:
        print(f"GPU registration not possible ({error}); running ANTs")
        return run_ants(fixed, moving, out_prefix)
```

`run_ants` stands for your own ANTs call, for example {func}`ants.registration` or
`antsRegistration` with the [same schedule](recipes.md#the-same-schedule-in-ants). Other
failures raise {class}`~fastwarp.RegistrationError`; see [Errors](../errors.md).

## Waiting for the GPU

Each wait for submitted GPU work is limited to 300 seconds; a longer wait fails with
{class}`~fastwarp.RegistrationError`. `FASTWARP_GPU_WAIT_TIMEOUT_S` changes the limit, for
example on a slow or busy GPU.
