# Errors

FastWarp raises six exceptions of its own, and the standard `ValueError` and
`FileNotFoundError` for mistakes in the arguments. [Troubleshooting](troubleshooting.md) says
what to do about the most common ones.

| Exception | Base class | Meaning |
| --- | --- | --- |
| {class}`~fastwarp.GpuUnavailableError` | `RuntimeError` | No usable GPU. Raised before any work. |
| {class}`~fastwarp.AdapterNotFoundError` | {class}`~fastwarp.GpuUnavailableError` | The requested adapter name matches no GPU adapter. Raised before any work, whatever the engine. |
| {class}`~fastwarp.GpuCapacityError` | `RuntimeError` | The registration does not fit the GPU. Raised before any work. |
| {class}`~fastwarp.HostMemoryError` | `RuntimeError` | On the CPU engine, the registration does not fit the computer's available memory. Raised before any work. |
| {class}`~fastwarp.RegistrationError` | `RuntimeError` | The engine failed while it worked. |
| {class}`~fastwarp.RegistrationCancelled` | `Exception` | The run was stopped through its {class}`~fastwarp.CancelToken`. |
| `ValueError` | | An argument or the recipe is wrong. Raised before any work. |
| `FileNotFoundError` | | A transform file does not exist. Raised before any work. |

All six FastWarp exceptions can be pickled, so they also cross process boundaries, for example
out of a {class}`concurrent.futures.ProcessPoolExecutor` worker.

## GpuUnavailableError

Raised by every function that opens the GPU when:

- no hardware adapter is found for the graphics APIs tried (see
  [Which GPU is used](guide/gpu.md#which-gpu-is-used));
- only a software adapter is present and `FASTWARP_GPU_ALLOW_SOFTWARE` is not set;
- `FASTWARP_GPU_BACKEND` has an unknown value;
- the driver fails to create the device, or the GPU fails the short self-test of the kernels
  that FastWarp runs when it opens the device.

The message lists what was tried, for example:

```{code-block} text
:class: fw-output
no usable GPU adapter: Backends(VULKAN): 1 adapter(s), none eligible; Backends(DX12): 2 adapter(s), none eligible; software adapters not allowed (FASTWARP_GPU_ALLOW_SOFTWARE=1)
```

{func}`~fastwarp.gpu_available` and {func}`~fastwarp.fits_gpu` return `False` in these cases
instead of raising. [Troubleshooting](troubleshooting.md#no-gpu-is-found) explains each part of
the message.

## AdapterNotFoundError

Raised by every function that opens the GPU when `adapter=` (`--adapter`) or
`FASTWARP_GPU_ADAPTER` names no adapter that exists: no adapter of the graphics APIs tried has a
name that contains the text, ignoring case. It is an error for every engine choice:
`engine="auto"` does not fall back to the CPU engine for it, `fastwarp.ants` with
`fallback="ants"` does not fall back to ANTs, and {func}`~fastwarp.gpu_available` and
{func}`~fastwarp.fits_gpu` raise it instead of returning `False`. The message lists the adapters
found and, when one is close (a small typo, other spacing, the words in another order), suggests
it:

```{code-block} text
:class: fw-output
the adapter name 'nvidai rtx' matches no GPU adapter; adapters found: NVIDIA RTX PRO 6000 Blackwell (Vulkan), Microsoft Basic Render Driver (Dx12); did you mean 'NVIDIA RTX PRO 6000 Blackwell'?
```

It is a subclass of {class}`~fastwarp.GpuUnavailableError`, so code that catches that one still
catches it. An adapter that exists but cannot be used (a software adapter without
`FASTWARP_GPU_ALLOW_SOFTWARE=1`, a failed self-test) raises
{class}`~fastwarp.GpuUnavailableError` itself.

## GpuCapacityError

Raised by {func}`~fastwarp.register_syn` when the recipe does not fit the GPU: the SyN stage
needs more GPU memory than the budget, or a buffer larger than the device allows, or the linear
stages do (their message starts with "the linear stages on the registration grid"). Each phase is
checked on its own. The message gives the grid, the bytes needed and the budget. Use a coarser `iso_mm`, a GPU with more memory, or ANTs. See
[What fits](guide/gpu.md#what-fits) and
[Troubleshooting](troubleshooting.md#the-registration-does-not-fit).

## HostMemoryError

Raised by {func}`~fastwarp.register_syn`, {func}`~fastwarp.register_atlases` and
{func}`fastwarp.ants.registration` on the [CPU engine](guide/cpu.md) when the estimated memory of
the registration is more than the memory the computer has available ({func}`~fastwarp.linear_session`
does not check it). The message gives the estimate, the available memory and the grid.
Use a coarser `iso_mm`, free memory, or the GPU engine. With `engine="auto"` it is raised only
when there is no usable GPU.

## RegistrationError

Raised while the engine works, for example when:

- an input image has NaN or infinite voxels;
- the GPU runs out of memory, is reset by the driver, or does not finish within
  `FASTWARP_GPU_WAIT_TIMEOUT_S`;
- a transform file cannot be read: it is damaged, or its extension is not one of those listed
  under [Resampling](guide/resampling.md#transform-lists);
- an output file cannot be written.

An internal error of the engine (a Rust panic) is also raised as `RegistrationError`, not as
PyO3's `PanicException`. So `except RuntimeError` catches `GpuUnavailableError`,
`GpuCapacityError` and `RegistrationError`, but not a cancellation.

## RegistrationCancelled

Raised by {func}`~fastwarp.register_syn` and {meth}`~fastwarp.LinearSession.register` when the
{class}`~fastwarp.CancelToken` passed as `cancel=` is cancelled, before or during the run. No
files are written. It derives from `Exception`, not from `RuntimeError`, so that a handler for
failures does not catch a cancellation by accident.

## ValueError and FileNotFoundError

These come from the arguments and are raised before any registration work starts:

| Function | Raised when |
| --- | --- |
| all | An image is not 3D; a geometry is invalid. |
| {func}`~fastwarp.register_syn` | No fixed images, or not as many moving as fixed images; `iso_mm` is not a positive number; `outputs` is not one of {data}`~fastwarp.OUTPUT_LAYOUTS`; the recipe has an unknown key, a value of the wrong type, lists of different lengths, a value outside its [limits](guide/recipes.md), an empty or misordered stage list, keys of both recipe forms, a start file that does not exist or a start affine that cannot be inverted; several channels without a SyN stage. |
| {func}`~fastwarp.fits_gpu` | The same image counts, `iso_mm` and recipe checks as `register_syn`; `channels` is below 1, or differs from the number of fixed images in a list. |
| {func}`~fastwarp.cohort_recipe` | `channels` is below 1. |
| {func}`~fastwarp.linear_session` | `winsorize` is not `(lower, upper)` with `0 <= lower < upper <= 1`. |
| {meth}`~fastwarp.LinearSession.register` | `transform` is not `"rigid"` or `"affine"`; the stage is wrong as for a recipe; `initial` does not hold 12 parameters and 3 centre values; a `mask` is given with `use_mask=False`; the session is closed or busy with another fit. |
| {func}`~fastwarp.apply_transforms` | `interpolation` is not one of {data}`~fastwarp.INTERPOLATORS`; the transform list is empty; `invert` does not hold one `bool` per file, or inverts a displacement field; a transform file does not exist (`FileNotFoundError`). |
| {func}`~fastwarp.compose_field` | `compression_level` is above 9 (a negative value raises `OverflowError`); the transform list is empty; `invert` as for `apply_transforms`; a transform file does not exist (`FileNotFoundError`). The transforms are checked before the output folder is created. |

The message names the argument or the recipe field, for example
`recipe['rigid']['bins']: the GPU MI kernels support 32 bins, not 64`.

## Command line

The [command line](cli.md) reports the same errors as exit status: 2 for invalid arguments, 3
for a file error, 4 for {class}`~fastwarp.GpuUnavailableError`, 5 for
{class}`~fastwarp.GpuCapacityError`, 7 for {class}`~fastwarp.HostMemoryError`, 8 for
{class}`~fastwarp.AdapterNotFoundError`, 1 for other failures. See
[Exit status](cli.md#exit-status).
