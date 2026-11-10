# Changelog

## Unreleased

- Licensed under Apache-2.0 (`LICENSE`); the wheel and the crates declare the licence and ship
  `LICENSE` and `NOTICE`.
- The `Private :: Do Not Upload` classifier is gone, so the wheels can go to PyPI.
- README timings measured on five GPUs (NVIDIA RTX A4000, Apple M5 Pro, Intel Lunar Lake, AMD
  Radeon Pro 5500 XT, NVIDIA Quadro K620 on Windows).
- Engine updated to WyssCenter/fastwarp `90c1df3`: wgpu 30, ndarray 0.17, nifti 0.18; the
  `dbsloc-matfile` tests carry their own data; the ANTs comparison tests read the programs from
  `DBSLOC_ANTS_DIR`. `ENGINE_SOURCE` is now `"fastwarp 90c1df3"`.
- Licence labels in REUSE form: an SPDX header in every source file, `REUSE.toml` for the rest,
  `LICENSES/Apache-2.0.txt`; `NOTICE` carries the ITK notice; `THIRD_PARTY_PORTS.md` moved to
  `docs/`.

## 0.4.0 (2026-11-09)

- `linear_session` / `LinearSession` (was `rigid_session` / `RigidSession`, still available under
  those names): a fit can be an `Affine` stage (`register(transform="affine")`), started as the
  identity centred on `initial`, as ANTs centres a stage after its initial transform, and can
  bring its own fixed mask (`register(mask=...)`), for a template alignment (rigid, then affine)
  and a brain-shift refinement (an affine fit restricted to a subcortical region). Against ANTs on
  phantoms: an affine fit from the centres of mass within 0.5 mm of ANTs', a masked one no
  farther from the truth than ANTs' (`tests/test_linear.py`). On the Lead-Tutor cohort, as
  leaddbs-py's brain-shift correction, the masked fit lands 0.19 mm from ANTs' (as close as two
  ANTs runs land to each other) in 2.4-4.8 s instead of 5-11 s. `check_linear_stage` replaces
  `check_rigid_stage`.

## 0.3.0 (2026-11-08)

- `rigid_session` / `RigidSession` and `register_rigid`: rigid registration on the GPU (one
  `Rigid` stage on Mattes MI from any start, an optional fixed-image mask, winsorizing), for
  coregistration. A session prepares and uploads the image pair once and keeps its smoothed
  copies between fits, so a multi-start of five coarse and two full fits costs little more than
  one: a post-op CT onto a T1w anchor in 17 s on a laptop GPU where ANTs takes 32 s, the fits
  within 0.7 mm of ANTs' over a head-sized cube. `check_rigid_stage` checks a stage without a GPU.

## 0.2.1 (2026-11-07)

- Fix: on Linux with an NVIDIA GPU, the GPU disappeared after ten registrations (or `gpu_info`
  calls) in one process: every call created and destroyed a Vulkan instance, the loader unloaded
  NVIDIA's driver library each time, and from the eleventh load on it no longer provided
  `vkCreateInstance`. A cohort run in one process fell back to the CPU after ten subjects. One idle
  Vulkan instance now lives for the whole process, so the driver stays loaded (no device, no GPU
  memory). Measured on an RTX A4000, driver 580: 30 `gpu_info` calls and 25 registrations in one
  process, all on the GPU.

## 0.2.0 (2026-11-07)

- `register_syn(recipe=...)`: the caller sets the whole schedule (Rigid, Affine, SyN: iterations,
  shrink factors, smoothing, metric parameters, channel weights) instead of the built-in cohort
  recipe; `cohort_recipe(channels)` returns that recipe in the same form, and the result reports
  the recipe that ran. Left out, it is the cohort recipe as before, bit for bit. A recipe is
  checked before any work, including the GPU limits (32 MI bins, CC radius at most 8).
- `compose_field` and `apply_transforms`: what a caller does with the composites after the
  registration (`antsApplyTransforms --compose`, and resampling with the linear, nearest or
  Lanczos windowed-sinc interpolator), in parallel Rust: about 10x faster than ANTs, equal to
  `float32` rounding. `compression_level=0` writes the field as stored gzip, which ANTs reads back
  about 7x faster than its own compressed `float64` field.

## 0.1.1 (2026-11-07)

- float32 images are passed to the engine without copies (a view) and converted to float64
  once: about 130 MB instead of 660 MB extra memory per 0.5 mm template.
- An internal engine panic is raised as `RegistrationError` instead of `PanicException`, so
  callers that catch `Exception` can fall back.
- A missing output folder is reported before the registration runs; the Python wrapper creates
  it.
- The package version is read from `Cargo.toml` only; CI runs a RustSec dependency audit.
- New tests: determinism, float32 vs float64, NaN input, output folder, integer input.

## 0.1.0 (2026-11-06)

- First version: the dbs-localizer GPU registration engine (commit `00faf42`) as a Python
  wheel. `register_syn` runs the cohort recipe (centre of mass, Rigid, Affine, SyN; one or two
  channels) and writes ANTs `Composite.h5` / `InverseComposite.h5` / `0GenericAffine.mat`.
- GPU capacity check before any work (`GpuCapacityError`), `fits_gpu`, `gpu_info`,
  `gpu_available`; cancellation from another thread (`CancelToken`); the GIL is released while
  the engine runs.
- Wheels for macOS (arm64, x86_64), Linux (x86_64, aarch64, manylinux 2.28) and Windows
  (x86_64); one abi3 wheel per platform for Python ≥ 3.10.
