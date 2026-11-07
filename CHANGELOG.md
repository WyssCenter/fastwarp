# Changelog

## Unreleased

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
