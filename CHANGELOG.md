# Changelog

## 0.1.0 (unreleased)

- First version: the dbs-localizer GPU registration engine (commit `00faf42`) as a Python
  wheel. `register_syn` runs the cohort recipe (centre of mass, Rigid, Affine, SyN; one or two
  channels) and writes ANTs `Composite.h5` / `InverseComposite.h5` / `0GenericAffine.mat`.
- GPU capacity check before any work (`GpuCapacityError`), `fits_gpu`, `gpu_info`,
  `gpu_available`; cancellation from another thread (`CancelToken`); the GIL is released while
  the engine runs.
- Wheels for macOS (arm64, x86_64), Linux (x86_64, aarch64, manylinux 2.28) and Windows
  (x86_64); one abi3 wheel per platform for Python ≥ 3.10.
