# Changelog

## Unreleased

- A command line, `fastwarp` (also `python -m fastwarp`), for NIfTI files: `register`
  (`register_syn`), `linear` (one rigid or affine fit), `apply` (`apply_transforms`), `compose`
  (`compose_field`), `recipe` (the default recipe as JSON), `gpu-info`, `fits` and `bench` (timings
  on built-in synthetic volumes or on your images, with a description of the machine). Images are
  read as ITK and ANTsPy read them. Distinct exit status for invalid arguments (2), file errors (3),
  no usable GPU (4) and a grid too large for the GPU (5); `--json` for machine-readable output.
  See `docs/cli.md`. The Python extension and the standalone program built from the new crate
  `fastwarp-cli` run the same code.
- Each GitHub release carries the standalone `fastwarp` program, which needs no Python, for Linux
  x86_64 and aarch64 (glibc 2.28 or later), macOS arm64 and x86_64 (macOS 11 or later) and Windows
  x86_64 (static C runtime), as `fastwarp-<target>.tar.gz` / `.zip` with a `.sha256` file each,
  and the installer scripts `fastwarp-installer.sh` (Linux, macOS) and `fastwarp-installer.ps1`
  (Windows). The installer of a release installs that release and holds the SHA-256 of its
  archives: it refuses a download that does not match, installs without administrator rights into `~/.local/bin` (Windows:
  `%LOCALAPPDATA%\Programs\fastwarp\bin`), adds that folder to PATH unless told not to
  (`--no-modify-path`, `-NoModifyPath`), and removes both with `--uninstall` (`-Uninstall`). Every
  release file has a GitHub build provenance attestation (`gh attestation verify`). See
  `docs/cli.md`.
- `fits_gpu` takes the moving image(s) as an optional second argument and checks the moving grid
  as `register_syn` does; without it a moving image larger than the fixed one could pass
  `fits_gpu` and still raise `GpuCapacityError`. `channels` defaults to the number of fixed
  images, and invalid arguments (image counts, `channels` other than 1 or 2, `iso_mm`) raise
  `ValueError` as in `register_syn` instead of returning a result.
- The benchmark (`benchmarks/accuracy`) now also measures speed in the same run, on the same
  public data: every registration records its Linear, SyN and total time for FastWarp and for
  ANTs (read from `antsRegistration --verbose 1`); `linear` times `linear_session` fits
  (rigid, affine, a rigid multi-start) against `antsRegistration` with the same stages, with
  their landmark errors; `post` times `compose_field` and `apply_transforms` against
  `antsApplyTransforms` on uncompressed outputs (`--gzip` adds `.nii.gz`). `report` tabulates
  the median and quartiles of each, per dataset and registration grid, with the machine and
  software they ran on (CPU, GPU and its driver name, OS, fastwarp and ANTs versions, the thread
  setting and core counts of each command). Both engines use all cores unless `--threads`
  caps both alike, and the benchmark no longer leaves `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` set
  in the calling process.
- `docs/performance.md` explains how the speed is measured and how to reproduce it; the
  earlier figures, measured on non-public data, are labelled as such with what is known about
  them.
- An accuracy benchmark that reviewers can run (`benchmarks/accuracy`, `docs/benchmarks.md`):
  FastWarp and ANTs register the same pairs with the same recipe on public data with manual
  cortical, subcortical and deep-nuclei labels and anatomical fiducials; every download is checked
  against a pinned SHA-256. A `benchmark` dependency group adds SciPy and Matplotlib (already
  locked as ANTsPy dependencies). No results yet.
- README badges: CI, PyPI version, Python versions, licence, documentation.
- `dir(fastwarp)` lists only the public API (`__all__`); the implementation moved to the private
  module `fastwarp._api`, and the public Python functions report `fastwarp` as their module.
- `LinearSession.register` creates the folder of `out_prefix`, as `register_syn` and
  `register_rigid` do.
- An `out_prefix` that ends in a path separator (`/`, and on Windows also `\`) names a folder, as
  in ANTs: `register_syn(..., out_prefix="out/sub/")` writes `out/sub/Composite.h5` (was
  `out/subComposite.h5`), and the folder `out/sub` is created. The same holds for
  `LinearSession.register` and the deprecated `register_rigid` (#28). A prefix of `.` or `..` is
  text too, as in ANTs (`.` gives `.Composite.h5`): use `./` for the current folder. Breaking for
  Rust callers of the engine crate `fastwarp-reg`: `transform::ants_io::with_suffix` (and so
  `write_outputs`, `write_linear_output` and `write_composite_pair` / `read_composite_pair`)
  appends the suffix to the prefix as text instead of to its file name, and `fastwarp_reg_syn
  --out` follows the same rule.
- `GpuCapacityError`, `GpuUnavailableError`, `RegistrationCancelled` and `RegistrationError`
  report `fastwarp` as their module (was `_core`, which cannot be imported), so they pickle and
  cross process boundaries, for example from a `concurrent.futures.ProcessPoolExecutor` worker.
  `gpu_info` reports `fastwarp` as well (#26).

### Deprecated

- `register_rigid`, `rigid_session` and `RigidSession` still work but emit a `DeprecationWarning`
  and will be removed in fastwarp 0.3.0; use `linear_session(...).register()`,
  `linear_session` and `LinearSession`. They are no longer in `fastwarp.__all__` or `dir(fastwarp)`.

## 0.1.0 (2026-11-12)

First public release.

- `register_syn`: ANTs-style registration on the GPU (centre of mass, Rigid, Affine, SyN) of one
  image pair, or of two channels such as T1w + T2w, onto a fixed image. It writes ANTs
  `Composite.h5`, `InverseComposite.h5` and `0GenericAffine.mat`, which `antsApplyTransforms`
  reads. The schedule is the built-in recipe (`cohort_recipe(channels)`) or one the caller passes
  as `recipe=`; the result reports the recipe that ran and the time of each phase. Images are
  ANTsPy images or `(array, spacing, origin, direction)` tuples in ITK (LPS) geometry, so ANTsPy
  is optional; the registration runs on an isotropic grid of `iso_mm` (default 1 mm), whose size
  `iso_grid_dims` reports.
- `linear_session` / `LinearSession`: rigid or affine fits on Mattes mutual information from any
  start, with an optional fixed-image mask and winsorizing. A session uploads the image pair once
  and keeps its smoothed copies between fits, so a multi-start search costs little more than one
  fit. `register_rigid` is a one-off rigid fit; `rigid_session` / `RigidSession` are aliases of
  `linear_session` / `LinearSession`.
- `compose_field` and `apply_transforms`: write a list of ANTs transform files (`.h5`, `.mat`,
  `.txt`, displacement fields) as one displacement field (`antsApplyTransforms --compose`;
  `compression_level=0` writes stored gzip, which ANTs reads back quickly), and resample images
  with the linear, nearest or Lanczos windowed-sinc interpolator, in parallel on the CPU.
- GPU checks before any work: `gpu_available`, `gpu_info`, `fits_gpu`, and a capacity check that
  raises `GpuCapacityError` when a registration would not fit the GPU's memory budget
  (`FASTWARP_GPU_MEMORY_BUDGET_MB` overrides it). Failures raise `GpuUnavailableError`,
  `GpuCapacityError` or `RegistrationError` (all `RuntimeError`), an internal panic included, so
  a caller can fall back to ANTs; a cancelled run raises `RegistrationCancelled`; invalid
  arguments raise `ValueError` before any work.
- `CancelToken` cancels a running registration from another thread; the GIL is released while
  the engine runs.
- Runs on Vulkan, Metal and DirectX 12 through wgpu; `FASTWARP_GPU_BACKEND` and
  `FASTWARP_GPU_ADAPTER` choose the API and the device. A hardware GPU is required: software
  adapters (lavapipe, WARP) are refused unless `FASTWARP_GPU_ALLOW_SOFTWARE=1`. The README lists
  every environment variable.
- The HDF5 transform reader rejects malformed or crafted files with an error instead of
  exhausting memory or aborting the process.
- Wheels for Linux (x86_64, aarch64; manylinux 2.28), macOS (arm64, x86_64) and Windows
  (x86_64): one `abi3` wheel per platform for Python 3.11 and later, with NumPy 2.0 or later.
- Apache-2.0; parts are ports of ITK and ANTs, and the sample-point generator derives from
  BSD-3-Clause Mersenne Twister code; all are listed in `NOTICE` and `docs/THIRD_PARTY_PORTS.md`.
- The engine was developed inside Wyss DBS Localizer under the name `dbsloc-gpureg`, which was
  never published. Code written against it imports `fastwarp` instead, with the same Python API,
  on Python 3.11 or later and NumPy 2.0 or later. The former `DBSLOC_*` environment variables are
  still read when the `FASTWARP_*` name is unset (a `FASTWARP_*` name set to an empty value
  wins). `ENGINE_SOURCE` (`"fastwarp-legacy 90c1df3"`) names the revision of that earlier
  standalone copy of the engine, not a revision of this repository.
