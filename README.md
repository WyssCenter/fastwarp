# dbsloc-gpureg

GPU image registration for Python: the SyN normalization engine of **Wyss DBS Localizer** by
**Richard Köhler** (Wyss Center for Bio and Neuroengineering), packaged as a wheel.

The engine is a port of ANTs/ITK's registration (Mattes mutual information and
cross-correlation metrics, conjugate-gradient linear stages, SyN with Gaussian field
regularisation and field inversion) to GPU compute shaders via [wgpu](https://wgpu.rs). It runs
on **Metal** (macOS), **Vulkan** (Linux, Windows) and **DX12** (Windows), from one wheel per
platform for every Python ≥ 3.10. Outputs are ANTs files (`Composite.h5`,
`InverseComposite.h5`, `0GenericAffine.mat`) that ANTsPy and `antsApplyTransforms` read as is.

> **Status: private.** The licence for the engine crates is being agreed with the Wyss Center
> (planned: Apache-2.0). Do not redistribute until `LICENSE` is in place. The files ported from
> ANTs/ITK carry their Apache-2.0 notices (`THIRD_PARTY_PORTS.md`).

## Speed

T1w + T2w to the MNI 2009b template (0.5 mm), 1 mm registration grid, Lead-Tutor sub-57245:

| Hardware | Linear | SyN | Registration total |
| --- | --- | --- | --- |
| Intel Lunar Lake integrated GPU (laptop, Vulkan) | 3.5 s | 14.0 s | 19.9 s |
| NVIDIA RTX A4000 (Vulkan) | – | – | ≈ 13 s |
| ANTs on 16 CPU threads, same recipe | | | 1000–1500 s |

Intel iMac (AMD Radeon Pro 5500 XT, Metal): 24 s. Apple Silicon timings: pending.

After the registration, turning the composites into displacement fields on the 0.5 mm template
and anchor grids and the Lanczos re-warp of the anchor (what Lead-DBS / leaddbs-py do next):

| Laptop, 8 threads | ANTs (ITK) | `compose_field` / `apply_transforms` |
| --- | --- | --- |
| Forward field (69 M voxels) | 36.5 s | 1.4 s |
| Inverse field (37 M voxels) | 20.9 s | 0.95 s |
| Lanczos re-warp | 31.0 s | 2.7 s |

## Install

From a release (wheels for macOS arm64 / x86_64, Linux x86_64 / aarch64, Windows x86_64):

```bash
gh release download v0.1.0 -R JPPayonk/dbsloc-gpureg -p "*.whl" -D wheels   # private repo: needs gh access
pip install --no-index --find-links wheels dbsloc-gpureg
```

From source (needs a Rust toolchain; `rust-toolchain.toml` pins the version):

```bash
pip install "git+ssh://git@github.com/JPPayonk/dbsloc-gpureg.git"
```

## Use

```python
import ants
import dbsloc_gpureg as gpureg

fixed = [ants.image_read("tpl_t1.nii"), ants.image_read("tpl_t2.nii")]
moving = [ants.image_read("sub_T1w.nii.gz"), ants.image_read("sub_T2w_in_T1w.nii.gz")]

if gpureg.fits_gpu(fixed[0], iso_mm=1.0, channels=2):
    info = gpureg.register_syn(fixed, moving, "out/sub_", iso_mm=1.0)
    warped = ants.apply_transforms(fixed[0], moving[0], ["out/sub_Composite.h5"])
    print(info["adapter"], f'{info["total_s"]:.1f} s')
else:
    ...  # no GPU or too little GPU memory: use ANTs
```

- `register_syn(fixed, moving, out_prefix, *, iso_mm=1.0, write_warped=False, adapter=None,
  cancel=None, recipe=None)`: one image or a list of two (T1w + T2w) on each side; images are `ANTsImage`s
  or `(array, spacing, origin, direction)` tuples in ITK (LPS) geometry. Writes
  `<out_prefix>Composite.h5` (`[Affine, Warp]`), `<out_prefix>InverseComposite.h5`
  (`[InverseWarp, Affine⁻¹]`) and `<out_prefix>0GenericAffine.mat`. Returns timings, the
  registration grid, per-level SyN diagnostics, the affine and peak GPU memory. Releases the GIL.
- `CancelToken()`: pass as `cancel=`, call `.cancel()` from another thread; the run stops at
  the next iteration with `RegistrationCancelled`.
- `compose_field(transforms, reference, out_path, *, compression_level=None)`: the transforms as
  one displacement field on `reference`'s grid, the file `antsApplyTransforms --compose` writes
  (`float32`, parallel gzip; `compression_level=0` stores the data as valid gzip without
  compressing it, which suits displacement fields).
- `apply_transforms(moving, reference, transforms, *, interpolation="linear", default_value=0.0)`:
  `antsApplyTransforms` with `linear`, `nearestNeighbor` or `lanczosWindowedSinc`; returns an
  `ANTsImage` like `reference` (or an array for a tuple reference). `transforms` is a list as
  `ants.apply_transforms` takes it. Both run on all CPU cores and need no GPU; they equal ANTs
  to `float32` rounding (`tests/test_post.py`).
- `linear_session(fixed, moving, *, fixed_mask=None, winsorize=None, adapter=None)`: linear fits
  of one image pair (Mattes MI, regular sampling, one `Rigid` or `Affine` stage per fit from any
  start), the pair prepared, winsorized and uploaded once and its smoothed copies kept between
  fits, for a multi-start or a rigid fit followed by an affine one.
  `session.register(stage=None, initial=None, use_mask=True, out_prefix=None, *,
  transform="rigid", mask=None)` returns the collapsed affine (`parameters`, `center`) and the
  final MI value; `initial` is `(12 parameters, centre)` of an ITK affine (`None` = centres of
  intensity mass), and the fitted transform starts as the identity centred there, as an ANTs stage
  after `--initial-moving-transform` does; `stage` the fields of `cohort_recipe()["rigid"]`;
  `transform` `"rigid"` or `"affine"`; `mask` a fixed-image mask for this fit (`None` = the
  session's `fixed_mask`; ANTs' moving-image mask has no equivalent); `out_prefix` writes
  `<prefix>0GenericAffine.mat`. Use it in a `with` block. `register_rigid(...)` does one rigid
  fit in one call. `rigid_session` / `RigidSession` are the 0.3 names of `linear_session` /
  `LinearSession`.
- `gpu_available()`, `gpu_info()`, `fits_gpu(fixed, iso_mm=, channels=)`,
  `iso_grid_dims(img, iso_mm)`.
- Errors: `GpuUnavailableError` and `GpuCapacityError` (checked before any work: fall back to
  ANTs), `RegistrationError`, `RegistrationCancelled`.

### Environment

| Variable | Effect |
| --- | --- |
| `DBSLOC_GPU_BACKEND` | `vulkan`, `metal`, `dx12` or `auto` (default) |
| `DBSLOC_GPU_ADAPTER` | pick the adapter whose name contains this text |
| `DBSLOC_GPU_ALLOW_SOFTWARE=1` | accept a software adapter (lavapipe, WARP); slow, for tests |
| `DBSLOC_GPU_MEMORY_BUDGET_MB` | override the memory budget of the capacity check |

The capacity check uses the device memory the graphics API reports, halved on integrated GPUs
(they share memory with the host, e.g. Apple Silicon).

## Recipe

The schedule is the caller's: `register_syn(..., recipe={"rigid": {...}, "affine": {...},
"syn": {...}})`. The structure:

- centre-of-mass initialisation, then `rigid` and `affine` on Mattes MI over the first channel
  (fields `gradient_step`, `iterations`, `convergence_threshold`, `convergence_window`,
  `shrink_factors`, `smoothing_sigmas`, `sigma_units` (`"vox"` / `"mm"`), `bins`,
  `sampling_percentage`, `seed`; regular sampling),
- then `syn` on CC over every channel (`gradient_step`, `update_field_variance`,
  `total_field_variance`, `cc_radius`, `channel_weights`, and the same schedule fields),
- winsorize `[0.005, 0.995]`, no histogram matching, all on an isotropic grid (`iso_mm`) built
  from the fixed image.

A stage or field left out keeps the dbs-localizer cohort value, which `cohort_recipe(channels)`
returns in full: Rigid and Affine `MI[32 bins, regular 25 %]`, `1000x500x250x100`; SyN
`[0.1, 3, 0]` with `CC[radius 4]` (T1w + T2w: weight 0.5 each), `100x70x50x20`; shrink
`8x4x2x1`, smoothing `3x2x1x0 vox`. An unknown key, a value of the wrong type, mismatched level
counts, or what the GPU kernels do not support (MI bins other than 32, a CC radius above 8) is a
`ValueError` before any work. The result's `"recipe"` is the full recipe that ran.

## Development

```bash
maturin develop --release && pytest            # Python tests (skip without a GPU)
cargo nextest run --workspace --exclude dbsloc-gpureg-py --features dbsloc-gpu/dev-tools
```

CI (`ci.yml`) runs both on Linux with Mesa's software Vulkan driver. `wheels.yml` builds all
wheels on a `v*` tag (and creates the GitHub release) or by hand (`gh workflow run wheels`).
Release: bump `version` in `Cargo.toml` (`[workspace.package]`; the wheel takes it from there), add a `CHANGELOG.md` section, tag `v<version>`.

## Source and credit

`crates/dbsloc-{core,niftiio,matfile,reg,gpu}` are copies of
[Wyss DBS Localizer](https://github.com/WyssCenter/dbs-localizer) at commit `00faf42`
(2026-10-26); the only change is the removal of the VAT solver (and its `dbsloc-vat`
dependency) from `dbsloc-gpu`. `crates/dbsloc-gpureg-py` is the Python binding (PyO3). Please
cite the engine as in `CITATION.cff`.
