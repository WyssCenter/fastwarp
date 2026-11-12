# fastwarp

GPU image registration for Python: the registration engine of **Wyss DBS Localizer** by
**Richard Köhler** (Wyss Center for Bio and Neuroengineering), packaged as a wheel. It runs a
full Rigid → Affine → SyN normalization in one call, and rigid or affine fits (coregistration,
template alignment, brain-shift refinement) in a session that keeps the image pair on the GPU.

The engine is a port of ANTs/ITK's registration (Mattes mutual information and
cross-correlation metrics, conjugate-gradient linear stages, SyN with Gaussian field
regularisation and field inversion) to GPU compute shaders via [wgpu](https://wgpu.rs). It runs
on **Metal** (macOS), **Vulkan** (Linux, Windows) and **DX12** (Windows), from one wheel per
platform for every Python ≥ 3.10. Outputs are ANTs files (`Composite.h5`,
`InverseComposite.h5`, `0GenericAffine.mat`) that ANTsPy and `antsApplyTransforms` read as is.

Licence: [Apache-2.0](LICENSE). The files ported from ANTs/ITK carry their Apache-2.0 notices
(`NOTICE`, [`docs/THIRD_PARTY_PORTS.md`](docs/THIRD_PARTY_PORTS.md)). Every file states its
licence in an SPDX header or through `REUSE.toml` ([REUSE](https://reuse.software/)).

## Speed

T1w + T2w of Lead-Tutor sub-57245 (raw T1w, T2w aligned to it by one rigid fit) onto the MNI
2009b template (0.5 mm), 1 mm registration grid, default recipe, `fastwarp` 0.4.0; median
of three runs, the engine's own `linear_s` / `syn_s` / `total_s` (2026-11-09):

| Hardware | Linear | SyN | Registration total |
| --- | --- | --- | --- |
| NVIDIA RTX A4000 (Linux, Vulkan) | 3.5 s | 6.4 s | 12.4 s |
| Apple M5 Pro (MacBook Pro, Metal) | 4.1 s | 9.4 s | 13.7 s |
| Intel Lunar Lake integrated GPU (Linux laptop, Vulkan) | 3.5 s | 16.4 s | 21.0 s |
| AMD Radeon Pro 5500 XT (Intel iMac, Metal) | 6.1 s | 16.7 s | 23.5 s |
| NVIDIA Quadro K620, 2 GB (Windows Server 2022, Vulkan), T1w only¹ | 10.6 s | 54.7 s | 67.3 s |
| ANTs on 16 CPU threads, same recipe | | | 1000–1500 s |

The total includes preparing and uploading the images. The A4000 machine was under heavy CPU
load (load average about 110 on 48 cores) during its runs.

¹ 2 GB is too little for T1w + T2w at 1 mm (1.6 GB): the capacity check refuses it up front
(`GpuCapacityError`), so a caller falls back to ANTs. T1w alone needs 1.4 GB and ran only with
`DBSLOC_GPU_MEMORY_BUDGET_MB=1900`, since the default budget keeps 1 GB free for the desktop.

Linear sessions: one rigid fit of the T2w onto the T1w (the default rigid stage) takes 1.2 s on
the M5 Pro, 1.6 s on the RTX A4000, 1.8 s on the Radeon Pro 5500 XT, 2.1 s on Lunar Lake and
2.7 s on the Quadro K620. As leaddbs-py runs them on the Lead-Tutor cohort: a post-op CT onto
the T1w anchor, five coarse and two full rigid fits, takes 17 s on the laptop GPU where ANTs
takes 32 s; the masked affine brain-shift fit takes 2.4–4.8 s on the RTX A4000 where ANTs takes
5–11 s.

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
gh release download -R WyssCenter/fastwarp -p "*.whl" -D wheels   # the latest release; needs the GitHub CLI
pip install --find-links wheels fastwarp
```

From source (needs a Rust toolchain; `rust-toolchain.toml` pins the version):

```bash
pip install "git+https://github.com/WyssCenter/fastwarp.git"
```

## Use

```python
import ants
import fastwarp

fixed = [ants.image_read("tpl_t1.nii"), ants.image_read("tpl_t2.nii")]
moving = [ants.image_read("sub_T1w.nii.gz"), ants.image_read("sub_T2w_in_T1w.nii.gz")]

if fastwarp.fits_gpu(fixed[0], iso_mm=1.0, channels=2):
    info = fastwarp.register_syn(fixed, moving, "out/sub_", iso_mm=1.0)
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
| `DBSLOC_GPU_WAIT_TIMEOUT_S` | seconds to wait for the GPU before failing (default 300) |

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

The Python tools (maturin, pytest, ruff; ANTsPy in the `ants` group) are locked in `uv.lock`:

```bash
export UV_PYTHON=3.13   # ANTsPy has no Python 3.14 wheels yet; see below
uv sync --locked --no-install-project --group ants
uv run --no-sync maturin develop --uv --release && uv run --no-sync pytest   # Python tests (skip without a GPU)
PYO3_PYTHON=.venv/bin/python cargo nextest run --locked --workspace --features dbsloc-gpu/dev-tools   # Windows: .venv/Scripts/python.exe
cargo clippy --locked --workspace --all-targets --features dbsloc-gpu/dev-tools -- -D warnings
```

Development uses Python 3.14 (`.python-version`). ANTsPy, which the comparisons with ANTs need,
has no 3.14 wheels yet, so work with the `ants` group sets `UV_PYTHON=3.13`, as CI does;
without it a later `uv sync` rebuilds the environment on 3.14 and drops ANTsPy. The lock covers
Python 3.13 and 3.14 (`[tool.uv] environments`); the wheel itself supports 3.10 and later.
`PYO3_PYTHON` is the Python the binding's Rust tests are built against.

`constraint-dependencies = ["scipy>=1.15"]` keeps the resolver from trading SciPy
down to 1.6 (no wheels, open advisories) for a newer NumPy; drop it once ANTsPy's dependencies
resolve to a current SciPy without it.

Test variables:

| Variable | Effect |
| --- | --- |
| `DBSLOC_GPU_REQUIRE_ADAPTER=1` | a missing GPU adapter fails the Rust and Python GPU tests instead of skipping them |
| `DBSLOC_REQUIRE_ANTSPY=1` | ANTsPy failing to import fails the Python tests instead of skipping the comparisons with ANTs |
| `WGPU_FORCE_FALLBACK_ADAPTER=1` | select software adapters only (with `DBSLOC_GPU_ALLOW_SOFTWARE=1`) |
| `DBSLOC_ATLAS_INDEX_MAT` | path to an atlas index `.mat` for a `dbsloc-matfile` test; unset skips it, a missing file fails it |
| `DBSLOC_UIPREFS_DIR` | folder of real `.mat` files for an `#[ignore]`d `dbsloc-matfile` test |
| `DBSLOC_REG_AB_DIR` | folder of a real subject (ANTs outputs) for the `#[ignore]`d A/B tests in `dbsloc-gpu` |
| `DBSLOC_REG_PROTO_DIR` | folder of prototype composite transforms for a `dbsloc-reg` test; unset skips it |
| `DBSLOC_REG_ANTS_COMPOSITE` | path of a real ANTs composite `.h5` for a `dbsloc-reg` test; unset skips it |
| `DBSLOC_GPU_BENCH` | enables the `#[ignore]`d GPU benchmarks in `syn_bench` |

The Rust tests that compare the engine with the real ANTs programs (`antsRegistration`,
`antsApplyTransforms`, `N4BiasFieldCorrection`) are `#[ignore]`d. Run them with
`DBSLOC_ANTS_DIR` set to the folder holding ANTs 2.6.5; unset, they skip, and a folder without
the program fails them:

```bash
DBSLOC_ANTS_DIR=/path/to/ants/bin cargo nextest run --locked -p dbsloc-reg --run-ignored only \
  -E 'binary_id(dbsloc-reg::ants_compat) | binary_id(dbsloc-reg::coreg_metric) | binary_id(dbsloc-reg::n4_ants)'
```

CI (`ci.yml`) runs the Rust and Python tests on Linux with Mesa's software Vulkan driver, and
fails when the adapter or ANTsPy is missing. `full.yml` runs the ANTs comparison tests on
non-draft PRs, nightly when main moved and by hand, with the three programs from the official
ANTs 2.6.5 Ubuntu 24.04 release, checked against the SHA256 sums in
`.github/ants-2.6.5-ubuntu-24.04.sha256` and cached. `audit.yml` runs `cargo deny` and `uv audit` when a
lockfile or manifest changes and weekly; Dependabot opens one combined monthly PR. `wheels.yml`
builds all wheels on a `v*` tag (and creates the GitHub release) or by hand
(`gh workflow run wheels`).
Release: bump `version` in `Cargo.toml` (`[workspace.package]`; the wheel takes it from there), add a `CHANGELOG.md` section, tag `v<version>`.

## Source and credit

`crates/dbsloc-{core,niftiio,matfile,reg,gpu}` are copies of the registration engine of
Wyss DBS Localizer (an internal Wyss Center repository), taken at revision `90c1df3`
(2026-11-10) of an earlier standalone copy of that engine, which this repository replaces
(`ENGINE_SOURCE` names that revision); later changes are in `CHANGELOG.md`. `crates/fastwarp-py` is the Python binding (PyO3). Please
cite the engine as in `CITATION.cff`.
