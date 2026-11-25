# Changelog

## Unreleased

- New: `fastwarp.release_idle_gpus()` frees the GPU devices that a process keeps between
  registrations.
- Changed: the field inversion of the SyN stage, which usually converges after one of its 20
  steps, no longer runs the remaining steps on the GPU for large grids: the GPU itself sets the
  size of their dispatches to zero once the inversion has converged. Registration is 3 to 6 %
  faster on an RTX PRO 6000 (more on GPUs where these steps weigh more), with the same files to the
  bit.
- Changed: `fastwarp.register_atlases` registers and resamples up to four atlases at the same
  time on one GPU, as many as fit its memory, with the same files to the bit. On an RTX PRO 6000,
  deep7t with 9 atlases: 27.8 s to 12.5 s for the call, 31.9 s to 16.6 s for the whole Python
  process.
- Changed: joint label fusion on a dedicated GPU is 2.5 to 3.6 times faster with the same labels,
  offsets and intensities to the bit. The patch search kernel handles 64 neighbouring voxels of
  one atlas per workgroup and two candidate offsets per thread, the size of each dispatch follows
  the measured kernel time, the GPU mode cuts the image into blocks by the number of fused voxels
  and gathers the votes only around them, and the host steps around the kernels run in parallel.
  On an RTX PRO 6000, `mode="gpu"`: deep7t with 9 atlases 12.1 s to 3.4 s, a 1 mm brain with 5
  atlases 9.7 s to 3.4 s, with 20 atlases 23.2 s to 9.3 s.
- Changed: while a level of a linear stage runs on the GPU, a second thread smooths the images
  and draws the sample points of the next level (same results to the bit).
- Changed: the composite `.h5` files are written without building the whole file in memory
  first (the same bytes).
- Changed: a process keeps the GPU device of a finished registration (its idle buffers freed; four
  at most, by adapter and `FASTWARP_GPU_*` / `WGPU_*` settings) and reuses it for the next
  registration with the same settings, which saves the device creation, the kernel compilation and
  the known-answer check (about 0.2 s). `fastwarp bench` still times every run with a new device. With the two
  changes above, a 1 mm brain pair on an RTX PRO 6000 takes 2.1 to 3.5 s instead of 2.8 to 5.0 s
  (1.3 to 1.4 times faster), with the same files to the bit.
- Changed: the end of a registration is faster: the displacement fields are converted on several
  threads and the transform files are written in parallel. The files are the same to the bit.
- Changed: the linear stages prepare their sample points faster: at each level the affine stage
  reuses the points of the rigid stage (same seed, level grid and images, no fixed mask) and the
  jittered points are computed on all cores. The results are the same to the bit.
- New: a result viewer on the page *Try FastWarp in your browser*. After a run it shows the template
  and your image in three linked views (axial, coronal, sagittal, at the voxel
  proportions, left on the left; click or tap to place the cross, sliders and arrow keys for the
  slice, a full-screen button) and compares them as a blend, a flicker (paused with the system
  setting *reduce motion*) or as the edges of the template in yellow over the registered one, with a
  switch to the moving image before the registration. The crate `fastwarp-web` keeps 8-bit copies
  of the images for it until the next run and exposes `view_info` and `view_slice` to the worker.
- New: joint label fusion almost entirely on the GPU: `fastwarp.joint_label_fusion(...,
  mode="gpu")` adds to the GPU search the joint error matrix and an `f32` Cholesky solve per voxel
  (`jlf_weights.wgsl`; voxels whose solve is not reliable, and all with `nonnegative`, get the CPU
  `f64` weights) and the votes and intensities gathered per voxel in z slabs (`jlf_gather.wgsl`;
  16 label slots per voxel, more are voted on the CPU). On a 1 mm brain with 5 atlases on an
  integrated GPU: 40 s instead of 245 s on the CPU (51 to 68 s hybrid); with 20 atlases 243 s
  instead of 1065 s (305 to 330 s hybrid); the same labels as the hybrid mode; in the tests, with the default `alpha`, 99 % of the GPU
  weights within 1e-4 of the CPU's. More than 32 atlases run as hybrid. The default stays `mode="cpu"`. In Rust:
  `JlfDevice::Gpu` and `fastwarp_gpu::jlf::joint_fusion_gpu_with` (with the search's budgets);
  `GpuJlfSearch::stats` also gives the seconds spent in the search, weights and gather kernels, so
  `GpuJlfStats` no longer implements `Eq` (its timing fields are `f64`).
- New: `fastwarp.register_atlases(target, atlases, labels, out_dir)` registers each atlas to the
  target with `register_syn` and writes its images and labels on the target's grid
  (`atlasNN_Warped.nii.gz`, `atlasNN_WarpedLabels.nii.gz`), ready for `joint_label_fusion`.
- New: the patch search of joint label fusion on the GPU: `fastwarp.joint_label_fusion(...,
  mode="hybrid", adapter=...)` runs the search (a WGSL kernel, `jlf_search.wgsl`, in `f32`, over
  z slabs with a halo) on the GPU and the weights, votes and intensities on the CPU in `f64`. About
  10 times faster on a 1 mm brain with 5 atlases on an integrated GPU; near-tied patches can be
  ranked differently than on the CPU (0.025 % of the voxel-atlas pairs there). The default stays
  `mode="cpu"`. In Rust: `JlfOptions::device` (`JlfDevice::Cpu` / `Hybrid`), the `PatchSearch`
  trait with `CpuSearch` and `joint_fusion_with` in `fastwarp_reg::jlf`, and
  `fastwarp_gpu::jlf::{joint_fusion, joint_fusion_async, GpuJlfSearch}`.
- New: joint label fusion as ANTs' `antsJointFusion` (ANTs 2.6.5) computes it, on the CPU and in
  parallel, with results equal to `antsJointFusion` on one thread except where search scores tie:
  `fastwarp.joint_label_fusion`
  (labels, intensities, label posteriors and atlas voting weights; masks, exclusions, a search
  radius image, PC or MSQ search, non-negative weights), the command
  `fastwarp ants-joint-fusion` with the `antsJointFusion` arguments, and
  `fastwarp.ants.joint_label_fusion` with those of ANTsPy's `ants.joint_label_fusion`. The CPU
  reference is the module `jlf` of `fastwarp-reg`. Guide: *Label fusion*.
- New: *Try FastWarp in your browser*, a page of the documentation that runs the default
  registration of `fastwarp register` (centre of mass, Rigid, Affine, SyN) on the visitor's GPU
  through WebGPU, with nothing uploaded, on a 1 mm or 2 mm registration grid, and returns `0GenericAffine.mat`, `1Warp.nii.gz`,
  `1InverseWarp.nii.gz`, `Warped.nii.gz` (the moving image on the fixed image's grid),
  `run.json` and `README.txt` as one zip file. It comes from a new crate, `fastwarp-web` (not
  published), with the wasm-bindgen entry points `gpu_info`, `register` and `cancel`, progress
  per stage, level and iteration, a check of every stage's buffers against the browser's
  `maxBufferSize` and `maxStorageBufferBindingSize` before the run, messages that suggest the
  2 mm grid when the GPU runs out of memory or is lost, and a small zip writer. Read the Docs
  builds the module and fetches the sample images, checked against their SHA-256, with
  `docs/_scripts/build_web_demo.sh`. Native behaviour and output are unchanged.
- Changed (breaking for code that implements or calls the `Backend`, `LinearEngine` or `LinearMetric` traits): the public `Backend` trait of `fastwarp-reg` is async. Every operation on images and fields is an `async fn` (`name`, `field_grid` and `release_idle_buffers` stay synchronous), so an implementation must return futures for them (for example as `async fn`), and generic code that calls them must `.await` them or use the synchronous entry points `run_syn`, `run_syn_from`, `run_syn_cached`, `register`, `register_full` and `stages::register_stages`, which keep their signatures. The linear stages are async in the same way: `LinearEngine::prepare_level` / `prepare_level_borrowed` and `LinearMetric::value` / `value_and_derivative` are `async fn`s. The SyN loop and the rigid and affine stages are written once as async code: `run_syn_async`, `run_syn_from_async`, `run_syn_cached_async`, `register::register_async`, `register::register_full_async`, `stages::register_stages_async`, `stages::run_linear_stages_async`, `linear::run_linear_stage_cached_async`, `linear::run_linear_chain_cached_async`, `linear::run_rigid_registration_from_cancellable_async` and `optimizer::optimise_level_async`; the synchronous entry points drive them and fail with an error when a backend's or an engine's future does not complete at once. On a native backend every future is ready on its first poll, so native behaviour and output are unchanged. The synchronous methods that `CpuBackend` keeps for its parity tests are hidden from the documentation and not a stable API.
- Changed: the operations of `WgpuBackend` (its `Backend` and `LinearEngine` methods and the metric of a linear level, `GpuMiLevel`) await the async device layer (buffer allocation, error scopes, read-backs, waits) instead of blocking, so on wasm32 they no longer block. Its hidden synchronous duplicates of the `Backend` methods are removed: call the trait methods and await them (natively, `pollster::block_on` drives them). The hidden test helpers `WgpuBackend::smooth_field_gather` and `smoothing_is_tiled` are now `async fn`s. New async forms: `WgpuBackend::download_image_async`, `WgpuBackend::prepare_level_with_chunk_async`, `parity::check_all_async`, `check_mi_async` and `check_mi_chunked_async`; the synchronous ones keep their signatures. Native behaviour and output are unchanged. `WgpuBackend::new_async`, `from_context_async` and `from_shared_context_async` create it without blocking (the kernel compilation's error scopes and the known-answer self-test are awaited), so it can be created on wasm32; the synchronous constructors keep their signatures. New: `GpuContext::compile_async` and `SynKernels::new_async`.

- Internal: an async device layer in `fastwarp-gpu` for the browser, which cannot block on the GPU: `GpuContext::new_async` / `new_with_async`, `sync_async`, `scoped_async`, `timed_async`, `staging_buffer_async`, `read_bytes_async` / `read_f32_async` / `read_u32_async` (and the staged variants), `BufferPool::acquire_async`, `device::request_device_async` and `enumerate_async`. On native targets every future is ready on its first poll and the blocking functions are the same code driven to completion, so native behaviour and output are unchanged; on wasm32 the blocking functions return an error and the async ones await the browser (adapter and device requests, error scopes, buffer mapping, work done). The staging buffers of the WebGPU backend are taken out of their slot for a read-back instead of being locked during it, and a read-back whose future is dropped before it completes unmaps its staging buffer. The buffer pool is no longer locked while an allocation is awaited. Changed (breaking): the public `device::enumerate_async` returns `GpuResult<Vec<wgpu::Adapter>>` instead of `Vec<wgpu::Adapter>`: on wasm32 a refused adapter request is reported with the browser's reason in the no-adapter error instead of as an empty list, and the adapter granted for `WGPU_FORCE_FALLBACK_ADAPTER` is kept even when it does not report itself as a CPU adapter.

- Internal: every input and output of a registration can go through bytes in memory (for a browser without a file system): `NiftiFile::from_bytes` (gzip detected by its magic bytes), `WriterOptions::in_memory` with `to_bytes` and `write_nifti_to`, `Volume::from_nifti_bytes` and `to_nifti_bytes`, `parse_field_nifti` and `format_field_nifti`, `parse_nifti_grid`, `parse_transform`, and `stages::stage_output_files` / `visit_stage_outputs`; `write_stage_outputs` writes each file in turn through the visitor, so its memory use and the files written are unchanged.
- Internal: the engine crates `fastwarp-reg` and `fastwarp-gpu` build for `wasm32-unknown-unknown` (timing without `std::time::Instant`, the WebGPU backend on wasm32, a 32-bit safe size limit in the HDF5 reader), and the GPU kernels compile and pass their self-tests under the WebGPU default limits; native behaviour is unchanged.
- The SyN stage registers any number of channels (for example T1w + T2w + FGATIR), not 1 or 2.
  `register_syn`, `fits_gpu` and `cohort_recipe` take one or more images or channels; each channel
  weighs `1 / channels` by default (unchanged for one and two). `fastwarp register` takes
  `--channel FIXED MOVING` (repeatable) after the positional pair, and `fastwarp recipe` and
  `fastwarp fits` take any `--channels` of 1 or more. The ANTs compatibility mode takes any number
  of `CC` metrics in the SyN stage, and `fastwarp.ants.registration` one or more
  `multivariate_extras`. The GPU memory is the only limit: its plan adds the same 20 bytes per
  voxel for every further channel, which bounded the measured peak for one to four channels.
- Changed: `fastwarp register` no longer takes `--fixed2` and `--moving2`; `--fixed2 F2 --moving2 M2`
  becomes `--channel F2 M2`.
- Masks in the SyN stage (cost-function masking), so that a lesion or a resection cavity does not
  drive the deformation: `register_syn(fixed_mask=..., moving_mask=...)` and `fastwarp register
  --mask FIXED_MASK --moving-mask MOVING_MASK`, on any grid. As in ITK's SyN with ANTs'
  neighbourhood cross-correlation, every iteration resamples the fixed mask through the fixed-side
  and the moving mask through the moving-side transform (nearest neighbour), and the metric of
  every channel uses only the voxels inside both: window sums, value and gradient. The fixed mask
  also applies to the linear stages; the moving mask applies to the SyN stage only for now, and a
  recipe without a SyN stage refuses it. `fits_gpu` takes the masks for its memory plan (each
  level about 13 bytes per voxel more, plus 4 bytes per mask voxel). `fastwarp ants-registration`
  takes `-x [fixedMask,movingMask]` on the SyN stage and refuses a moving mask on a Rigid or
  Affine stage; `fastwarp.ants.registration` takes `mask` and `moving_mask` for the SyN types as
  ANTsPy applies them (the last stage, every stage with `mask_all_stages`) and refuses a moving
  mask that ANTsPy would give to a linear stage. A level with no voxel inside both masks fails
  with a clear error. Runs without masks are unchanged, bit for bit. With one mask, or with two
  or more levels, `antsRegistration` 2.6.5 reuses masks it warped earlier in the run; FastWarp
  does not, so masked results differ slightly from ANTs' there (see the differences listed under
  `fastwarp ants-registration`).
- Several SyN stages in one registration, for example a whole-brain SyN followed by a SyN that
  only a subcortical region of the template drives ("subcortical refine"). A stage-list recipe
  (`register_syn(recipe=...)`, `fastwarp register --recipe`) takes any number of `syn` stages
  after the linear stages; they must end with the same shrink factor. As in `antsRegistration`,
  each SyN stage starts from everything before it: its moving image and moving mask are sampled
  through its moving-side field, the earlier SyN field and the affine, with one interpolation of
  the image (a new GPU kernel, `k_warp_pre`; the existing kernels and results are unchanged, bit
  for bit). After each stage the fields are collapsed as `--collapse-output-transforms 1` does,
  so the outputs stay one affine and one warp per direction. Each stage of a stage list takes the
  mask switches `"fixed_mask"` and, on SyN stages, `"moving_mask"` (default `true`), which keep
  the call's masks out of single stages; a mask that every stage switches off is refused, and a
  `"moving_mask"` switch on a linear stage is an error. The result has `syn_stages`, the levels
  of each SyN stage, next to `levels` (all SyN levels, one stage after the other); the plan
  JSON of the ANTs compatibility mode now reports `syn_masks` as a list, one entry per SyN
  stage. `fastwarp ants-registration` takes several `-t SyN[...]` stages with the same CC images
  and one `-x` per stage, and refuses two different fixed-mask or moving-mask files across the
  SyN stages. The GPU capacity check plans each SyN stage on its own plus 12 bytes per voxel for
  the earlier field. Against ANTs 2.6.5, two SyN stages agree to an RMS of 5e-6 mm without
  masks; with masks the agreement is that of a masked single stage (see `docs/cli.md`).
  `fastwarp.ants` is unchanged: no ANTsPy type runs more than one SyN stage.
- Removed: `register_rigid`, `rigid_session` and `RigidSession`, deprecated in 0.2.0; use
  `LinearSession` for `RigidSession` and `linear_session(...)` for `rigid_session(...)`. For
  `register_rigid(fixed, moving, ...)`, use `with fastwarp.linear_session(fixed, moving,
  fixed_mask=..., winsorize=..., adapter=...) as s: s.register(stage=..., initial=...,
  out_prefix=..., cancel=...)`.

## 0.2.0 (2026-11-15)

- ANTs compatibility mode: `fastwarp ants-registration ARGS...` and
  `fastwarp ants-apply-transforms ARGS...` take the arguments of `antsRegistration` and
  `antsApplyTransforms` (ANTs 2.6.5), run the subset FastWarp supports and write the files ANTs
  writes under ANTs' names: `0GenericAffine.mat`, `1Warp.nii.gz`, `1InverseWarp.nii.gz` (or
  `0Warp.nii.gz` for SyN without `-r`), `Composite.h5` / `InverseComposite.h5` alone with `-a 1`,
  and the warped and inverse-warped images of `-o [prefix,warped,inverseWarped]` on the native
  grids. Supported: any Rigid / Affine stages with Mattes MI (regular sampling, 32 bins), then
  at most one SyN stage with one or two CC metrics; `-r [f,m,1]`, `Identity` or an affine file;
  fixed masks on linear stages; `-w`; `--random-seed` and `ANTS_RANDOM_SEED`; both stage-option
  orders and brackets split over shell words. Warped images and fields are `float64`, or
  `float32` with `--float 1`, as in ANTs. Everything that would change the result and is not
  supported (MI in SyN, dense or random MI sampling, `-z 0`, histogram matching, SyN masks,
  moving masks, other transforms, metrics and interpolators, ...) is refused with exit status 2
  and the reason, before the GPU is opened; `ants-apply-transforms` accepts `-z`, `--time-index`
  and `-u` with a transform output, which change nothing, with a warning.
  The compatibility mode is reached only through these subcommands, never by the program's name.
  The registration grid is the fixed image's smallest spacing; with an anisotropic fixed image the
  warp fields are resampled onto the fixed image's grid, where ANTs writes them, so the result
  differs a little more from ANTs' there (about 1 mm at the corners of the field of view in the
  tests, against 0.06 to 0.5 mm for an isotropic one). Stage options are assigned to stages as
  ANTs assigns them. Integer `-u` outputs of `ants-apply-transforms` wrap as in ANTs. The parsers
  are pure functions (`fastwarp_cli::ants::parse_registration` / `parse_apply`, and
  `resolve_seed` for `--random-seed` / `ANTS_RANDOM_SEED`) that return the plan, its recipe and
  its output files; the Python layer `fastwarp.ants` runs them through the same code. See
  `docs/cli.md`.

- Linear stages skip a level whose sample points leave Mattes mutual information undefined
  (no point on the fixed image, or one fixed value only; typical for a coarse level of a small
  image or a small mask) and continue with the finer levels, with a warning: on standard error
  in the command line (`register`, `linear`, `ants-registration`), and as a `UserWarning` in
  Python, also listed under `"warnings"` in the results of `register_syn` and
  `LinearSession.register` (the result of `fastwarp.ants.registration` keeps ANTsPy's keys).
  ANTs 2.6.5 drops the whole stage in that case without an error.
  When every level of a stage is skipped, the run fails and the message names the levels, their
  shrink factors, the candidate counts and the remedies.

- `fastwarp.ants`: ANTsPy's `ants.registration` and `ants.apply_transforms` (ANTsPy 0.6.3) with
  their arguments and results, for the part FastWarp supports. Each call builds the argument list
  ANTsPy builds (compared token for token in the tests) and runs it with the ANTs compatibility
  mode, the same code as `fastwarp ants-registration` / `ants-apply-transforms`, with the
  `ANTsImage`s handed over in memory. Runs `Rigid`, `Affine`, `AffineFast`, `BOLDAffine`,
  `QuickRigid`, `DenseRigid`, `BOLDRigid`, `SyNCC`, `SyNabp`, `antsRegistrationSyN[r|a|s|sr|so]`,
  `antsRegistrationSyNQuick[r|a]`, and `SyN`, `SyNRA`, `SyNOnly`, `SyNAggro`, `SyNLessAggro`,
  `ElasticSyN`, `SyNBold`, `SyNBoldAff` with `syn_metric="CC"`. ANTsPy's default `"SyN"` (Mattes
  mutual information in SyN), Translation, Similarity, BSplineSyN, GC, time-varying and other
  unsupported arguments (SyN masks, moving masks, histogram matching, other interpolators, ...)
  raise `fastwarp.ants.UnsupportedArgument` (a `ValueError`) before any work; arguments ANTsPy
  ignores for a type are ignored with a `UserWarning`; unknown keyword arguments raise
  `TypeError`. The result dict has ANTsPy's keys, warped images on the native grids with the
  fixed image's pixel type, and the transform lists in ANTsPy's order. Keyword-only additions:
  `iso_mm` (default: the fixed image's smallest spacing), `adapter`, `cancel` and
  `fallback="ants"`, which runs ANTsPy instead after a refusal, without a GPU or when the
  registration does not fit, with a warning and an `"engine"` key in the result. A raw argument
  list, `registration([...], None)`, runs as the subcommand does. `outprefix=""` makes a private
  temporary folder instead of ANTsPy's `tempfile.mktemp` name. ANTsPy is needed at call time
  only. See `docs/guide/ants.md`.

- Stage lists: `register_syn(recipe={"initial": ..., "stages": [...]})` runs any number of
  Rigid / Affine stages followed by at most one SyN stage (SyN alone is allowed), from the centres
  of mass (`"com"`, the default), the identity (`"identity"`), one affine from a `.mat` / `.txt`
  file (optionally inverted) or given parameters. Without a SyN stage the composites hold only the
  affine, as ANTs writes them. The capacity check (in `register_syn`, `fits_gpu(recipe=...)` and
  `fastwarp fits --recipe`) covers every stage: a linear-only chain is checked against the memory
  its sample buffers need, not the SyN stage's, and a chain with both is refused when the larger
  of the two phases does not fit. A relative path of the start file is read from the current
  folder in Python and from the recipe file's folder in `fastwarp register --recipe`. `None` for
  `initial` or `stages` means the default. The existing recipe form keeps working and gives
  bit-identical outputs. The result dict adds `initial_parameters`, `initial_center`,
  `linear_stages`, `deformable` and `files`.
- The recipe key `"winsorize"`: `[lower, upper]` quantiles (default `[0.005, 0.995]`, as before) or
  `None` for ANTs' plain rescale to [0, 1], what `antsRegistration` (and every ANTsPy call) does
  without `--winsorize-image-intensities`. `LinearSession(winsorize=None)` is unchanged: it uses
  the intensities without any rescale; `winsorize=(0.0, 1.0)` is ANTs' default preprocessing.
- `register_syn(outputs="split" | "both")` (`fastwarp register --outputs`) writes ANTs'
  non-composite files `0GenericAffine.mat`, `1Warp.nii.gz`, `1InverseWarp.nii.gz`; the fields are
  `float32` (ANTs writes `float64`).
- `apply_transforms(..., invert=[...])` and `compose_field(..., invert=[...])`, ANTsPy's
  `whichtoinvert` (one flag per transform), and `fastwarp apply` / `compose -t [file,1]`, as
  `antsApplyTransforms -t [file,1]`: a linear transform is replaced by its inverse in place.
  Inverting a displacement field raises `ValueError` (exit status 2) instead of ANTs' silent exit 1.
  `fastwarp apply` / `compose` and `compose_field` read and check the transforms before they load
  an image or create the output folder.
- A command line, `fastwarp` (also `python -m fastwarp`), for NIfTI files: `register`
  (`register_syn`), `linear` (one rigid or affine fit), `apply` (`apply_transforms`), `compose`
  (`compose_field`), `recipe` (the default recipe as JSON), `gpu-info`, `fits` and `bench` (timings
  on built-in synthetic volumes or on your images, with a description of the machine). Images are
  read as ITK and ANTsPy read them. Distinct exit status for invalid arguments (2), file errors (3),
  no usable GPU (4) and a grid too large for the GPU (5); `--json` for machine-readable output.
  An internal error prints one line and exits with status 1; Rust's panic message is shown only
  when `RUST_BACKTRACE` is set to a value other than 0. `register` and `linear` read their input images before they create
  the output folder. See `docs/cli.md`. The Python extension and the standalone program built from the new crate
  `fastwarp-cli` run the same code.
- Each GitHub release carries the standalone `fastwarp` program, which needs no Python, for Linux
  x86_64 and aarch64 (glibc 2.28 or later), macOS arm64 and x86_64 (macOS 11 or later) and Windows
  x86_64 (static C runtime), as `fastwarp-<target>.tar.gz` / `.zip` with a `.sha256` file each,
  and the installer scripts `fastwarp-installer.sh` (Linux, macOS) and `fastwarp-installer.ps1`
  (Windows). The installer of a release installs that release and holds the SHA-256 of its
  archives, so it refuses a download that does not match. It installs without administrator
  rights into `~/.local/bin` (Windows: `%LOCALAPPDATA%\Programs\fastwarp\bin`), adds that folder
  to PATH unless told not to (`--no-modify-path`, `-NoModifyPath`), and records what it did in a
  receipt next to the program; `--uninstall` (`-Uninstall`) undoes exactly that and leaves alone a
  `fastwarp` it did not install. Every release file has a GitHub build provenance attestation
  (`gh attestation verify`). See `docs/cli.md`.
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
- The documentation is reworked: the PyData Sphinx theme with light and dark modes, a user guide
  in short pages (how it works, images, normalization, output files and the direction each one
  maps, recipes with the type, unit and limits of every field, linear fits, resampling, GPU
  memory, limitations), troubleshooting, a glossary, reference pages for the API, the
  environment variables and the errors, and project pages for citing, security, the code of
  conduct and related projects. The README is shorter and links into it. The API reference moved
  from `api/index.html` to `api.html`.
- The docstrings of the public API are rewritten in NumPy style, with parameters, defaults,
  units, return values and the exceptions each function raises.
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
- The GPU capacity check (`fits_gpu`, `fastwarp fits` and the check before `register_syn`) no
  longer underestimates small grids: it now counts the buffers the coarser SyN levels and the
  device self-test leave allocated, which the finest level's bytes per voxel missed (on a
  48 x 48 x 48 grid the check planned 18.6 MB for a run that peaked at 20.9 MB). Small grids near
  the budget can now be refused where they passed before; on a 1 mm head the figure grows by
  under 2%. Its error now reads "the SyN stage on the registration grid ... needs ... bytes".

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
