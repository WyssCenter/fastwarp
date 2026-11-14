# Command line

The `fastwarp` command runs the registration on NIfTI files without writing Python. It comes
with the Python package: after `pip install fastwarp`, both `fastwarp ...` and
`python -m fastwarp ...` work. Both run the same code as the Python functions, so options,
results and exit status are the same as described here. Each command below names the Python
function it runs; the [user guide](user-guide.md) explains those functions in more detail.

```bash
fastwarp register template.nii.gz sub_T1w.nii.gz -o out/sub_
fastwarp apply -r template.nii.gz -i sub_T1w.nii.gz -o out/sub_in_template.nii.gz \
    -t out/sub_Composite.h5
```

`fastwarp --help` lists the commands; `fastwarp <command> --help` lists the options of one
command. `fastwarp --version` prints the version.

## Images

The command line reads and writes NIfTI-1 images (`.nii` and `.nii.gz`, the `.gz` in lower
case) only. A file with another extension (`.mha`, `.nrrd`, Analyze `.img`/`.hdr`) is refused
with exit status 2; convert it first. A `.nii` file that is not single-file NIfTI-1 (for example
NIfTI-2) cannot be read and gives exit status 3.

Images must be 3-D. A 4-D file with more than one volume is refused.

An image is read as ITK reads it, and so as ANTsPy's `ants.image_read` reads it:

- the geometry (origin, spacing, direction) comes from the sform or the qform by ITK's rules and
  is used in ITK's LPS convention;
- `scl_slope` and `scl_inter` are applied;
- the voxel values are rounded to `float32`, as an `ANTsImage` of type `float` holds them.

Registering the files with the command line therefore gives the same result as reading them with
`ants.image_read` and passing them to the Python functions with the same settings; note that the
default grid spacing of `register` differs from that of `register_syn` (see below). The tests
check this (`tests/test_cli.py`).

Output images are written as `float32` NIfTI files with both sform and qform set. An output
whose name ends in `.gz` is compressed. The folder of an output file or prefix is created when
it does not exist.

## Commands

### register

```bash
fastwarp register FIXED MOVING -o PREFIX [--fixed2 F2 --moving2 M2] [--iso-mm MM]
                  [--recipe FILE] [--outputs composite|split|both] [--write-warped]
                  [--adapter NAME] [--json]
```

Registers MOVING to FIXED: {term}`centre of mass`, Rigid and Affine
({term}`Mattes mutual information`, first channel), then {term}`SyN` ({term}`cross-correlation`,
every channel), or the start and stages a [stage-list recipe](guide/recipes.md#stage-lists)
names. This is {func}`fastwarp.register_syn`; see [Normalization](guide/normalization.md) and
[Output files](guide/outputs.md).

- `-o, --output PREFIX`: writes `<PREFIX>Composite.h5` (`[Affine, Warp]`),
  `<PREFIX>InverseComposite.h5` (`[InverseWarp, Affine^-1]`) and `<PREFIX>0GenericAffine.mat`.
  A prefix that ends in a path separator, such as `out/sub/`, is a folder, as in ANTs.
- `--outputs`: `composite` (default, the files above), `split` (what `antsRegistration` writes
  without `--write-composite-transform`: `<PREFIX>0GenericAffine.mat` and, after a SyN stage,
  `<PREFIX>1Warp.nii.gz` and `<PREFIX>1InverseWarp.nii.gz`, written as `float32` where ANTs
  writes `float64`) or `both`. Without a SyN stage the
  composites hold only the affine.
- `--fixed2 F2 --moving2 M2`: a second channel (for example T2w) for the SyN stage. The second
  moving image must already be in the space of MOVING. Give both or neither.
- `--iso-mm MM`: spacing of the isotropic grid the registration runs on, in millimetres. The
  default is the smallest voxel spacing of FIXED, so that the registration keeps the detail of
  the fixed image: a 1 mm template registers at 1 mm, a 0.5 mm template at 0.5 mm.

  This default differs from the Python function: `register_syn` uses 1.0 mm unless `iso_mm` is
  given. To get the result of a Python call that leaves `iso_mm` out, pass `--iso-mm 1`. A finer
  grid needs more GPU memory and time; `fastwarp fits` tells whether it fits.
- `--recipe FILE`: the schedule as a JSON file, in the form `fastwarp recipe` prints or in the
  stage-list form of the Python API, for example a SyN-only run from the identity:

  ```json
  {"initial": "identity", "stages": [{"transform": "syn", "cc_radius": 2}], "winsorize": null}
  ```

  `initial` is `"com"` (the default, also for `null`; `antsRegistration` without `-r` starts from
  the identity instead), `"identity"`, `{"path": "start.mat", "invert": false}` (a relative path
  is read from the folder of the recipe file) or `{"parameters": [...], "center": [...]}`;
  `stages` lists
  `rigid` / `affine` stages and at most one `syn` stage, last; `winsorize` is `[lower, upper]`
  or `null` (ANTs' plain rescale to [0, 1]). A stage or field left out keeps the default value.
  An unknown key, a wrong type, an unsupported stage order or a value the GPU kernels do not
  support is refused with exit status 2 before any work.
- `--write-warped`: also writes `<PREFIX>_Warped.nii.gz`, MOVING on the registration grid.
- `--adapter NAME`: use the GPU whose name contains NAME.
- `--json`: print the timings and diagnostics as JSON: the keys of the dictionary
  `register_syn` returns (`files` among them), with `levels` as a list of objects, and
  `outputs`, the files written (the same list as `files`).

The capacity check runs before the registration: a grid that does not fit the GPU stops with
exit status 5, and no transform files are written (the output folder may already have been
created). It checks the linear stages and the SyN stage each on its own and requires the larger
of the two to fit.

### linear

```bash
fastwarp linear FIXED MOVING [-o PREFIX] [--transform rigid|affine] [--mask MASK]
                [--winsorize LOWER,UPPER] [--initial FILE] [--stage FILE]
                [--adapter NAME] [--json]
```

One rigid or affine fit of MOVING to FIXED (Mattes mutual information, regular sampling). This
is {func}`fastwarp.linear_session` with one {meth}`~fastwarp.LinearSession.register` call; see
[Linear fits](guide/linear.md).

- `-o, --output PREFIX`: writes `<PREFIX>0GenericAffine.mat`. Without it nothing is written and
  the result is only printed.
- `--transform`: `rigid` (default) or `affine`.
- `--mask MASK`: a fixed-image mask on any grid; the metric samples only its non-zero voxels.
- `--winsorize LOWER,UPPER`: intensity quantiles, for example `0.005,0.995`, as ANTs'
  `--winsorize-image-intensities`.
- `--initial FILE`: start from this affine transform (`.mat` or `.txt`, one affine that maps
  fixed points to moving points, such as a `0GenericAffine.mat`). Without it the fit starts
  from the centres of intensity mass, like `--initial-moving-transform [fixed,moving,1]`.
- `--stage FILE`: the schedule as a JSON file with the fields of the `rigid` entry of
  `fastwarp recipe`. Fields left out keep the default.
- `--json`: print the result as JSON: the keys `LinearSession.register` returns
  (`parameters`, `center`, `metric`, `iterations`, ...) and `outputs`.

### apply

```bash
fastwarp apply -r REFERENCE -i INPUT -o OUTPUT -t T1 [-t T2 ...]
               [-n linear|nearestNeighbor|lanczosWindowedSinc] [-f VALUE]
```

Resamples INPUT onto the grid of REFERENCE through the transforms, as `antsApplyTransforms`
does. This is {func}`fastwarp.apply_transforms`; it runs on the CPU and needs no GPU.

- `-t, --transform FILE`: a transform file: `.h5` composite, `.mat` or `.txt` affine, or a
  `.nii`/`.nii.gz` displacement field. Repeat it for several. The order is that of
  `antsApplyTransforms -t` and of the list `ants.apply_transforms` takes: a point of the
  reference grid goes through the first transform given first, then through the next. For the
  output of `fastwarp register`, give `-t <PREFIX>Composite.h5` to bring MOVING onto FIXED, and
  `-t <PREFIX>InverseComposite.h5` to bring FIXED onto MOVING. `-t [FILE,1]` uses the inverse of
  a linear transform (`.mat`, `.txt`, or a composite of affines) in its place, as in ANTs
  (quote it in a shell that expands brackets); with the split outputs FIXED goes onto
  MOVING with `-t [<PREFIX>0GenericAffine.mat,1] -t <PREFIX>1InverseWarp.nii.gz`. Inverting a
  displacement field is refused with exit status 2 (ANTs exits 1 without a message): pass the
  matching `InverseWarp` / `InverseComposite` file.
- `-n, --interpolation`: `linear` (default), `nearestNeighbor` or `lanczosWindowedSinc`.
- `-f, --default-value`: value of points that map outside INPUT (default 0).

The output is a `float32` image with the grid of REFERENCE.

### compose

```bash
fastwarp compose -r REFERENCE -o FIELD -t T1 [-t T2 ...] [--compression-level N]
```

Writes the transforms as one displacement field on the grid of REFERENCE, the file
`antsApplyTransforms --compose` writes (`float32` vectors in LPS millimetres). This is
{func}`fastwarp.compose_field`; it needs no GPU. The order of `-t` is as for `apply`.
`--compression-level` (0 to 9) sets the gzip level of a `.nii.gz` output; 0 stores the data
without compressing it, which suits displacement fields.

### recipe

```bash
fastwarp recipe [--channels 1|2]
```

Prints the default [recipe](guide/recipes.md) as JSON, the same dictionary as
{func}`fastwarp.cohort_recipe`. Edit a copy and pass it with `--recipe`.

### gpu-info

```bash
fastwarp gpu-info [--adapter NAME] [--json]
```

Prints the GPU FastWarp would use as JSON (`name`, `backend`, `device_type`, `driver`,
`memory_budget_bytes`), the same as {func}`fastwarp.gpu_info`. Exit status 4 when no usable GPU is
found.

### fits

```bash
fastwarp fits FIXED [MOVING] [--iso-mm MM] [--channels 1|2] [--recipe FILE] [--adapter NAME]
              [--json]
```

Checks whether a registration of FIXED (and MOVING) at `--iso-mm` fits the GPU, with the check
`register` runs first. `--iso-mm` defaults to the smallest voxel spacing of FIXED, as for
`register` (the Python function `fits_gpu` defaults to 1.0 mm). Only the image headers are
read. Exit status 0 when it fits, 5 when it does not, 4 when there is no usable GPU. Without
MOVING, the moving grid is taken to be the fixed grid. `--recipe` checks the stages of that
recipe (default: the default recipe); a chain without a SyN stage needs far less GPU memory. This
is {func}`fastwarp.fits_gpu`; see [GPU and memory](guide/gpu.md#what-fits).

### bench

```bash
fastwarp bench [--size NAME ...] [--repeat N] [--warmup N] [--threads N]
               [--recipe FILE] [--adapter NAME] [--json]
fastwarp bench FIXED MOVING [--iso-mm MM] [--repeat N] ...
```

Times the registration (the `register` path, without writing files) and prints the machine it
ran on. It needs no ANTs and downloads nothing.

Without images it registers built-in synthetic head volumes. They are computed from formulas, so
every run on every machine registers the same data and timings can be compared. The moving
volume is the fixed one deformed smoothly and moved by a small rotation and shift. The sizes
cover the MNI152 field of view (182 x 218 x 182 mm):

| `--size` | Grid | Spacing |
| --- | --- | --- |
| `tiny` | 30 x 36 x 30 | 6 mm (for tests) |
| `mni-2mm` | 91 x 109 x 91 | 2 mm |
| `mni-1mm` | 182 x 218 x 182 | 1 mm |
| `mni-0.5mm` | 364 x 436 x 364 | 0.5 mm |

The default is `mni-1mm` and `mni-0.5mm`. Each size registers at its own spacing. A size that
does not fit the GPU is skipped with a note. A size whose registration fails is reported as
failed and the other sizes still run. The exit status is 1 when a size failed, else 5 when every
size was skipped. Only GPU memory is checked: the `mni-0.5mm` size also needs about 2 GB of host
memory. With FIXED and MOVING, it times your images at `--iso-mm` (default: the smallest voxel
spacing of FIXED, as for `register`).

- `--repeat N`: timed runs per size (default 1).
- `--warmup N`: untimed runs before the timed ones. The default is 1 when `--repeat` is above 1,
  else 0.
- `--threads N`: CPU threads for the host-side work. The default is all logical cores, or
  `RAYON_NUM_THREADS` when it is set. It must be set before anything in the process has started
  the thread pool, so with `python -m fastwarp` use it in a fresh process.
- `--recipe FILE`: a recipe as for `register` (default: the default recipe).
- `--json`: print the report as JSON.

Each run reports four times in seconds:

- `linear_s`: the centre of mass, Rigid and Affine stages;
- `syn_s`: the SyN stage;
- `total_s`: all phases of the registration: resampling to the isotropic grid, intensity
  preprocessing, linear, SyN and the final warp;
- `wall_s`: the whole call, including opening the GPU and the capacity check.

With more than one run, a `med` line gives the median of each column. The machine block lists
the GPU name, driver, backend, device type and memory budget, the CPU model, logical cores and
threads used, the operating system and the FastWarp version.

## Exit status

| Status | Meaning |
| --- | --- |
| 0 | Success. For `fits`: the registration fits. |
| 1 | The registration, or another step of the engine, failed. |
| 2 | Invalid arguments: an unknown option, a bad value, an unsupported file format, a bad recipe. |
| 3 | A file could not be read or written (missing input, unreadable NIfTI or transform file). |
| 4 | No usable GPU (`GpuUnavailableError` in Python). |
| 5 | The grid does not fit the GPU (`GpuCapacityError` in Python); for `bench`, every size. |
| 6 | The run was cancelled (`RegistrationCancelled` in Python). |

Status 4 and 5 mean that the GPU cannot run this registration, so a script can run ANTs instead
([Fall back to ANTs](guide/gpu.md#fall-back-to-ants)). Errors are printed as one line on
standard error, starting with `fastwarp: error:`. Results go to standard output.

Ctrl-C stops the program at once, as for any other program; files that were being written may be
incomplete. Status 6 is not produced by Ctrl-C in this version; it is reserved for a cancelled
run.

## Environment

The command line reads the same [environment variables](environment.md) as the Python package,
at the moment the GPU is opened:

| Variable | Effect |
| --- | --- |
| `FASTWARP_GPU_BACKEND` | `vulkan`, `metal`, `dx12` or `auto` (default) |
| `FASTWARP_GPU_ADAPTER` | pick the adapter whose name contains this text (`--adapter` wins) |
| `FASTWARP_GPU_ALLOW_SOFTWARE=1` | accept a software adapter (lavapipe, WARP); slow, for tests |
| `FASTWARP_GPU_MEMORY_BUDGET_MB` | override the memory budget of the capacity check |
| `FASTWARP_GPU_WAIT_TIMEOUT_S` | seconds to wait for the GPU before failing (default 300) |
| `RAYON_NUM_THREADS` | CPU threads for the host-side work (default: all logical cores) |

The former `DBSLOC_` names are still read when the `FASTWARP_` name is unset.

## The standalone program

The same command line is also built as a program of its own from the crate `fastwarp-cli`:

```bash
cargo build --release --locked -p fastwarp-cli
target/release/fastwarp --help
```

It behaves exactly like `python -m fastwarp` and needs no Python. Release builds of this program
are not published yet.
