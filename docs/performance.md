# Performance

This page says how FastWarp's speed is measured against ANTs, how to repeat the measurement,
and where the results come from. The measurements are part of the
[accuracy benchmark](benchmarks.md): the same run that scores accuracy on public data also
times every registration, a set of linear fits and the post-processing, and records the machine
and software it ran on.

No benchmark results are reported here yet. The tables below will be the `report` output of a
full run, with its machine block. Until then, the only numbers on this page are earlier
measurements on non-public data ([below](#earlier-measurements-on-non-public-data)), which a
reader cannot repeat.

## What is measured

All times are wall-clock seconds. ANTs runs in a child process, one at a time, never at the same
time as FastWarp. Both use all cores unless `--threads N` caps both alike (ANTs through
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, FastWarp's CPU work through `RAYON_NUM_THREADS`; the GPU
work is not affected). Every record states the setting and the core counts; timings with
different settings are not comparable, and the report keeps them apart.
[Timings](benchmarks.md#timings) on the benchmark page defines every field exactly.

| Measurement | Command | FastWarp | ANTs | Data |
| --- | --- | --- | --- | --- |
| Registration: Linear, SyN, total | `run` | `register_syn`'s `linear_s`, `syn_s`, `total_s`, and the call's wall time | `antsRegistration` with the same recipe: the stage times and `Total elapsed time` of its `--verbose 1` log, and the process wall time | every pair of the benchmark (default 50) |
| Linear sessions | `linear` | one `linear_session` per pair: preparation (including opening the GPU), rigid fit, affine fit from it, a rigid multi-start; and preparation + rigid + affine together | `antsRegistration` with the same rigid stage, then the same affine stage from its result: each stage's time, and both together | `snsx` (subjects onto the MNI 2009c template; default 20 pairs) |
| Post-processing | `post` | `compose_field` (forward and inverse displacement field), `apply_transforms` (Lanczos re-warp), each with reading and writing an uncompressed `.nii` (with `--gzip` also `.nii.gz`) | `antsApplyTransforms --float 1` doing the same | the pairs `run` registered |

Each step also records an accuracy figure: the registrations are scored against manual labels
and landmarks (`evaluate`); the linear fits by their landmark error and their distance from
ANTs' fit; the post-processing by the difference between the two tools' fields (mm) and
re-warped images. The engines' totals and wall times do not cover exactly the same work;
[Timings](benchmarks.md#timings) says what each includes.

The recipe is `--recipe cohort`, FastWarp's default (`fastwarp.cohort_recipe(1)`): one T1w
channel, Rigid and Affine on Mattes MI, SyN on CC, on a 1 mm grid. None of the public datasets
has a T2w aligned to the T1w, so the benchmark does not time the two-channel registration.

## How to reproduce

Install as in [Running it](benchmarks.md#running-it) (Python 3.13, the `benchmark` dependency
group, FastWarp from the checkout or from PyPI), then, with folders outside the repository:

```bash
B="uv run --no-sync python -m benchmarks.accuracy"
$B fetch --data ~/fastwarp-bench/data
$B run --data ~/fastwarp-bench/data --work ~/fastwarp-bench/work
$B linear --data ~/fastwarp-bench/data --work ~/fastwarp-bench/work
$B post --data ~/fastwarp-bench/data --work ~/fastwarp-bench/work
$B evaluate --data ~/fastwarp-bench/data --work ~/fastwarp-bench/work
$B report --work ~/fastwarp-bench/work
```

`--ants-dir DIR` runs the ANTs programs in `DIR` (for example a release of ANTs 2.6.5) instead of
the ones bundled with ANTsPy; `--adapter NAME` picks a GPU. Use a machine that does nothing
else during the run; `--threads N` caps both engines' CPU threads if the machine is needed
meanwhile, and its timings are then reported separately. `plan` estimates the time left; every
step resumes where it stopped.

## Where the results are

- `<work>/results/cohort/summary.md`: the tables to paste here, with the machine and software
  (CPU, GPU and its driver name, operating system, fastwarp and ANTs versions, thread setting of
  each command, the benchmark's git commit);
- `timings.csv`, `linear.csv`, `post.csv` beside it: one row per registration, fit or step;
- the records they come from: `<work>/<dataset>/cohort/<pair>/{fastwarp,ants}/run.json`,
  `<work>/<dataset>/linear/<pair>/` and `<work>/<dataset>/cohort/<pair>/post.json`, each with
  the `environment` it ran in.

## Results

To be added from the first full run: the `Registration time`, `Linear fits` and
`Post-processing` tables of `summary.md`, with its machine block.

## Earlier measurements on non-public data

These numbers predate the benchmark. They were measured on data of a non-public cohort with
scripts that are not in this repository, so they cannot be repeated from this page, and they
will be replaced by the benchmark results. What is known about them is stated below; anything else is
not documented.

Registration: T1w + T2w of one subject (raw T1w, T2w aligned to it by one rigid fit) onto the
MNI 2009b template (0.5 mm), 1 mm registration grid, the default recipe, a
development build before 0.1.0 (internal version 0.4.0). Each figure is the median of three runs
of the engine's own `linear_s`, `syn_s` and `total_s` (2026-11-09).

| Hardware | Linear | SyN | Registration total |
| --- | --- | --- | --- |
| NVIDIA RTX A4000 (Linux, Vulkan) | 3.5 s | 6.4 s | 12.4 s |
| Apple M5 Pro (MacBook Pro, Metal) | 4.1 s | 9.4 s | 13.7 s |
| Intel Lunar Lake integrated GPU (Linux laptop, Vulkan) | 3.5 s | 16.4 s | 21.0 s |
| AMD Radeon Pro 5500 XT (Intel iMac, Metal) | 6.1 s | 16.7 s | 23.5 s |
| NVIDIA Quadro K620, 2 GB (Windows Server 2022, Vulkan), T1w only[^1] | 10.6 s | 54.7 s | 67.3 s |
| ANTs on 16 CPU threads, same recipe | | | 1000 to 1500 s |

The total includes preparing and uploading the images. The A4000 machine was under heavy CPU
load (load average about 110 on 48 cores) during its runs. For the ANTs row, the ANTs version,
the command line and the machine are not documented.

[^1]: 2 GB is too little for T1w + T2w at 1 mm (1.6 GB): the capacity check refuses it up
    front (`GpuCapacityError`), so a caller falls back to ANTs. T1w alone needs 1.4 GB and ran
    only with `FASTWARP_GPU_MEMORY_BUDGET_MB=1900`, since the default budget keeps 1 GB free for
    the desktop.

Linear sessions: one rigid fit of the T2w onto the T1w (the default rigid stage) took 1.2 s on
the M5 Pro, 1.6 s on the RTX A4000, 1.8 s on the Radeon Pro 5500 XT, 2.1 s on Lunar Lake and
2.7 s on the Quadro K620. A coregistration of a post-operative CT onto the T1w with five coarse
and two full rigid fits took 17 s on the laptop GPU where ANTs took 32 s, and a masked affine
fit 2.4 to 4.8 s on the RTX A4000 where ANTs took 5 to 11 s. The ANTs version and commands, and
the subjects behind these ranges, are not documented.

Post-processing, on a laptop with 8 threads: turning the composites into displacement fields on
the 0.5 mm template and subject grids, and a Lanczos re-warp of the subject image.

| Step | ANTs (ITK) | `compose_field` / `apply_transforms` |
| --- | --- | --- |
| Forward field (69 M voxels) | 36.5 s | 1.4 s |
| Inverse field (37 M voxels) | 20.9 s | 0.95 s |
| Lanczos re-warp | 31.0 s | 2.7 s |

Which laptop, which ANTs version and command lines, and which script produced this table are
not documented.
