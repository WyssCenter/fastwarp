# Performance

This page says how FastWarp's speed is measured against ANTs, how to repeat the measurement,
and what it gave on five machines. The measurements are part of the
[accuracy benchmark](benchmarks.md): the same run that scores accuracy on public data also
times every registration, a set of linear fits and the post-processing, and records the machine
and software it ran on. The [results](#results) are the `report` output of the full default run
on each machine.

To time FastWarp alone on your own machine, without ANTs and without downloading data, run
`fastwarp bench` (see [Command line](cli.md#bench)): it registers built-in synthetic volumes and
prints the machine it ran on.

## What is measured

All times are wall-clock seconds. ANTs runs in a child process, one at a time, never at the same
time as FastWarp. Both use all cores unless `--threads N` caps both alike; timings with
different settings are kept apart. Each record also says which machine and software it ran on.
[Timings](benchmarks.md#timings) on the benchmark page defines every field and what it
includes.

| Measurement | Benchmark command | FastWarp | ANTs | Data |
| --- | --- | --- | --- | --- |
| Registration | `run` | {func}`~fastwarp.register_syn` | `antsRegistration` with the same recipe | every pair of the benchmark (default 50) |
| Linear fits | `linear` | {func}`~fastwarp.linear_session`: rigid, affine, a rigid multi-start | `antsRegistration` with the same rigid and affine stages | `snsx` (default 20 pairs) |
| Post-processing | `post` | {func}`~fastwarp.compose_field`, {func}`~fastwarp.apply_transforms` | `antsApplyTransforms` | the pairs `run` registered |

Each step also records an accuracy figure: the registrations are scored against manual labels
and landmarks (`evaluate`), the linear fits by their landmark error and their distance from
ANTs' fit, and the post-processing by the difference between the two tools' outputs.

The recipe is `--recipe cohort`, FastWarp's default ({func}`fastwarp.cohort_recipe` with one
channel): one {term}`T1-weighted <T1w, T2w>` image per side, Rigid and Affine on
{term}`Mattes mutual information`, {term}`SyN` on {term}`cross-correlation`, on a 1 mm grid. None of the public datasets
has a T2w aligned to the T1w, so the benchmark does not time a registration with more than one
channel.

A registration's time in the [results](#results) is its wall time (`wall_s`), the time a caller
waits: for FastWarp the {func}`~fastwarp.register_syn` call with the images already read (reading
them is recorded separately and took 0.02 to 0.3 s per pair), for ANTs the whole
`antsRegistration` process, including its start and reading and writing its files. The engines'
own totals (`total_s`: FastWarp's includes writing the transforms, ANTs' leaves reading and
writing out) are in the tables per machine; their ratios are within a few per cent of the
wall-time ratios.

## How to reproduce

Install as in [Running it](benchmarks.md#running-it) (Python 3.13, the `benchmark` dependency
group, FastWarp from the checkout or from PyPI), then run every step unattended, with folders
outside the repository:

```bash
uv run --no-sync python benchmarks/scripts/chain.py --data ~/fastwarp-bench/data --work ~/fastwarp-bench/work
```

`benchmarks/scripts/chain.py` runs `fetch`, `run`, `linear`, `post`, `evaluate` and `report`
in this order, writes every command it ran to `~/fastwarp-bench/logs/commands.txt` and each
step's output beside it, and keeps the machine awake where that needs no admin rights
(`caffeinate` on macOS, `systemd-inhibit` on Linux, `SetThreadExecutionState` on Windows).
Meanwhile `benchmarks/scripts/monitor.py` samples the machine every 5 s (load, the CPU use of the
benchmark and of other processes, GPU use where the platform reports it, mains or battery
power) into `~/fastwarp-bench/monitor/samples.csv`; at the end it writes `per_record.csv`, the
load during each registration, fit and post-processing step. Running the script again
continues an interrupted run. The steps one by one:

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
meanwhile, and its timings are then reported separately. `chain.py` runs the steps with their
defaults; for these options, run the steps one by one. `plan` estimates the time left; every
step resumes where it stopped.

## Where the results are

- `<work>/results/cohort/summary.md`: the tables below, with the machine and software
  (CPU, GPU and its driver name, operating system, fastwarp and ANTs versions, thread setting of
  each command, the benchmark's git commit);
- `timings.csv`, `linear.csv`, `post.csv` beside it: one row per registration, fit or step;
- the records they come from: `<work>/<dataset>/cohort/<pair>/{fastwarp,ants}/run.json`,
  `<work>/<dataset>/linear/<pair>/` and `<work>/<dataset>/cohort/<pair>/post.json`, each with
  the `environment` it ran in.

For the runs below, these files are in the repository, one folder per machine:
`benchmarks/results/2026-11/<gpu>/` holds its `summary.md` and `summary.png`, the CSV tables of
`<work>/results/cohort/` and the `environment*.json` of `run`, `linear` and `post`.

## Results

### The runs

The full default benchmark ran on each of five machines on 17 and 18 November 2026, both
engines on every pair (on the Quadro K620, FastWarp on the pairs that fit its memory, see
[Registration](#registration)):

- **Code:** the benchmark at commit `124d690` (a clean checkout); FastWarp 0.2.0, the wheel from
  PyPI installed against its pinned hashes (`benchmarks/fastwarp-pypi.txt`; engine
  `fastwarp-legacy 90c1df3`); ANTs `v2.5.1.post95-g2236a75`, the `antsRegistration` and
  `antsApplyTransforms` bundled with ANTsPy 0.6.3 as `uv.lock` pins it, the same build on every
  machine; Python 3.13, NumPy 2.3.5, SciPy 1.15.3.
- **Data:** the default datasets and pairs: 20 `mindboggle`, 10 `deep7t` and 20 `snsx` pairs
  registered (50), the linear fits on the 20 `snsx` pairs, the post-processing on all 50 pairs.
- **Settings:** `--recipe cohort` on a 1 mm grid; all cores for both engines (no `--threads`);
  FastWarp's defaults, with no `FASTWARP_*` variable set; post-processing outputs uncompressed
  (`.nii`).
- **Order:** `fetch`, then `run`, `linear`, `post`, `evaluate` and `report`, unattended, by a
  script that `benchmarks/scripts/chain.py` now replaces; on the Apple M5 Pro and the Intel Arc
  140V, `run` and `linear` ran for FastWarp first and then for ANTs (`--engine`).

| GPU | GPU memory | API and driver | CPU | Operating system |
| --- | --- | --- | --- | --- |
| NVIDIA RTX A4000 | 16 GB | Vulkan, NVIDIA | 2 × Intel Xeon Gold 6136 (24 cores, 48 threads) | Ubuntu 24.04 (Linux 6.8) |
| Apple M5 Pro (MacBook Pro) | unified | Metal | Apple M5 Pro (18 cores) | macOS 26.5 |
| AMD Radeon Pro 5500 XT (iMac, 2020) | 8 GB | Metal | Intel Core i7-10700K (8 cores, 16 threads) | macOS 26.6 |
| Intel Arc 140V (Lunar Lake laptop) | shared | Vulkan, Mesa | Intel Core Ultra 7 258V (8 cores) | Ubuntu 24.04 (Linux 6.17) |
| NVIDIA Quadro K620 | 2 GB | Vulkan, NVIDIA | 2 × Intel Xeon E5-2687W v4 (24 cores, 48 threads) | Windows Server 2022 |

### Registration

Seconds per registration (wall time), median (quartiles) over the pairs. ANTs / FastWarp: the
median over pairs of the ratio of the two wall times.

| GPU | Dataset | Pairs | FastWarp (s) | ANTs (s) | ANTs / FastWarp |
| --- | --- | --- | --- | --- | --- |
| NVIDIA RTX A4000 | mindboggle | 20 | 5.3 (5.2–5.4) | 230.2 (222.6–239.1) | 42.8x |
| NVIDIA RTX A4000 | deep7t | 10 | 4.4 (4.2–4.6) | 242.5 (236.0–265.0) | 55.6x |
| NVIDIA RTX A4000 | snsx | 20 | 7.3 (7.2–7.3) | 365.8 (364.4–368.4) | 50.2x |
| Apple M5 Pro | mindboggle | 20 | 2.9 (2.7–2.9) | 91.3 (88.0–95.9) | 32.1x |
| Apple M5 Pro | deep7t | 10 | 2.5 (2.5–2.6) | 105.2 (99.2–116.2) | 41.0x |
| Apple M5 Pro | snsx | 20 | 4.3 (4.3–4.4) | 166.4 (165.8–166.9) | 38.3x |
| AMD Radeon Pro 5500 XT | mindboggle | 20 | 9.7 (9.4–9.9) | 363.4 (344.5–376.3) | 36.8x |
| AMD Radeon Pro 5500 XT | deep7t | 10 | 8.2 (8.0–8.6) | 375.8 (362.9–438.0) | 46.9x |
| AMD Radeon Pro 5500 XT | snsx | 20 | 13.4 (13.2–13.5) | 616.9 (612.6–620.6) | 46.2x |
| Intel Arc 140V (Lunar Lake) | mindboggle | 20 | 8.5 (8.0–8.8) | 352.3 (327.7–434.4) | 40.3x |
| Intel Arc 140V (Lunar Lake) | deep7t | 10 | 7.9 (7.3–8.1) | 355.6 (343.9–394.4) | 46.7x |
| Intel Arc 140V (Lunar Lake) | snsx | 20 | 19.6 (19.4–19.7) | 595.2 (591.8–596.9) | 30.4x |
| NVIDIA Quadro K620 | mindboggle | 20 | 37.4 (35.7–38.7) | 380.6 (367.5–398.4) | 10.3x |
| NVIDIA Quadro K620 | deep7t | 10 | 31.2 (30.3–32.8) | 411.6 (403.8–451.2) | 13.1x |
| NVIDIA Quadro K620 | snsx | 20 | does not fit ¹ | 616.5 (613.3–617.7) | n/a |

¹ `snsx` at 1 mm needs more GPU memory than the Quadro K620's 2 GB: FastWarp's capacity check
refuses these pairs before any GPU work ({class}`~fastwarp.GpuCapacityError`), and the
benchmark records them as skipped.

FastWarp's own split of a registration (median, RTX A4000, `snsx`): 2.5 s for the rigid and
affine stages, 3.3 s for SyN and 7.1 s in total, which includes opening the GPU, resampling,
preprocessing, warping the moving image and writing the transforms. ANTs on the same pairs:
18.8 s, 331.4 s and 350.2 s. The tables per machine below give the split everywhere.

### Linear fits

Seconds per fit on the 20 `snsx` pairs, median (quartiles). FastWarp's preparation (opening the
GPU, converting, winsorizing and uploading the images, the centre-of-mass start) is done once per
{func}`~fastwarp.linear_session`; the rigid + affine column includes it. ANTs: the rigid stage
and then the affine stage, each in its own `antsRegistration` process with its own image
preparation.

| GPU | FastWarp preparation (s) | FastWarp rigid + affine, with preparation (s) | ANTs rigid + affine (s) | ANTs / FastWarp |
| --- | --- | --- | --- | --- |
| NVIDIA RTX A4000 | 0.57 (0.56–0.57) | 3.07 (3.00–3.14) | 19.15 (18.07–19.97) | 6.1x |
| Apple M5 Pro | 0.10 (0.09–0.10) | 1.12 (1.08–1.15) | 7.36 (6.71–7.84) | 6.6x |
| AMD Radeon Pro 5500 XT | 0.28 (0.28–0.29) | 3.71 (3.58–3.84) | 23.44 (21.81–24.25) | 6.2x |
| Intel Arc 140V (Lunar Lake) | 0.25 (0.25–0.26) | 2.89 (2.72–3.00) | 23.85 (21.91–24.41) | 8.0x |
| NVIDIA Quadro K620 | 0.48 (0.47–0.49) | 9.39 (8.90–9.61) | 19.70 (18.46–20.39) | 2.1x |

### Post-processing

Seconds per step on the 20 `snsx` pairs (the forward field and the re-warped image have 8.5
million voxels, the inverse field 26.2 million), median over the pairs, FastWarp / ANTs, with the
median ratio in brackets. Each step includes reading its inputs (from the file cache) and
writing its output: composing the forward and the inverse composite transform into
a displacement field ({func}`~fastwarp.compose_field` against `antsApplyTransforms` with a field
output), and re-warping the moving image with Lanczos interpolation
({func}`~fastwarp.apply_transforms` against `antsApplyTransforms`). The fields of the two tools
differ by at most 7 × 10⁻⁵ mm, the re-warped images by a relative RMS of at most 4 × 10⁻⁵
(median 3 × 10⁻⁶), on all machines and pairs.

| GPU | Forward field: FastWarp / ANTs (s) | Inverse field: FastWarp / ANTs (s) | Lanczos re-warp: FastWarp / ANTs (s) |
| --- | --- | --- | --- |
| NVIDIA RTX A4000 | 0.57 / 0.81 (1.4x) | 1.10 / 1.48 (1.4x) | 1.10 / 1.92 (1.8x) |
| Apple M5 Pro | 0.13 / 0.16 (1.2x) | 0.27 / 0.35 (1.3x) | 0.28 / 0.95 (3.3x) |
| AMD Radeon Pro 5500 XT | 0.35 / 0.51 (1.5x) | 0.78 / 1.16 (1.5x) | 0.74 / 2.23 (3.0x) |
| Intel Arc 140V (Lunar Lake) | 0.33 / 0.46 (1.4x) | 0.64 / 0.90 (1.4x) | 0.78 / 3.99 (5.0x) |
| NVIDIA Quadro K620 | 1.34 / 1.56 (1.2x) | 1.92 / 2.39 (1.2x) | 1.74 / 3.31 (1.9x) |

### Accuracy of the timed runs

The registrations timed here are the ones the [accuracy benchmark](benchmarks.md#results)
scores. Both engines reach nearly the same accuracy on every dataset, and the results hardly
depend on the machine: between any two machines, the Dice score of one pair, engine and label
group differs by at most 0.0021, its 95th-percentile Hausdorff distance by at most 0.05 mm and
the pair's mean landmark error by at most 0.1 mm.

### Tables per machine

The `Registration time`, `Linear fits` and `Post-processing` tables of each machine's
`summary.md`, as `report` wrote them.

:::{dropdown} NVIDIA RTX A4000
**Registration time**

Seconds per registration, median (quartiles); `[n=k]`: only k registrations have the value. Linear: the rigid and affine stages; SyN: the SyN stage; total: the engine's own total (FastWarp: opening the GPU, resampling, preprocessing, both stages, warping the moving image onto the grid and writing the transforms; ANTs: `Total elapsed time`, the stages and their set-up without reading the images or writing the transforms), so the totals do not cover the same work; wall: FastWarp's `register_syn` call (images already read), ANTs' whole process (start, reading, writing; with ANTsPy also starting Python). ANTs' split comes from its `--verbose 1` log. ANTs / FastWarp: the median over pairs of the ratio, on the pairs where both engines have the value and ran on the same machine and software.

| Dataset | Grid | Threads | Engine | Registrations | Linear (s) | SyN (s) | Total (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | fastwarp | 10 | 1.8 (1.8–1.9) | 1.7 (1.6–1.9) | 4.2 (4.1–4.5) | 4.4 (4.2–4.6) |
| deep7t | 1 mm | all cores | ants | 10 | 14.0 (13.3–14.1) | 221.4 (214.2–241.1) | 234.8 (228.1–255.9) | 242.5 (236.0–265.0) |
| deep7t | 1 mm | all cores | ANTs / FastWarp |  | 7.4x (10 pairs) | 128.1x (10 pairs) | 55.4x (10 pairs) | 55.6x (10 pairs) |
| mindboggle | 1 mm | all cores | fastwarp | 20 | 2.3 (2.2–2.4) | 2.0 (1.9–2.1) | 5.2 (5.0–5.2) | 5.3 (5.2–5.4) |
| mindboggle | 1 mm | all cores | ants | 20 | 14.5 (13.7–15.3) | 206.0 (199.2–211.1) | 220.8 (213.8–226.0) | 230.2 (222.6–239.1) |
| mindboggle | 1 mm | all cores | ANTs / FastWarp |  | 6.3x (20 pairs) | 101.9x (20 pairs) | 42.4x (20 pairs) | 42.8x (20 pairs) |
| snsx | 1 mm | all cores | fastwarp | 20 | 2.5 (2.5–2.6) | 3.3 (3.3–3.4) | 7.1 (7.0–7.1) | 7.3 (7.2–7.3) |
| snsx | 1 mm | all cores | ants | 20 | 18.8 (18.0–19.6) | 331.4 (329.9–334.0) | 350.2 (348.6–352.7) | 365.8 (364.4–368.4) |
| snsx | 1 mm | all cores | ANTs / FastWarp |  | 7.3x (20 pairs) | 99.2x (20 pairs) | 49.5x (20 pairs) | 50.2x (20 pairs) |

**Linear fits**

Seconds per fit, median (quartiles). FastWarp: one `linear_session` per pair; its preparation (opening the GPU, converting, winsorizing and uploading the images, the centre-of-mass start) is its own row and not part of the fits; rigid from the centres of mass, affine from the rigid result, then a rigid multi-start. ANTs: one `antsRegistration` stage time each, which includes the stage's own image preparation; no multi-start. Rigid + affine compares FastWarp's preparation and both fits with ANTs' two stages. Pairs: FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio (same machine and software). Landmark error: mean over the landmarks of a pair, mm; FastWarp - ANTs: mean and max distance between the two fits over the fixed brain mask, mm (the multi-start against ANTs' rigid fit).

| Dataset | Grid | Threads | Fit | Pairs | FastWarp (s) | ANTs (s) | ANTs / FastWarp | Landmarks FastWarp (mm) | Landmarks ANTs (mm) | FastWarp - ANTs mean (mm) | FastWarp - ANTs max (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snsx | 1 mm | all cores | session preparation | 20 / 0 | 0.57 (0.56–0.57) | n/a | n/a | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid + affine (FastWarp with its preparation) | 20 / 20 | 3.07 (3.00–3.14) | 19.15 (18.07–19.97) | 6.1x (20 pairs) | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid | 20 / 20 | 1.47 (1.42–1.52) | 6.47 (6.11–6.74) | 4.3x (20 pairs) | 14.43 (9.03–17.32) | 14.47 (10.78–17.04) | 1.15 (0.85–3.65) | 2.38 (1.49–5.76) |
| snsx | 1 mm | all cores | affine (from the rigid fit) | 20 / 20 | 1.05 (0.99–1.07) | 13.10 (11.78–13.52) | 12.5x (20 pairs) | 2.94 (2.47–3.50) | 2.94 (2.46–3.52) | 0.14 (0.10–0.29) | 0.31 (0.20–0.56) |
| snsx | 1 mm | all cores | rigid multi-start | 20 / 0 | 3.02 (2.96–3.15) | n/a | n/a | 12.87 (9.93–16.76) | n/a | 3.18 (1.95–8.92) | 5.92 (4.08–15.73) |

**Post-processing**

Seconds per step, median (quartiles), each including reading the inputs (read once, untimed, before both tools, so both find them cached) and writing the output; ANTs: `antsApplyTransforms --float 1`. `.nii`: uncompressed output; `.nii.gz`: FastWarp's fields gzip level 1 in parallel, ITK (ANTs' outputs and FastWarp's re-warped image) its default level on one thread. Size: median output MB, FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio. Difference (of the `.nii` outputs): median over pairs of the max field difference (mm) or of the relative RMS difference of the re-warped images.

| Dataset | Grid | Threads | Step | Output | Pairs | Voxels | ANTs (s) | FastWarp (s) | ANTs / FastWarp | Size (MB) | Difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | forward field (compose) | .nii | 10 | 14.1 M | 0.86 (0.81–0.94) | 0.57 (0.55–0.61) | 1.5x (10 pairs) | 169 / 169 | max 2.67e-05 mm |
| deep7t | 1 mm | all cores | inverse field (compose) | .nii | 10 | 24.0 M | 1.30 (1.25–1.31) | 0.84 (0.84–0.86) | 1.5x (10 pairs) | 288 / 288 | max 3.52e-05 mm |
| deep7t | 1 mm | all cores | Lanczos re-warp | .nii | 10 | 14.1 M | 2.54 (2.45–2.63) | 1.11 (1.09–1.16) | 2.3x (10 pairs) | 56 / 56 | relative RMS 4.38e-06 |
| mindboggle | 1 mm | all cores | forward field (compose) | .nii | 20 | 4.7 M | 0.38 (0.37–0.41) | 0.23 (0.23–0.25) | 1.6x (20 pairs) | 56 / 56 | max 2.22e-05 mm |
| mindboggle | 1 mm | all cores | inverse field (compose) | .nii | 20 | 10.5 M | 0.62 (0.60–0.64) | 0.40 (0.40–0.41) | 1.5x (20 pairs) | 126 / 126 | max 3.08e-05 mm |
| mindboggle | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 4.7 M | 1.01 (0.99–1.10) | 0.44 (0.40–0.45) | 2.4x (20 pairs) | 19 / 19 | relative RMS 1.70e-06 |
| snsx | 1 mm | all cores | forward field (compose) | .nii | 20 | 8.5 M | 0.81 (0.79–0.92) | 0.57 (0.56–0.58) | 1.4x (20 pairs) | 102 / 102 | max 2.70e-05 mm |
| snsx | 1 mm | all cores | inverse field (compose) | .nii | 20 | 26.2 M | 1.48 (1.47–1.52) | 1.10 (1.09–1.11) | 1.4x (20 pairs) | 314 / 314 | max 4.92e-05 mm |
| snsx | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 8.5 M | 1.92 (1.83–2.04) | 1.10 (1.09–1.11) | 1.8x (20 pairs) | 34 / 34 | relative RMS 3.02e-06 |
:::

:::{dropdown} Apple M5 Pro
**Registration time**

Seconds per registration, median (quartiles); `[n=k]`: only k registrations have the value. Linear: the rigid and affine stages; SyN: the SyN stage; total: the engine's own total (FastWarp: opening the GPU, resampling, preprocessing, both stages, warping the moving image onto the grid and writing the transforms; ANTs: `Total elapsed time`, the stages and their set-up without reading the images or writing the transforms), so the totals do not cover the same work; wall: FastWarp's `register_syn` call (images already read), ANTs' whole process (start, reading, writing; with ANTsPy also starting Python). ANTs' split comes from its `--verbose 1` log. ANTs / FastWarp: the median over pairs of the ratio, on the pairs where both engines have the value and ran on the same machine and software.

| Dataset | Grid | Threads | Engine | Registrations | Linear (s) | SyN (s) | Total (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | fastwarp | 10 | 0.8 (0.8–0.9) | 1.6 (1.6–1.7) | 2.5 (2.5–2.6) | 2.5 (2.5–2.6) |
| deep7t | 1 mm | all cores | ants | 10 | 4.6 (4.3–4.9) | 97.4 (91.7–107.6) | 102.1 (96.2–112.8) | 105.2 (99.2–116.2) |
| deep7t | 1 mm | all cores | ANTs / FastWarp |  | 5.8x (10 pairs) | 60.1x (10 pairs) | 39.9x (10 pairs) | 41.0x (10 pairs) |
| mindboggle | 1 mm | all cores | fastwarp | 20 | 0.9 (0.9–0.9) | 1.8 (1.7–1.8) | 2.8 (2.7–2.9) | 2.9 (2.7–2.9) |
| mindboggle | 1 mm | all cores | ants | 20 | 5.7 (5.2–6.1) | 82.0 (79.4–85.7) | 88.0 (84.8–92.4) | 91.3 (88.0–95.9) |
| mindboggle | 1 mm | all cores | ANTs / FastWarp |  | 6.2x (20 pairs) | 46.2x (20 pairs) | 31.0x (20 pairs) | 32.1x (20 pairs) |
| snsx | 1 mm | all cores | fastwarp | 20 | 1.0 (1.0–1.1) | 3.1 (3.1–3.1) | 4.3 (4.3–4.4) | 4.3 (4.3–4.4) |
| snsx | 1 mm | all cores | ants | 20 | 7.3 (6.9–7.6) | 153.2 (152.8–153.7) | 160.5 (159.9–161.0) | 166.4 (165.8–166.9) |
| snsx | 1 mm | all cores | ANTs / FastWarp |  | 6.8x (20 pairs) | 49.6x (20 pairs) | 37.0x (20 pairs) | 38.3x (20 pairs) |

**Linear fits**

Seconds per fit, median (quartiles). FastWarp: one `linear_session` per pair; its preparation (opening the GPU, converting, winsorizing and uploading the images, the centre-of-mass start) is its own row and not part of the fits; rigid from the centres of mass, affine from the rigid result, then a rigid multi-start. ANTs: one `antsRegistration` stage time each, which includes the stage's own image preparation; no multi-start. Rigid + affine compares FastWarp's preparation and both fits with ANTs' two stages. Pairs: FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio (same machine and software). Landmark error: mean over the landmarks of a pair, mm; FastWarp - ANTs: mean and max distance between the two fits over the fixed brain mask, mm (the multi-start against ANTs' rigid fit).

| Dataset | Grid | Threads | Fit | Pairs | FastWarp (s) | ANTs (s) | ANTs / FastWarp | Landmarks FastWarp (mm) | Landmarks ANTs (mm) | FastWarp - ANTs mean (mm) | FastWarp - ANTs max (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snsx | 1 mm | all cores | session preparation | 20 / 0 | 0.10 (0.09–0.10) | n/a | n/a | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid + affine (FastWarp with its preparation) | 20 / 20 | 1.12 (1.08–1.15) | 7.36 (6.71–7.84) | 6.6x (20 pairs) | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid | 20 / 20 | 0.55 (0.51–0.57) | 3.07 (2.73–3.21) | 5.5x (20 pairs) | 14.12 (10.08–16.57) | 14.47 (11.46–17.04) | 1.26 (0.80–5.73) | 2.22 (1.18–11.02) |
| snsx | 1 mm | all cores | affine (from the rigid fit) | 20 / 20 | 0.47 (0.45–0.49) | 4.29 (3.99–4.58) | 9.2x (20 pairs) | 2.95 (2.46–3.46) | 2.96 (2.47–3.55) | 0.20 (0.12–0.33) | 0.38 (0.24–0.69) |
| snsx | 1 mm | all cores | rigid multi-start | 20 / 0 | 1.31 (1.26–1.37) | n/a | n/a | 12.10 (9.95–16.52) | n/a | 9.67 (1.47–11.97) | 17.27 (3.08–23.90) |

**Post-processing**

Seconds per step, median (quartiles), each including reading the inputs (read once, untimed, before both tools, so both find them cached) and writing the output; ANTs: `antsApplyTransforms --float 1`. `.nii`: uncompressed output; `.nii.gz`: FastWarp's fields gzip level 1 in parallel, ITK (ANTs' outputs and FastWarp's re-warped image) its default level on one thread. Size: median output MB, FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio. Difference (of the `.nii` outputs): median over pairs of the max field difference (mm) or of the relative RMS difference of the re-warped images.

| Dataset | Grid | Threads | Step | Output | Pairs | Voxels | ANTs (s) | FastWarp (s) | ANTs / FastWarp | Size (MB) | Difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | forward field (compose) | .nii | 10 | 14.1 M | 0.18 (0.18–0.19) | 0.14 (0.13–0.15) | 1.3x (10 pairs) | 169 / 169 | max 2.27e-05 mm |
| deep7t | 1 mm | all cores | inverse field (compose) | .nii | 10 | 24.0 M | 0.25 (0.24–0.26) | 0.19 (0.18–0.19) | 1.3x (10 pairs) | 288 / 288 | max 3.45e-05 mm |
| deep7t | 1 mm | all cores | Lanczos re-warp | .nii | 10 | 14.1 M | 1.60 (1.46–1.67) | 0.26 (0.25–0.27) | 6.0x (10 pairs) | 56 / 56 | relative RMS 4.26e-06 |
| mindboggle | 1 mm | all cores | forward field (compose) | .nii | 20 | 4.7 M | 0.05 (0.05–0.06) | 0.04 (0.04–0.04) | 1.5x (20 pairs) | 56 / 56 | max 2.04e-05 mm |
| mindboggle | 1 mm | all cores | inverse field (compose) | .nii | 20 | 10.5 M | 0.10 (0.10–0.11) | 0.07 (0.07–0.07) | 1.4x (20 pairs) | 126 / 126 | max 2.69e-05 mm |
| mindboggle | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 4.7 M | 0.47 (0.44–0.48) | 0.08 (0.08–0.08) | 5.7x (20 pairs) | 19 / 19 | relative RMS 1.63e-06 |
| snsx | 1 mm | all cores | forward field (compose) | .nii | 20 | 8.5 M | 0.16 (0.16–0.16) | 0.13 (0.13–0.13) | 1.2x (20 pairs) | 102 / 102 | max 2.41e-05 mm |
| snsx | 1 mm | all cores | inverse field (compose) | .nii | 20 | 26.2 M | 0.35 (0.35–0.35) | 0.27 (0.27–0.27) | 1.3x (20 pairs) | 314 / 314 | max 4.06e-05 mm |
| snsx | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 8.5 M | 0.95 (0.93–0.96) | 0.28 (0.28–0.29) | 3.3x (20 pairs) | 34 / 34 | relative RMS 3.04e-06 |
:::

:::{dropdown} AMD Radeon Pro 5500 XT
**Registration time**

Seconds per registration, median (quartiles); `[n=k]`: only k registrations have the value. Linear: the rigid and affine stages; SyN: the SyN stage; total: the engine's own total (FastWarp: opening the GPU, resampling, preprocessing, both stages, warping the moving image onto the grid and writing the transforms; ANTs: `Total elapsed time`, the stages and their set-up without reading the images or writing the transforms), so the totals do not cover the same work; wall: FastWarp's `register_syn` call (images already read), ANTs' whole process (start, reading, writing; with ANTsPy also starting Python). ANTs' split comes from its `--verbose 1` log. ANTs / FastWarp: the median over pairs of the ratio, on the pairs where both engines have the value and ran on the same machine and software.

| Dataset | Grid | Threads | Engine | Registrations | Linear (s) | SyN (s) | Total (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | fastwarp | 10 | 2.7 (2.5–2.8) | 5.1 (5.0–5.3) | 8.1 (7.9–8.6) | 8.2 (8.0–8.6) |
| deep7t | 1 mm | all cores | ants | 10 | 15.7 (14.1–16.8) | 354.2 (343.0–412.9) | 369.2 (356.7–430.8) | 375.8 (362.9–438.0) |
| deep7t | 1 mm | all cores | ANTs / FastWarp |  | 5.8x (10 pairs) | 71.4x (10 pairs) | 46.3x (10 pairs) | 46.9x (10 pairs) |
| mindboggle | 1 mm | all cores | fastwarp | 20 | 3.6 (3.3–3.8) | 5.5 (5.3–5.7) | 9.6 (9.4–9.8) | 9.7 (9.4–9.9) |
| mindboggle | 1 mm | all cores | ants | 20 | 18.7 (17.2–21.2) | 335.0 (318.0–350.2) | 356.1 (337.4–368.6) | 363.4 (344.5–376.3) |
| mindboggle | 1 mm | all cores | ANTs / FastWarp |  | 5.4x (20 pairs) | 60.7x (20 pairs) | 36.3x (20 pairs) | 36.8x (20 pairs) |
| snsx | 1 mm | all cores | fastwarp | 20 | 3.5 (3.3–3.6) | 9.1 (9.1–9.2) | 13.3 (13.1–13.4) | 13.4 (13.2–13.5) |
| snsx | 1 mm | all cores | ants | 20 | 24.4 (22.8–24.8) | 580.8 (577.3–583.9) | 604.8 (600.4–608.5) | 616.9 (612.6–620.6) |
| snsx | 1 mm | all cores | ANTs / FastWarp |  | 6.9x (20 pairs) | 63.7x (20 pairs) | 45.5x (20 pairs) | 46.2x (20 pairs) |

**Linear fits**

Seconds per fit, median (quartiles). FastWarp: one `linear_session` per pair; its preparation (opening the GPU, converting, winsorizing and uploading the images, the centre-of-mass start) is its own row and not part of the fits; rigid from the centres of mass, affine from the rigid result, then a rigid multi-start. ANTs: one `antsRegistration` stage time each, which includes the stage's own image preparation; no multi-start. Rigid + affine compares FastWarp's preparation and both fits with ANTs' two stages. Pairs: FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio (same machine and software). Landmark error: mean over the landmarks of a pair, mm; FastWarp - ANTs: mean and max distance between the two fits over the fixed brain mask, mm (the multi-start against ANTs' rigid fit).

| Dataset | Grid | Threads | Fit | Pairs | FastWarp (s) | ANTs (s) | ANTs / FastWarp | Landmarks FastWarp (mm) | Landmarks ANTs (mm) | FastWarp - ANTs mean (mm) | FastWarp - ANTs max (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snsx | 1 mm | all cores | session preparation | 20 / 0 | 0.28 (0.28–0.29) | n/a | n/a | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid + affine (FastWarp with its preparation) | 20 / 20 | 3.71 (3.58–3.84) | 23.44 (21.81–24.25) | 6.2x (20 pairs) | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid | 20 / 20 | 1.81 (1.72–1.85) | 10.03 (9.44–11.18) | 5.9x (20 pairs) | 13.81 (10.08–17.04) | 13.95 (10.01–17.04) | 1.32 (0.77–2.09) | 2.50 (1.40–3.79) |
| snsx | 1 mm | all cores | affine (from the rigid fit) | 20 / 20 | 1.61 (1.54–1.71) | 12.85 (12.21–13.44) | 7.6x (20 pairs) | 2.93 (2.45–3.51) | 2.94 (2.45–3.41) | 0.13 (0.10–0.26) | 0.31 (0.25–0.56) |
| snsx | 1 mm | all cores | rigid multi-start | 20 / 0 | 4.42 (4.33–4.62) | n/a | n/a | 13.37 (9.95–16.66) | n/a | 4.75 (2.13–9.54) | 7.98 (4.10–19.32) |

**Post-processing**

Seconds per step, median (quartiles), each including reading the inputs (read once, untimed, before both tools, so both find them cached) and writing the output; ANTs: `antsApplyTransforms --float 1`. `.nii`: uncompressed output; `.nii.gz`: FastWarp's fields gzip level 1 in parallel, ITK (ANTs' outputs and FastWarp's re-warped image) its default level on one thread. Size: median output MB, FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio. Difference (of the `.nii` outputs): median over pairs of the max field difference (mm) or of the relative RMS difference of the re-warped images.

| Dataset | Grid | Threads | Step | Output | Pairs | Voxels | ANTs (s) | FastWarp (s) | ANTs / FastWarp | Size (MB) | Difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | forward field (compose) | .nii | 10 | 14.1 M | 0.64 (0.63–0.68) | 0.43 (0.41–0.46) | 1.5x (10 pairs) | 169 / 169 | max 2.62e-05 mm |
| deep7t | 1 mm | all cores | inverse field (compose) | .nii | 10 | 24.0 M | 0.89 (0.88–0.90) | 0.60 (0.59–0.63) | 1.4x (10 pairs) | 288 / 288 | max 3.85e-05 mm |
| deep7t | 1 mm | all cores | Lanczos re-warp | .nii | 10 | 14.1 M | 3.40 (3.33–3.69) | 0.92 (0.88–0.96) | 3.8x (10 pairs) | 56 / 56 | relative RMS 4.49e-06 |
| mindboggle | 1 mm | all cores | forward field (compose) | .nii | 20 | 4.7 M | 0.26 (0.25–0.27) | 0.15 (0.14–0.16) | 1.7x (20 pairs) | 56 / 56 | max 2.13e-05 mm |
| mindboggle | 1 mm | all cores | inverse field (compose) | .nii | 20 | 10.5 M | 0.46 (0.44–0.48) | 0.28 (0.27–0.29) | 1.6x (20 pairs) | 126 / 126 | max 3.15e-05 mm |
| mindboggle | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 4.7 M | 1.25 (1.19–1.32) | 0.33 (0.32–0.36) | 3.6x (20 pairs) | 19 / 19 | relative RMS 1.68e-06 |
| snsx | 1 mm | all cores | forward field (compose) | .nii | 20 | 8.5 M | 0.51 (0.51–0.52) | 0.35 (0.34–0.35) | 1.5x (20 pairs) | 102 / 102 | max 2.68e-05 mm |
| snsx | 1 mm | all cores | inverse field (compose) | .nii | 20 | 26.2 M | 1.16 (1.14–1.18) | 0.78 (0.77–0.79) | 1.5x (20 pairs) | 314 / 314 | max 4.50e-05 mm |
| snsx | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 8.5 M | 2.23 (2.19–2.27) | 0.74 (0.73–0.76) | 3.0x (20 pairs) | 34 / 34 | relative RMS 3.10e-06 |
:::

:::{dropdown} Intel Arc 140V (Lunar Lake)
**Registration time**

Seconds per registration, median (quartiles); `[n=k]`: only k registrations have the value. Linear: the rigid and affine stages; SyN: the SyN stage; total: the engine's own total (FastWarp: opening the GPU, resampling, preprocessing, both stages, warping the moving image onto the grid and writing the transforms; ANTs: `Total elapsed time`, the stages and their set-up without reading the images or writing the transforms), so the totals do not cover the same work; wall: FastWarp's `register_syn` call (images already read), ANTs' whole process (start, reading, writing; with ANTsPy also starting Python). ANTs' split comes from its `--verbose 1` log. ANTs / FastWarp: the median over pairs of the ratio, on the pairs where both engines have the value and ran on the same machine and software.

| Dataset | Grid | Threads | Engine | Registrations | Linear (s) | SyN (s) | Total (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | fastwarp | 10 | 2.3 (2.2–2.4) | 5.0 (4.8–5.2) | 7.8 (7.3–8.0) | 7.9 (7.3–8.1) |
| deep7t | 1 mm | all cores | ants | 10 | 15.8 (14.4–16.7) | 335.6 (323.4–370.9) | 350.7 (339.0–388.9) | 355.6 (343.9–394.4) |
| deep7t | 1 mm | all cores | ANTs / FastWarp |  | 6.8x (10 pairs) | 66.6x (10 pairs) | 46.4x (10 pairs) | 46.7x (10 pairs) |
| mindboggle | 1 mm | all cores | fastwarp | 20 | 2.6 (2.5–2.7) | 5.4 (5.2–5.5) | 8.4 (7.9–8.8) | 8.5 (8.0–8.8) |
| mindboggle | 1 mm | all cores | ants | 20 | 20.5 (18.7–33.0) | 326.2 (301.8–389.9) | 345.9 (322.2–425.6) | 352.3 (327.7–434.4) |
| mindboggle | 1 mm | all cores | ANTs / FastWarp |  | 7.5x (20 pairs) | 58.7x (20 pairs) | 40.0x (20 pairs) | 40.3x (20 pairs) |
| snsx | 1 mm | all cores | fastwarp | 20 | 2.9 (2.8–3.2) | 15.9 (15.8–16.0) | 19.5 (19.3–19.6) | 19.6 (19.4–19.7) |
| snsx | 1 mm | all cores | ants | 20 | 24.0 (21.7–24.8) | 561.1 (559.4–562.8) | 585.1 (581.8–586.9) | 595.2 (591.8–596.9) |
| snsx | 1 mm | all cores | ANTs / FastWarp |  | 7.9x (20 pairs) | 35.3x (20 pairs) | 30.0x (20 pairs) | 30.4x (20 pairs) |

**Linear fits**

Seconds per fit, median (quartiles). FastWarp: one `linear_session` per pair; its preparation (opening the GPU, converting, winsorizing and uploading the images, the centre-of-mass start) is its own row and not part of the fits; rigid from the centres of mass, affine from the rigid result, then a rigid multi-start. ANTs: one `antsRegistration` stage time each, which includes the stage's own image preparation; no multi-start. Rigid + affine compares FastWarp's preparation and both fits with ANTs' two stages. Pairs: FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio (same machine and software). Landmark error: mean over the landmarks of a pair, mm; FastWarp - ANTs: mean and max distance between the two fits over the fixed brain mask, mm (the multi-start against ANTs' rigid fit).

| Dataset | Grid | Threads | Fit | Pairs | FastWarp (s) | ANTs (s) | ANTs / FastWarp | Landmarks FastWarp (mm) | Landmarks ANTs (mm) | FastWarp - ANTs mean (mm) | FastWarp - ANTs max (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snsx | 1 mm | all cores | session preparation | 20 / 0 | 0.25 (0.25–0.26) | n/a | n/a | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid + affine (FastWarp with its preparation) | 20 / 20 | 2.89 (2.72–3.00) | 23.85 (21.91–24.41) | 8.0x (20 pairs) | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid | 20 / 20 | 1.45 (1.38–1.51) | 10.04 (8.96–10.72) | 6.7x (20 pairs) | 13.65 (9.71–17.02) | 13.95 (10.85–17.04) | 1.32 (0.78–2.30) | 2.36 (1.34–4.51) |
| snsx | 1 mm | all cores | affine (from the rigid fit) | 20 / 20 | 1.19 (1.10–1.26) | 13.43 (12.85–14.21) | 11.3x (20 pairs) | 2.91 (2.47–3.49) | 2.92 (2.47–3.46) | 0.17 (0.08–0.22) | 0.35 (0.19–0.46) |
| snsx | 1 mm | all cores | rigid multi-start | 20 / 0 | 3.13 (2.95–3.29) | n/a | n/a | 12.84 (9.91–16.95) | n/a | 3.48 (2.28–10.39) | 6.74 (3.90–21.42) |

**Post-processing**

Seconds per step, median (quartiles), each including reading the inputs (read once, untimed, before both tools, so both find them cached) and writing the output; ANTs: `antsApplyTransforms --float 1`. `.nii`: uncompressed output; `.nii.gz`: FastWarp's fields gzip level 1 in parallel, ITK (ANTs' outputs and FastWarp's re-warped image) its default level on one thread. Size: median output MB, FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio. Difference (of the `.nii` outputs): median over pairs of the max field difference (mm) or of the relative RMS difference of the re-warped images.

| Dataset | Grid | Threads | Step | Output | Pairs | Voxels | ANTs (s) | FastWarp (s) | ANTs / FastWarp | Size (MB) | Difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | forward field (compose) | .nii | 10 | 14.1 M | 0.52 (0.49–0.55) | 0.35 (0.34–0.38) | 1.5x (10 pairs) | 169 / 169 | max 2.61e-05 mm |
| deep7t | 1 mm | all cores | inverse field (compose) | .nii | 10 | 24.0 M | 0.70 (0.69–0.71) | 0.48 (0.47–0.50) | 1.5x (10 pairs) | 288 / 288 | max 3.92e-05 mm |
| deep7t | 1 mm | all cores | Lanczos re-warp | .nii | 10 | 14.1 M | 6.69 (6.21–7.19) | 0.90 (0.87–0.95) | 7.4x (10 pairs) | 56 / 56 | relative RMS 4.42e-06 |
| mindboggle | 1 mm | all cores | forward field (compose) | .nii | 20 | 4.7 M | 0.22 (0.21–0.23) | 0.13 (0.13–0.14) | 1.6x (20 pairs) | 56 / 56 | max 2.25e-05 mm |
| mindboggle | 1 mm | all cores | inverse field (compose) | .nii | 20 | 10.5 M | 0.34 (0.33–0.34) | 0.22 (0.22–0.23) | 1.5x (20 pairs) | 126 / 126 | max 3.03e-05 mm |
| mindboggle | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 4.7 M | 2.26 (2.12–2.33) | 0.32 (0.31–0.33) | 6.9x (20 pairs) | 19 / 19 | relative RMS 1.70e-06 |
| snsx | 1 mm | all cores | forward field (compose) | .nii | 20 | 8.5 M | 0.46 (0.46–0.46) | 0.33 (0.33–0.34) | 1.4x (20 pairs) | 102 / 102 | max 2.83e-05 mm |
| snsx | 1 mm | all cores | inverse field (compose) | .nii | 20 | 26.2 M | 0.90 (0.90–0.91) | 0.64 (0.63–0.65) | 1.4x (20 pairs) | 314 / 314 | max 4.78e-05 mm |
| snsx | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 8.5 M | 3.99 (3.85–4.02) | 0.78 (0.77–0.79) | 5.0x (20 pairs) | 34 / 34 | relative RMS 3.08e-06 |
:::

:::{dropdown} NVIDIA Quadro K620
**Registration time**

Seconds per registration, median (quartiles); `[n=k]`: only k registrations have the value. Linear: the rigid and affine stages; SyN: the SyN stage; total: the engine's own total (FastWarp: opening the GPU, resampling, preprocessing, both stages, warping the moving image onto the grid and writing the transforms; ANTs: `Total elapsed time`, the stages and their set-up without reading the images or writing the transforms), so the totals do not cover the same work; wall: FastWarp's `register_syn` call (images already read), ANTs' whole process (start, reading, writing; with ANTsPy also starting Python). ANTs' split comes from its `--verbose 1` log. ANTs / FastWarp: the median over pairs of the ratio, on the pairs where both engines have the value and ran on the same machine and software.

| Dataset | Grid | Threads | Engine | Registrations | Linear (s) | SyN (s) | Total (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | fastwarp | 10 | 5.2 (4.9–5.4) | 25.1 (24.5–26.6) | 31.0 (30.1–32.7) | 31.2 (30.3–32.8) |
| deep7t | 1 mm | all cores | ants | 10 | 14.4 (13.1–15.4) | 386.8 (377.1–423.1) | 400.2 (392.4–438.5) | 411.6 (403.8–451.2) |
| deep7t | 1 mm | all cores | ANTs / FastWarp |  | 2.8x (10 pairs) | 15.2x (10 pairs) | 12.8x (10 pairs) | 13.1x (10 pairs) |
| mindboggle | 1 mm | all cores | fastwarp | 20 | 7.1 (6.8–7.3) | 29.4 (27.6–30.3) | 37.2 (35.5–38.5) | 37.4 (35.7–38.7) |
| mindboggle | 1 mm | all cores | ants | 20 | 16.1 (15.2–17.1) | 351.3 (339.0–368.9) | 367.3 (354.7–385.3) | 380.6 (367.5–398.4) |
| mindboggle | 1 mm | all cores | ANTs / FastWarp |  | 2.3x (20 pairs) | 12.1x (20 pairs) | 10.0x (20 pairs) | 10.3x (20 pairs) |
| snsx | 1 mm | all cores | ants | 20 | 20.7 (18.7–21.5) | 575.8 (574.6–576.2) | 596.4 (593.1–597.5) | 616.5 (613.3–617.7) |

**Linear fits**

Seconds per fit, median (quartiles). FastWarp: one `linear_session` per pair; its preparation (opening the GPU, converting, winsorizing and uploading the images, the centre-of-mass start) is its own row and not part of the fits; rigid from the centres of mass, affine from the rigid result, then a rigid multi-start. ANTs: one `antsRegistration` stage time each, which includes the stage's own image preparation; no multi-start. Rigid + affine compares FastWarp's preparation and both fits with ANTs' two stages. Pairs: FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio (same machine and software). Landmark error: mean over the landmarks of a pair, mm; FastWarp - ANTs: mean and max distance between the two fits over the fixed brain mask, mm (the multi-start against ANTs' rigid fit).

| Dataset | Grid | Threads | Fit | Pairs | FastWarp (s) | ANTs (s) | ANTs / FastWarp | Landmarks FastWarp (mm) | Landmarks ANTs (mm) | FastWarp - ANTs mean (mm) | FastWarp - ANTs max (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snsx | 1 mm | all cores | session preparation | 20 / 0 | 0.48 (0.47–0.49) | n/a | n/a | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid + affine (FastWarp with its preparation) | 20 / 20 | 9.39 (8.90–9.61) | 19.70 (18.46–20.39) | 2.1x (20 pairs) | n/a | n/a | n/a | n/a |
| snsx | 1 mm | all cores | rigid | 20 / 20 | 5.12 (4.74–5.27) | 8.65 (8.02–9.12) | 1.8x (20 pairs) | 13.93 (10.24–17.37) | 14.48 (11.50–17.04) | 1.04 (0.49–2.41) | 2.07 (0.94–4.05) |
| snsx | 1 mm | all cores | affine (from the rigid fit) | 20 / 20 | 3.78 (3.54–3.94) | 11.20 (10.81–11.70) | 2.9x (20 pairs) | 2.95 (2.47–3.53) | 2.95 (2.47–3.53) | 0.15 (0.11–0.24) | 0.34 (0.23–0.48) |
| snsx | 1 mm | all cores | rigid multi-start | 20 / 0 | 7.80 (7.03–8.07) | n/a | n/a | 12.50 (9.99–16.68) | n/a | 6.11 (1.92–10.29) | 11.92 (3.87–19.10) |

**Post-processing**

Seconds per step, median (quartiles), each including reading the inputs (read once, untimed, before both tools, so both find them cached) and writing the output; ANTs: `antsApplyTransforms --float 1`. `.nii`: uncompressed output; `.nii.gz`: FastWarp's fields gzip level 1 in parallel, ITK (ANTs' outputs and FastWarp's re-warped image) its default level on one thread. Size: median output MB, FastWarp / ANTs. ANTs / FastWarp: the median over pairs of the ratio. Difference (of the `.nii` outputs): median over pairs of the max field difference (mm) or of the relative RMS difference of the re-warped images.

| Dataset | Grid | Threads | Step | Output | Pairs | Voxels | ANTs (s) | FastWarp (s) | ANTs / FastWarp | Size (MB) | Difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep7t | 1 mm | all cores | forward field (compose) | .nii | 10 | 14.1 M | 0.85 (0.83–0.93) | 0.59 (0.57–0.63) | 1.5x (10 pairs) | 169 / 169 | max 2.59e-05 mm |
| deep7t | 1 mm | all cores | inverse field (compose) | .nii | 10 | 24.0 M | 4.03 (3.22–4.40) | 0.86 (0.85–0.86) | 4.7x (10 pairs) | 288 / 288 | max 3.67e-05 mm |
| deep7t | 1 mm | all cores | Lanczos re-warp | .nii | 10 | 14.1 M | 3.73 (3.57–3.99) | 0.89 (0.88–0.95) | 4.1x (10 pairs) | 56 / 56 | relative RMS 4.50e-06 |
| mindboggle | 1 mm | all cores | forward field (compose) | .nii | 20 | 4.7 M | 0.35 (0.34–0.37) | 0.25 (0.24–0.25) | 1.5x (20 pairs) | 56 / 56 | max 2.20e-05 mm |
| mindboggle | 1 mm | all cores | inverse field (compose) | .nii | 20 | 10.5 M | 0.62 (0.60–0.64) | 0.41 (0.40–0.42) | 1.5x (20 pairs) | 126 / 126 | max 3.22e-05 mm |
| mindboggle | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 4.7 M | 1.40 (1.36–1.50) | 0.38 (0.36–0.40) | 3.8x (20 pairs) | 19 / 19 | relative RMS 1.70e-06 |
| snsx | 1 mm | all cores | forward field (compose) | .nii | 20 | 8.5 M | 1.56 (1.55–1.61) | 1.34 (1.34–1.37) | 1.2x (20 pairs) | 102 / 102 | max 2.73e-05 mm |
| snsx | 1 mm | all cores | inverse field (compose) | .nii | 20 | 26.2 M | 2.39 (2.36–2.44) | 1.92 (1.91–1.95) | 1.2x (20 pairs) | 314 / 314 | max 4.52e-05 mm |
| snsx | 1 mm | all cores | Lanczos re-warp | .nii | 20 | 8.5 M | 3.31 (3.23–3.38) | 1.74 (1.73–1.76) | 1.9x (20 pairs) | 34 / 34 | relative RMS 3.13e-06 |
:::
