# The CPU engine

FastWarp registers on the GPU by default. For a computer without a usable GPU, for example a
cluster node without one, it also has a CPU engine. You choose it yourself: FastWarp does not
switch to it without telling you.

The CPU engine is the code the GPU kernels are checked against: the same registration, written
in 64-bit floating point and run on all host threads. It is **much slower than the GPU**, and its
results agree with the GPU's **within tolerance, not bit for bit**.

## Choosing the engine

| Where | How |
| --- | --- |
| Python | `engine="cpu"`, `"gpu"` or `"auto"` on {func}`~fastwarp.register_syn`, {func}`~fastwarp.register_atlases`, {func}`~fastwarp.linear_session` and {func}`fastwarp.ants.registration` |
| Command line | `--engine cpu`, `gpu` or `auto` on `register`, `linear`, `fits` and `bench` |
| Environment | `FASTWARP_ENGINE=cpu`, `gpu` or `auto`; also for `fastwarp ants-registration`, which takes only ANTs arguments |

An argument wins over the variable. Without either, the engine is the GPU, as before.

- `"gpu"`: the GPU. Without a usable GPU, {class}`~fastwarp.GpuUnavailableError` (exit status
  4), before any work.
- `"cpu"`: the CPU engine. `adapter=` (`--adapter`) picks a GPU, so it is refused with it.
- `"auto"`: the GPU when there is a usable one, otherwise the CPU engine. An `adapter` that
  matches no GPU adapter is not a reason to fall back: it raises
  {class}`~fastwarp.AdapterNotFoundError` (exit status 8), as with `"gpu"`. The switch is
  announced before the run starts: a `UserWarning` in Python, a `fastwarp: warning:` line on
  standard error on the command line. With warnings turned into errors
  (`python -W error`), the warning stops the call before any work. A GPU that is present but
  too small still raises {class}`~fastwarp.GpuCapacityError`: `"auto"` does not move a
  registration that does not fit the GPU to the CPU. {func}`~fastwarp.register_atlases` decides
  once for all its atlases.

The result records the engine that ran: `engine` is `"gpu"` or `"cpu"`, `adapter` reads
`cpu (f64, 8 threads)` on the CPU engine, and `peak_gpu_bytes` and `memory_budget_bytes` are
`None` (`null` in the command line's JSON). Report the engine and the operating system with
results from the CPU engine (see [Results](#results)).

## Speed

The CPU engine is not optimised yet; it is the reference implementation. One measurement on a
laptop (Intel Core Ultra 7 268V, 8 threads; GPU: its integrated Arc 140V), the synthetic head
of `fastwarp bench --size mni-2mm` (91 × 109 × 91 voxels at 2 mm), default recipe:

| Engine | Linear stages | SyN | Total |
| --- | --- | --- | --- |
| GPU (Arc 140V, Vulkan) | 0.9 s | 1.3 s | 2.3 s |
| CPU engine (8 threads) | 2.9 s | 49.4 s | 52.4 s |

That is about 20 times slower overall and about 40 times slower in SyN.

On real 1 mm brain images, measured on a 28-core AMD EPYC server (one pair, default recipe):

| Engine | Time per pair | Peak memory |
| --- | --- | --- |
| CPU engine (28 threads) | 111-122 s (linear stages about 11 s, SyN about 105 s) | about 2.6 GiB |
| GPU (datacenter GPU) | about 2.3 s | |
| ANTs 2.6.5 (same 28 cores) | about 260-280 s | |

Multi-core use is limited for now: on average about 7-8 of the 28 cores were busy.
`fastwarp bench --engine cpu --size mni-2mm` measures your machine.

The CPU engine uses every logical core, or `RAYON_NUM_THREADS` threads when that variable is
set before the first FastWarp call. On a shared cluster node, set it to the cores you were given
(with Slurm, `RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK`). {func}`~fastwarp.register_atlases`
registers one atlas at a time on the CPU engine, since each registration already uses every
core; do not run several CPU registrations in parallel processes on the same cores.

## Memory

The CPU engine keeps its images and fields in 64-bit floating point in the computer's memory.
Before any work it estimates what the registration needs, about 512 bytes per voxel of the
{term}`registration grid` for one channel with a SyN stage (more with further channels, masks or
several SyN stages, far less without a SyN stage), and refuses the run with
{class}`~fastwarp.HostMemoryError` (exit status 7) when that is more than the memory the computer
has available at that moment. For the 1 mm MNI grid (182 × 218 × 182 voxels) the estimate is about
3.4 GiB, at 0.5 mm about 28 GiB. A measured 1 mm brain pair on a grid of about 4.9 million voxels
(about 190-200 × 157-166 × 148-160) peaked at about 2.6 GiB for the whole process, of which about
0.5 GiB was the import and the loaded images. The input images are not counted in the estimate,
since they are already in memory. On Linux the available memory honours the cgroup limit of a
container or a Slurm job; on macOS FastWarp cannot read it, so the check is skipped there. `fastwarp
fits --engine cpu` runs the same check without registering.

## Results

- **CPU engine against the GPU:** the same algorithm, in 64-bit instead of 32-bit floating point.
  The parity tests hold single operations to tolerances such as 1e-5 for a warp, and whole SyN
  runs on small synthetic images to a difference of under 5 % in the warped image; results are
  not bit-identical, and the fields can differ more on some pairs. The 64-bit arithmetic is the
  precision ANTs uses as well.
- **CPU engine against itself:** its sums run in a fixed order, so it is designed to give the
  same result for any number of threads on the same operating system and processor architecture.
  Linux and Windows give slightly different bits (the reference results of the tests differ),
  most likely because their maths libraries differ.
- **Validation:** the comparison of FastWarp with ANTs on real images ([Validation](../validation.md))
  covers the GPU engine. On 5 Mindboggle pairs, the segmentation accuracy (Dice) of the CPU
  engine agreed with that of the GPU engine and of ANTs within rounding (cortical 0.5751 on the CPU
  engine, 0.5751 on the GPU engine, 0.5750 with ANTs).

## Software GPUs are not a fallback

Software adapters that emulate a GPU on the CPU (Mesa's lavapipe, Microsoft WARP) run FastWarp's
GPU kernels too, with `FASTWARP_GPU_ALLOW_SOFTWARE=1`, but far slower than the CPU engine: WARP
took 38 s for the `tiny` bench size, where the CPU engine took 0.8 s. They remain what the tests
use on machines without a GPU; for real images, use the CPU engine.
