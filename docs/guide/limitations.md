# What FastWarp does not do

FastWarp covers one part of ANTs: the registration recipe of Wyss DBS Localizer and the
resampling around it. If you need something on this list, use ANTs or ANTsPy for that step.

## Images

- **Images in memory, or NIfTI-1 files.** The Python functions take images in memory (see
  [Images](images.md)), for example read with ANTsPy. The [command line](../cli.md) reads and
  writes NIfTI-1 files (`.nii`, `.nii.gz`) only; other formats (`.mha`, `.nrrd`, Analyze,
  NIfTI-2) must be converted first.
- **3D only.** 2-D images, time series and vector or multi-component images are not supported.
- **Channels need GPU memory.** {func}`~fastwarp.register_syn` registers one image, or several
  images of the same subject, per side; every channel adds GPU memory, and that is the only limit.

## Registration

- **A GPU is required.** The registration functions do not run on the CPU. Without a usable GPU
  they raise {class}`~fastwarp.GpuUnavailableError` (see [Falling back to ANTs](gpu.md#fall-back-to-ants)).
- **Three transforms, in a fixed order.** {term}`Rigid`, {term}`affine` and {term}`SyN`: any
  number of rigid and affine stages, then any number of SyN stages that end with the same shrink
  factor (see [Stage lists](recipes.md#stage-lists)). ANTs' other transforms, such as
  `Similarity`, `Translation`, `BSplineSyN` or `TimeVaryingVelocityField`, and a linear stage
  after SyN are not available. With three or more SyN stages, a stage sees the earlier ones
  through their collapsed field rather than one after the other as in ANTs (see
  [Several SyN stages](recipes.md#several-syn-stages)).
- **Two metrics, with limited settings.** {term}`Mattes mutual information` with exactly 32
  histogram bins and regularly spaced samples for the linear stages; for SyN, per channel,
  {term}`cross-correlation` with a window radius from 1 to 8 voxels or Mattes mutual information
  with 5 to 32 bins and dense sampling. All CC channels of a SyN stage share one radius and all MI
  channels one number of bins. ANTs' other metrics (`MeanSquares`, `GC`, `Demons`), random
  sampling, and sampled metrics in SyN stages are not available.
- **Fixed preparation.** {func}`~fastwarp.register_syn` always works on an isotropic
  {term}`registration grid` and never matches histograms. It either
  {term}`winsorizes <winsorize>` the intensities to the quantiles the recipe gives, or only
  rescales them to [0, 1].
- **Collapsed outputs only.** FastWarp writes the combined files of ANTs'
  `--collapse-output-transforms 1`, not the per-stage files of `-z 0`. A SyN-only run without
  a start (ANTs' `0Warp.nii.gz` layout) is not produced: FastWarp always writes a
  `0GenericAffine.mat`.
- **The moving-image mask applies to SyN stages only, and a call has one mask per side.**
  {func}`~fastwarp.register_syn` takes one fixed mask for every stage and one moving mask for the
  SyN stages (see [Masks](normalization.md#masks)), as ANTs' `--masks [fixed,moving]` given once;
  a stage-list recipe can switch either off per stage ([Mask switches](recipes.md#mask-switches)),
  but not give a stage a mask of its own. The rigid and affine stages do not use a moving mask;
  one given for them (`fastwarp ants-registration -x` on a linear stage, or a `fastwarp.ants`
  call that would put it there) is refused, and `fastwarp ants-registration` refuses two
  different fixed or moving mask files across its SyN stages. A moving mask on the linear stages
  will follow.
- **One stage per linear fit.** A {class}`~fastwarp.LinearSession` fit runs one rigid or one
  affine stage; chain several fits yourself.
- **No bias-field correction.** Run N4 (for example `ants.n4_bias_field_correction`) before
  FastWarp if your images need it.
- **Not bit-identical to ANTs.** The results agree closely with ANTs but differ in the last
  digits; see [Troubleshooting](../troubleshooting.md#results-differ-slightly-from-ants).

## Resampling

- **Three interpolators.** Linear, nearest neighbour and Lanczos windowed sinc. ANTs' `BSpline`,
  `Gaussian` and `GenericLabel` interpolators are not available.
- **Only linear transforms can be inverted.** `invert` (and `[file,1]` on the command line)
  inverts affine files and composites of affines. A displacement field cannot be inverted inside
  a transform list; use the inverse file FastWarp writes, such as `InverseComposite.h5` or
  `1InverseWarp.nii.gz`.
- **ANTsPy only in part.** [`fastwarp.ants`](ants.md) runs ANTsPy's registration and
  resampling calls within these limits, including the default `registration(fixed, moving)`,
  and refuses the rest.
- **Label fusion on the CPU by default.** {func}`~fastwarp.joint_label_fusion` runs on the CPU;
  `mode="hybrid"` and `"gpu"` move the search (and with `"gpu"` the weights and votes) to the
  GPU, which can rank near-tied patches differently and rounds in `float32`
  ([Label fusion](label-fusion.md#running-on-the-gpu)). For many atlases and large search windows
  it takes minutes, so restrict it with a mask. `fastwarp ants-joint-fusion` always runs on the
  CPU.
- **No point transforms.** To move points, use `antsApplyTransformsToPoints` or
  `ants.apply_transforms_to_points` with the [right file](outputs.md).
