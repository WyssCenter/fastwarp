# What FastWarp does not do

FastWarp covers one part of ANTs: the registration recipe of Wyss DBS Localizer and the
resampling around it. If you need something on this list, use ANTs or ANTsPy for that step.

## Images

- **No image input from files.** Pass images in memory (see [Images](images.md)), for example
  read with ANTsPy. FastWarp reads and writes transform files and displacement fields; the only
  image it writes is the optional `_Warped.nii.gz` of {func}`~fastwarp.register_syn`.
- **3-D only.** 2-D images, time series and vector or multi-component images are not supported.
- **One or two channels.** {func}`~fastwarp.register_syn` registers one image, or two images of
  the same subject, per side.

## Registration

- **A GPU is required.** The registration functions do not run on the CPU. Without a usable GPU
  they raise {class}`~fastwarp.GpuUnavailableError` (see [Falling back to ANTs](gpu.md#fall-back-to-ants)).
- **Three transforms.** {term}`Rigid`, {term}`affine` and {term}`SyN`. ANTs' other transforms,
  such as `Similarity`, `Translation`, `BSplineSyN` or `TimeVaryingVelocityField`, are not
  available.
- **Two metrics, with limited settings.** {term}`Mattes mutual information` with exactly 32
  histogram bins and regularly spaced samples for the linear stages, and
  {term}`cross-correlation` with a window radius from 1 to 8 voxels for SyN. ANTs' other metrics (`MeanSquares`, `GC`, `Demons`,
  random sampling) are not available.
- **Fixed preparation.** {func}`~fastwarp.register_syn` always works on an isotropic
  {term}`registration grid`, always {term}`winsorizes <winsorize>` the intensities to the 0.005
  and 0.995 quantiles, and never matches histograms.
- **No moving-image mask.** A mask limits where the fixed image is sampled, as ANTs'
  `--masks [fixed,NULL]` does; there is no mask for the moving image. {func}`~fastwarp.register_syn`
  takes no mask at all; only the [linear fits](linear.md#masks) do.
- **One stage per linear fit.** A {class}`~fastwarp.LinearSession` fit runs one rigid or one
  affine stage; chain several fits yourself.
- **No bias-field correction.** Run N4 (for example `ants.n4_bias_field_correction`) before
  FastWarp if your images need it.
- **Not bit-identical to ANTs.** The results agree closely with ANTs but differ in the last
  digits; see [Troubleshooting](../troubleshooting.md#results-differ-slightly-from-ants).

## Resampling

- **Three interpolators.** Linear, nearest neighbour and Lanczos windowed sinc. ANTs' `BSpline`,
  `Gaussian` and `GenericLabel` interpolators are not available.
- **No inversion inside a transform list.** {func}`~fastwarp.apply_transforms` and
  {func}`~fastwarp.compose_field` take file paths only; use the inverse file, such as
  `InverseComposite.h5`, instead of ANTs' `[file,1]`. FastWarp writes no inverse of
  `0GenericAffine.mat`, so applying the inverse affine alone needs ANTs.
- **No point transforms.** To move points, use `antsApplyTransformsToPoints` or
  `ants.apply_transforms_to_points` with the [right file](outputs.md).
