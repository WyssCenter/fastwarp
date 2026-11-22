FastWarp registration from the browser
======================================

This archive holds the result of one registration of a moving image onto a
fixed image, run by FastWarp in a web browser on the computer's GPU (WebGPU).
The recipe is the default of the fastwarp command line: centre of mass, Rigid,
Affine (Mattes mutual information), then SyN (neighbourhood cross-correlation),
as in an antsRegistration call with the same stages. The transform files have
the names and formats that antsRegistration writes with
--collapse-output-transforms 1 --write-composite-transform 0, so ANTs, ANTsPy
and FastWarp read them.

Files
-----

0GenericAffine.mat    The affine part (centre of mass, Rigid and Affine in
                      one), ITK MATLAB format.
1Warp.nii.gz          The SyN displacement field, fixed image to moving image.
1InverseWarp.nii.gz   Its inverse, moving image to fixed image.
Warped.nii.gz         The moving image resampled onto the fixed image's grid
                      (linear interpolation).
run.json              Settings, FastWarp version, GPU, device limits, grids,
                      timings and the iterations of each stage.
README.txt            This file.

The fields lie on the registration grid (1 mm or 2 mm, isotropic, over the
fixed image), which can differ from the fixed image's own grid; ANTs and
FastWarp interpolate them, so the files work with any reference image.

Applying the transforms
-----------------------

Moving image onto the fixed image (what Warped.nii.gz holds):

  antsApplyTransforms -d 3 -i moving.nii.gz -r fixed.nii.gz -o warped.nii.gz \
      -t 1Warp.nii.gz -t 0GenericAffine.mat

  fastwarp apply -i moving.nii.gz -r fixed.nii.gz -o warped.nii.gz \
      -t 1Warp.nii.gz -t 0GenericAffine.mat

Fixed image onto the moving image:

  antsApplyTransforms -d 3 -i fixed.nii.gz -r moving.nii.gz -o fixed_in_moving.nii.gz \
      -t "[0GenericAffine.mat,1]" -t 1InverseWarp.nii.gz

  fastwarp apply -i fixed.nii.gz -r moving.nii.gz -o fixed_in_moving.nii.gz \
      -t "[0GenericAffine.mat,1]" -t 1InverseWarp.nii.gz

For a label image, add -n NearestNeighbor (antsApplyTransforms) or
-n nearestNeighbor (fastwarp apply). In Python with ANTsPy:

  import ants
  warped = ants.apply_transforms(fixed, moving, ["1Warp.nii.gz", "0GenericAffine.mat"])

FastWarp: https://github.com/WyssCenter/fastwarp (Apache-2.0)
Documentation: https://fastwarp.readthedocs.io
