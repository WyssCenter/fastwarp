# Resampling

Two functions apply ANTs transform files, like `antsApplyTransforms`. They run in parallel on
all CPU cores and need no GPU, so they also work on a machine where the registration itself ran
with ANTs.

## Transform lists

Both functions take `transforms`: one file path, or a list of paths, with the same meaning as
the `transformlist` of {func}`ants.apply_transforms` and the `-t` options of
`antsApplyTransforms`. A point of the reference grid goes through the first file of the list
first, then through the second, and so on. To warp a subject's scan into template space with
separate ANTs files, list them as ANTs writes them: `["sub_1Warp.nii.gz",
"sub_0GenericAffine.mat"]`.

These files can be read:

| Extension | Content |
| --- | --- |
| `.h5`, `.hdf5`, `.hdf` | An ITK HDF5 transform file, for example `Composite.h5`. |
| `.mat` | An ITK MATLAB affine file, for example `0GenericAffine.mat`. |
| `.txt`, `.tfm` | An ITK text transform file with a linear transform. |
| `.nii`, `.nii.gz` | A displacement field, for example `1Warp.nii.gz` or the output of {func}`~fastwarp.compose_field`. |

A file cannot be inverted inside the list, as `[file,1]` does in `antsApplyTransforms`; use the
inverse file instead, for example `InverseComposite.h5`. A missing file raises
`FileNotFoundError` before any work.

## Resample an image

```{eval-rst}
.. py:function:: fastwarp.apply_transforms(moving, reference, transforms, *, interpolation="linear", default_value=0.0)
   :no-index:
```

- `moving` is the [image](images.md) to resample.
- `reference` gives the output grid: an `ANTsImage` or a tuple `(shape, spacing, origin,
  direction)`.
- `interpolation` is one of {data}`~fastwarp.INTERPOLATORS`: `"linear"`, `"nearestNeighbor"`
  (for label images) or `"lanczosWindowedSinc"`.
- Output voxels that map outside `moving` get `default_value`.

The result is an `ANTsImage` like `reference` when `reference` is an `ANTsImage`, and otherwise a
`float32` NumPy array of the reference's shape.

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
moving = ants.image_read(ants.get_ants_data("ch2"))
grid = (fixed.shape, fixed.spacing, fixed.origin, fixed.direction)

image = fastwarp.apply_transforms(
    moving, fixed, ["out/ch2_to_mni_Composite.h5"], interpolation="lanczosWindowedSinc"
)
array = fastwarp.apply_transforms(moving, grid, ["out/ch2_to_mni_Composite.h5"])
print(type(image).__name__, type(array).__name__, array.dtype, array.shape)
```

```text
ANTsImage ndarray float32 (182, 218, 182)
```

The transform file comes from the [quickstart](../quickstart.md) example.

## Write one displacement field

```{eval-rst}
.. py:function:: fastwarp.compose_field(transforms, reference, out_path, *, compression_level=None)
   :no-index:
```

{func}`~fastwarp.compose_field` evaluates the whole transform list at every voxel of the
reference grid and writes the result as one displacement field. It is the file that
`antsApplyTransforms -o [field.nii.gz,1]` writes: a NIfTI vector image of `float32` displacements
in LPS millimetres, which ANTs and FastWarp read back as a single transform.

- `out_path` ending in `.nii.gz` is written with gzip compression, using several threads;
  `.nii` is written uncompressed. The folder is created when it is missing.
- `compression_level` sets the gzip level from 0 to 9; `None` uses level 1. Level 0 writes a
  valid gzip file without compressing the data, which suits displacement fields: they barely
  compress, and the file is written and read faster.
- It returns the output path as a {class}`pathlib.Path`.

```python
import ants
import fastwarp

fixed = ants.image_read(ants.get_ants_data("mni"))
path = fastwarp.compose_field(
    ["out/ch2_to_mni_Composite.h5"], fixed, "out/ch2_to_mni_field.nii.gz", compression_level=0
)
field = ants.image_read(str(path))
print(field.shape, field.components)
```

```text
(182, 218, 182) 3
```

## Agreement with ANTs

The tests compare both functions with ANTsPy on the same transforms (`tests/test_post.py`). For
every interpolator, the resampled images may differ by at most 0.001 in absolute value on test
images with values up to about 110, and the composed displacement fields by at most 0.0001 mm.
