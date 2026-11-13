# Output files

{func}`~fastwarp.register_syn` writes the two composite files that
`antsRegistration --write-composite-transform 1` writes, plus the collapsed affine as
`0GenericAffine.mat`. Tools that read ANTs transforms read them: `antsApplyTransforms`,
`antsApplyTransformsToPoints`, [ANTsPy](https://github.com/ANTsX/ANTsPy), and FastWarp's own
[resampling functions](resampling.md). A {class}`~fastwarp.LinearSession` fit writes only the
affine file.

## Which file to use

The prefix `out/sub_` stands for the `out_prefix` you passed. `fixed` is the image you
registered onto (for example a template), `moving` the image you registered (for example a
subject's scan).

| File | Maps points | Warps images | Use it to |
| --- | --- | --- | --- |
| `out/sub_Composite.h5` | from fixed space to moving space | from moving space onto the fixed grid | bring a subject's scan into template space |
| `out/sub_InverseComposite.h5` | from moving space to fixed space | from fixed space onto the moving grid | bring a template or an atlas into the subject's space |
| `out/sub_0GenericAffine.mat` | from fixed space to moving space, affine part only | from moving space onto the fixed grid, affine part only | apply only the linear part, in either direction |
| `out/sub__Warped.nii.gz` | | | look at the result (only with `write_warped=True`) |

Points and images go in opposite directions. This is the ANTs convention: to fill a voxel of
the output grid, a resampler looks up where that voxel's centre lies in the input image, so the
transform that warps an image from moving to fixed space maps points from fixed to moving space.
To move a point from the subject's space into the template's space, use the inverse composite.

## With antsApplyTransforms

A subject's scan in template space:

```bash
antsApplyTransforms -d 3 -i sub_T1w.nii.gz -r template.nii.gz \
  -t out/sub_Composite.h5 -o sub_T1w_in_template.nii.gz
```

Atlas labels in the subject's space, with nearest-neighbour interpolation so that the labels
stay whole numbers:

```bash
antsApplyTransforms -d 3 -i atlas_labels.nii.gz -r sub_T1w.nii.gz \
  -t out/sub_InverseComposite.h5 -n NearestNeighbor -o atlas_labels_in_sub.nii.gz
```

The affine part alone, forward and inverse (`[file,1]` inverts an affine file):

```bash
antsApplyTransforms -d 3 -i sub_T1w.nii.gz -r template.nii.gz \
  -t out/sub_0GenericAffine.mat -o sub_T1w_affine.nii.gz
antsApplyTransforms -d 3 -i template.nii.gz -r sub_T1w.nii.gz \
  -t "[out/sub_0GenericAffine.mat,1]" -o template_affine_in_sub.nii.gz
```

The forward transform as one displacement field on the template grid
({func}`~fastwarp.compose_field` writes the same file):

```bash
antsApplyTransforms -d 3 -r template.nii.gz -t out/sub_Composite.h5 \
  -o "[sub_field.nii.gz,1]"
```

Points measured in the subject's scan, for example electrode contacts, in template space. The
CSV file has the columns `x,y,z,t` in LPS millimetres:

```bash
antsApplyTransformsToPoints -d 3 -i points_in_sub.csv -o points_in_template.csv \
  -t out/sub_InverseComposite.h5
```

## With ANTsPy

```python
import ants
import pandas as pd
template = ants.image_read("template.nii.gz")
sub = ants.image_read("sub_T1w.nii.gz")

sub_in_template = ants.apply_transforms(template, sub, ["out/sub_Composite.h5"])
template_in_sub = ants.apply_transforms(sub, template, ["out/sub_InverseComposite.h5"])

points = pd.DataFrame({"x": [10.0], "y": [-20.0], "z": [5.0]})
points_in_template = ants.apply_transforms_to_points(3, points, ["out/sub_InverseComposite.h5"])
```

## With FastWarp

{func}`~fastwarp.apply_transforms` takes the same transform lists as `ants.apply_transforms`,
with the image to resample first and the reference grid second:

```python
import ants
import fastwarp

template = ants.image_read("template.nii.gz")
sub = ants.image_read("sub_T1w.nii.gz")

sub_in_template = fastwarp.apply_transforms(sub, template, ["out/sub_Composite.h5"])
template_in_sub = fastwarp.apply_transforms(template, sub, ["out/sub_InverseComposite.h5"])
```

FastWarp's functions take file paths only. They cannot invert a file the way `[file,1]` does;
use `InverseComposite.h5` for the inverse direction.

## What the files contain

`Composite.h5`
: An ITK HDF5 transform file with two transforms, applied to a point of the fixed space in this
  order: the SyN displacement field, then the affine. ANTs lists them as `[Affine, Warp]`. The
  displacement field lies on the registration grid (see
  [Normalization](normalization.md#the-registration-grid)). The data are stored in double
  precision, without compression.

`InverseComposite.h5`
: The inverse, `[InverseWarp, Affine⁻¹]` in ANTs' notation.

`0GenericAffine.mat`
: An ITK MATLAB transform file with one `AffineTransform_double_3_3`: the rigid and affine stages
  collapsed into one affine, as ANTs writes it with `--collapse-output-transforms 1`. Its 12
  parameters and centre are also in the result as `affine_parameters` and `affine_center`.

`_Warped.nii.gz`
: The first moving image warped with linear interpolation onto the registration grid, as
  `float32`. It lies on the registration grid, not on the fixed image's own grid; for an image on
  the fixed grid, resample with `Composite.h5`.

The files are tested against ANTs: `tests/test_register.py::test_ants_reads_the_transforms`
warps with both composites through ANTsPy, and the Rust tests in `crates/fastwarp-reg` compare
the engine's transform files with the ANTs 2.6.5 programs (see [Testing](../testing.md)).
