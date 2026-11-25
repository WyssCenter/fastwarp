# Label fusion

Multi-atlas segmentation labels a new image with the labels of several atlases: each atlas (an
image with its label image) is registered to the target, the atlas labels are resampled onto
the target's grid, and the resampled labels are combined voxel by voxel. *Joint label fusion*
(Wang et al. 2013) weights the atlases at each voxel by how well a small patch of each atlas
matches the target there, and takes into account that atlases tend to make the same errors.
FastWarp runs it as ANTs' `antsJointFusion` does, on the CPU and in parallel, or with most of
the work on the GPU ([below](#running-on-the-gpu)).

{func}`~fastwarp.register_atlases` registers each atlas to the target with
{func}`~fastwarp.register_syn` on the GPU and writes its images (linear interpolation) and its
label image (nearest neighbour) on the target's grid, up to four atlases at the same time on one
GPU (as many as fit its memory; the files are the same as one at a time);
{func}`~fastwarp.joint_label_fusion` then fuses them:

```python
import ants
import fastwarp

target = ants.image_read("sub-01_T1w.nii.gz")
names = ["atlas1", "atlas2", "atlas3"]
atlases = [ants.image_read(f"{n}_T1w.nii.gz") for n in names]
segs = [ants.image_read(f"{n}_labels.nii.gz") for n in names]

reg = fastwarp.register_atlases(target, atlases, segs, "out")
warped = [ants.image_read(str(files[0])) for files in reg["atlases"]]
labels = [ants.image_read(str(p)) for p in reg["labels"]]

out = fastwarp.joint_label_fusion(target, warped, labels, mask=ants.get_mask(target))
segmentation = out["labels"]
```

`register_atlases` writes, for atlas `NN` (from 01), `atlasNN_Composite.h5` (with the other
files of `register_syn`), `atlasNN_Warped.nii.gz` and `atlasNN_WarpedLabels.nii.gz` (`int`);
label values above 2^24 are refused there, as the resampling of `apply_transforms` returns
`float32` voxels (cast to `int` afterwards), which hold whole numbers exactly only up to 2^24.

The result holds the fused labels (`"labels"`), the fused intensities (`"intensity"`, one
image per atlas modality) and the labels found (`"label_set"`); `return_posteriors=True` adds
the posterior of each label and `return_weights=True` the voting weight of each atlas. Without
label images only the intensities are fused.

## How it works

For each voxel of the target (inside the mask), and each atlas:

1. **Search.** Within the search radius (default 3 voxels), the atlas patch (default radius 2,
   a 5×5×5 cube) that best matches the target patch: by Pearson correlation with the z-scored
   target patch (`metric="PC"`, the default) or by mean squared difference (`"MSQ"`).
   Candidates are visited from the nearest outwards; the first best one wins.
2. **Weights.** From the z-scored patch differences of all atlases, the joint error matrix
   `M(i, j) = (sum d_i d_j / (P - 1))^beta` (with `P` the patch size), plus `alpha` on its
   diagonal; the weights solve `M w = 1`, with negative weights set to 0 (or by non-negative
   least squares with `nonnegative=True`), and are divided by their sum.
3. **Votes.** Every voxel of the target patch receives each atlas's weight for the atlas label
   at the matching position, so patches overlap and each voxel collects many votes. The fused
   label is the label with the largest vote; the fused intensity is the average of the weighted
   atlas patches.

These are the steps of `antsJointFusion`, which differ from the paper in a few places: the
patches are z-scored, the search uses correlation by default, the votes of overlapping patches
add up, and intensities are fused as well (Wang and Yushkevich 2013 describe the open-source
implementation).

## Parameters

| Argument | `antsJointFusion` | Default | Meaning |
| --- | --- | --- | --- |
| `mask` | `-x` | none | fuse only where it is not 0 |
| `alpha` | `-a` | 0.1 | added to the diagonal of the joint error matrix; larger values pull the weights towards equal |
| `beta` | `-b` | 2.0 | exponent of the patch differences; larger values favour the best atlases more |
| `patch_radius` | `-p` | 2 | patch radius in voxels, one value or three; at most 5 |
| `search_radius` | `-s` | 3 | search radius in voxels, one value, three, or an image of radii per voxel; at most 10 |
| `metric` | `-m` | `"PC"` | `"PC"` or `"MSQ"` |
| `nonnegative` | `-c 1` | `False` | non-negative least squares for the weights |
| `exclusions` | `-e` | none | `{label: image}`: the label cannot win where the image is not 0 |

Label images must hold whole numbers from 0 to 4294967295 (other values are refused); large ids
such as those of the Allen atlas stay exact. `alpha` and `beta` are rounded to `float32`, as `antsJointFusion` reads them. The time grows
with the number of atlases, the number of search positions (343 for radius 3) and the patch
size; a mask that covers only the structures of interest saves most of it.

## Running on the GPU

The search for the best-matching atlas patch is almost all of the work: for every voxel, atlas
and search position, one patch comparison. Two modes move work to the GPU (`adapter` picks one by
name):

- `mode="hybrid"`: the search on the GPU, the weights, votes and intensities on the CPU in
  `float64`.
- `mode="gpu"`: the search, the weights (a `float32` Cholesky solve per voxel) and the votes and
  intensities on the GPU. Where the `float32` solve is not reliable (a pivot that is not positive,
  a 1-norm condition number of the matrix above 1000, weights that do not sum to a positive
  number) the voxel's weights are computed on the CPU in `float64`, as are all weights with
  `nonnegative=True`; a voxel with votes for more than 16 labels gets its votes on the CPU. More
  than 32 atlases run as `"hybrid"`.

```python
out = fastwarp.joint_label_fusion(target, atlases, labels, mask=mask, mode="gpu")
```

The GPU computes the same `float32` patch scores in the same order, but a GPU may round a
division less exactly and may fuse a multiplication with an addition, so two search positions
whose scores lie within `float32` rounding of each other can be ranked the other way. On an
Intel Arc 140V, for a brain-sized volume at 1 mm (182 x 218 x 182 voxels, 2.2 million in the
mask) with `patch_radius=2` and `search_radius=3` (two runs for `"hybrid"`; times vary by
about 20 % between runs on this laptop):

| Atlases | `"cpu"` (8 threads) | `"hybrid"` | `"gpu"` | Labels that differ from `"cpu"` |
| --- | --- | --- | --- | --- |
| 5 | 245 s | 51–68 s | 40 s | 62 voxels (2.8e-5 of the mask), the same in both GPU modes |
| 20 | 1065 s | 305–330 s | 243 s | 58 voxels (2.6e-5), the same in both GPU modes |

In the search, 0.025 % of the (voxel, atlas) pairs picked another position than on the CPU (5
atlases), each within 1.2e-4 (relative) of the best score; small test volumes with smooth
textures, where near-ties are more common, show up to 1 %. Between `"gpu"` and `"hybrid"`, which
share the search, the labels were equal and the fused intensities agreed to 1.5e-6 (relative).
In the tests (default `alpha`, no voxel on the CPU solve), the GPU weights matched the CPU's to
1.5e-5 (4 atlases) and 7.6e-5 (12 atlases) relative error at the 99th percentile; with
`alpha=1e-6` the error reached 1.8e-4, as the `float32` matrix then decides the weights. With many atlases
the GPU weights take about as long as the search, so `"gpu"` gains little over `"hybrid"` there. Atlas voxels that are not finite
count as on the CPU (a Pearson score of 0, a mean squares score that never wins). `mode="cpu"`
(the default) gives the results of `antsJointFusion` and needs no GPU; the GPU modes are for when
time matters more than those last near-ties. The atlases are held on the GPU in z slabs of at most
256 MiB with a halo of the search and patch radius, and the votes are gathered in z slabs, so
memory stays bounded. The command line `fastwarp ants-joint-fusion` takes the `antsJointFusion`
arguments only and always runs on the CPU.

## Matching ANTs

The results equal those of `antsJointFusion` run on one thread (`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`)
except where search scores tie (see below): FastWarp follows its arithmetic, `float` for the
search and `double` for the weights, and adds the votes in the same order whatever the number of
threads. Some behaviours of ANTs are kept on purpose:

- Near the image border, where a patch reaches outside the image, all atlases get equal weights
  (the patch differences there are not finite).
- A patch of constant intensity cannot be z-scored. A constant target patch gives all atlases
  equal weights; a constant atlas patch (for example in a zero background) has its row of the
  error matrix set to 0 and so gets the largest weight.
- The posteriors and voting weights are divided by the vote sum only where that sum is at least
  0.1.

Where the results can differ from ANTs: when two search positions at the same distance score
exactly the same, FastWarp takes the one first in neighbourhood order. ANTs takes the first in the
order its `std::sort` (not stable) leaves equal distances in, which depends on the C++ library,
so `antsJointFusion` itself can differ between Linux, macOS and Windows builds. Exact ties are
rare on textured images but common under `PC` where the atlas patches are flat (for example a
zero background): every flat patch scores 0. Near such regions the chosen positions, and so the
votes and labels, can differ from those of a particular ANTs build.

The command line [`fastwarp ants-joint-fusion`](../cli.md#ants-joint-fusion) takes the
`antsJointFusion` arguments, and {func}`fastwarp.ants.joint_label_fusion` the arguments of
ANTsPy's `ants.joint_label_fusion` (whose defaults differ: `beta=4`, `rho=0.01` for alpha, and
mean squares unless `usecor=True`).

## References

- Wang H, Suh JW, Das SR, Pluta JB, Craige C, Yushkevich PA (2013). Multi-atlas segmentation
  with joint label fusion. *IEEE Transactions on Pattern Analysis and Machine Intelligence*
  35(3):611–623. [doi:10.1109/TPAMI.2012.143](https://doi.org/10.1109/TPAMI.2012.143)
- Wang H, Yushkevich PA (2013). Multi-atlas segmentation with joint label fusion and corrective
  learning—an open source implementation. *Frontiers in Neuroinformatics* 7:27.
  [doi:10.3389/fninf.2013.00027](https://doi.org/10.3389/fninf.2013.00027)
