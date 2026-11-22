# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""fastwarp.joint_label_fusion and fastwarp.ants.joint_label_fusion.

The native function needs neither a GPU nor ANTsPy; the ANTsPy-compatible one is compared with
ANTsPy's own ants.joint_label_fusion (antsJointFusion, one ITK thread), which it matches exactly.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import fastwarp as g
import fastwarp.ants as fa
import numpy as np
import pytest

SHAPE = (16, 15, 14)
SPACING = (1.0, 1.1, 1.3)
ORIGIN = (3.0, -2.0, 1.0)
DIRECTION = np.eye(3)

_NIFTI_TYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    768: np.uint32,
}


def _read_nifti(path: Path) -> tuple[np.ndarray, int, np.ndarray]:
    """(voxels with x fastest, datatype code, the sform rows) of a NIfTI-1 file."""
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    dim = struct.unpack_from("<8h", raw, 40)
    datatype = struct.unpack_from("<h", raw, 70)[0]
    offset = int(struct.unpack_from("<f", raw, 108)[0])
    srow = np.array(struct.unpack_from("<12f", raw, 280)).reshape(3, 4)
    shape = tuple(dim[1 : 1 + dim[0]])
    count = int(np.prod(shape))
    data = np.frombuffer(raw, dtype=_NIFTI_TYPES[datatype], count=count, offset=offset)
    return data.reshape(shape, order="F"), datatype, srow


def _texture(shift: float, noise: float, seed: int) -> np.ndarray:
    idx = np.indices(SHAPE).astype(np.float64)
    v = 100 + 30 * np.sin(0.45 * idx[0] + shift) * np.cos(0.35 * idx[1])
    v += 20 * np.sin(0.3 * idx[2] + 0.5 * shift)
    v += noise * np.random.default_rng(seed).standard_normal(SHAPE)
    return v.astype(np.float32)


def _labels(shift: int) -> np.ndarray:
    idx = np.indices(SHAPE)
    return (1 + (idx[0] + shift >= 8) + 2 * (idx[2] >= 7)).astype(np.uint32)


def _tuple(arr: np.ndarray) -> tuple:
    return (arr, SPACING, ORIGIN, DIRECTION)


@pytest.fixture(scope="module")
def scene():
    target = _tuple(_texture(0.0, 2.0, 0))
    atlases = [_tuple(_texture(s, 3.0, i + 1)) for i, s in enumerate([0.15, -0.2, 0.9])]
    labels = [_tuple(_labels(s)) for s in (0, 1, -1)]
    return target, atlases, labels


def test_label_fusion_returns_labels_intensities_posteriors_and_weights(scene):
    target, atlases, labels = scene
    out = g.joint_label_fusion(
        target,
        atlases,
        labels,
        patch_radius=1,
        search_radius=1,
        return_posteriors=True,
        return_weights=True,
    )
    assert out["label_set"] == [1, 2, 3, 4]
    assert out["labels"].dtype == np.uint32 and out["labels"].shape == SHAPE
    assert set(np.unique(out["labels"])) <= {1, 2, 3, 4}
    assert len(out["intensity"]) == 1 and out["intensity"][0].shape == SHAPE
    assert sorted(out["posteriors"]) == [1, 2, 3, 4]
    total = sum(out["weights"])
    np.testing.assert_allclose(total, 1.0, atol=1e-5)
    posterior = np.stack([out["posteriors"][v] for v in (1, 2, 3, 4)])
    np.testing.assert_array_equal(np.argmax(posterior, axis=0) + 1, out["labels"])


def test_atlas_order_and_a_perfect_atlas(scene):
    target, atlases, labels = scene
    out = g.joint_label_fusion(
        target, [target, *atlases[1:]], labels, patch_radius=1, search_radius=1, return_weights=True
    )
    inner = (slice(3, -3),) * 3
    assert np.all(out["weights"][0][inner] > out["weights"][1][inner])
    plain = np.mean([a[0] for a in [target, *atlases[1:]]], axis=0)
    fused_error = np.abs(out["intensity"][0][inner] - target[0][inner]).mean()
    assert fused_error < np.abs(plain[inner] - target[0][inner]).mean()


def test_mask_radius_image_and_exclusions(scene):
    target, atlases, labels = scene
    mask = np.zeros(SHAPE, np.uint8)
    mask[2:-2, 2:-2, 2:-2] = 1
    radius = np.full(SHAPE, 1, np.uint8)
    out = g.joint_label_fusion(
        target,
        atlases,
        labels,
        mask=_tuple(mask),
        patch_radius=[1, 1, 1],
        search_radius=_tuple(radius),
        metric="msq",
        nonnegative=True,
        exclusions={4: _tuple(np.ones(SHAPE, np.uint8))},
    )
    assert np.all(out["labels"][mask == 0] == 0)
    assert np.all(out["intensity"][0][mask == 0] == 0)
    assert 4 not in np.unique(out["labels"])
    assert out["label_set"] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"atlases": "one"}, "at least 2 atlases"),
        ({"labels": "short"}, "label images for"),
        ({"metric": "CC"}, "metric must be"),
        ({"patch_radius": [1, 2]}, "patch_radius must be"),
        ({"alpha": float("nan")}, "alpha must be"),
    ],
)
def test_invalid_arguments(scene, kwargs, error):
    target, atlases, labels = scene
    args = {"atlases": atlases, "labels": labels}
    if kwargs.get("atlases") == "one":
        args["atlases"] = atlases[:1]
        args["labels"] = None
    elif kwargs.get("labels") == "short":
        args["labels"] = labels[:2]
    else:
        args.update(kwargs)
    with pytest.raises(ValueError, match=error):
        g.joint_label_fusion(target, args.pop("atlases"), args.pop("labels"), **args)


def test_intensity_fusion_only(scene):
    target, atlases, _ = scene
    out = g.joint_label_fusion(target, atlases, patch_radius=1, search_radius=1)
    assert out["labels"] is None and out["label_set"] == []
    assert np.all(np.isfinite(out["intensity"][0]))


def test_large_label_ids_stay_exact(scene):
    target, atlases, labels = scene
    big = [_tuple(lab[0].astype(np.uint32) + 600_000_000) for lab in labels]
    out = g.joint_label_fusion(target, atlases, big, patch_radius=1, search_radius=1)
    assert out["label_set"] == [600_000_001, 600_000_002, 600_000_003, 600_000_004]
    assert out["labels"].dtype == np.uint32
    assert set(np.unique(out["labels"])) <= set(out["label_set"])
    small = g.joint_label_fusion(target, atlases, labels, patch_radius=1, search_radius=1)
    np.testing.assert_array_equal(out["labels"], small["labels"] + 600_000_000)


@pytest.mark.parametrize("bad", [-1.0, 2.5, 2.0**32])
def test_labels_outside_uint32_or_not_whole_are_refused(scene, bad):
    target, atlases, labels = scene
    wrong = labels[0][0].astype(np.float64)
    wrong[3, 3, 3] = bad
    with pytest.raises(ValueError, match="is not supported; its values must be whole numbers"):
        g.joint_label_fusion(target, atlases, [_tuple(wrong), *labels[1:]])


def test_the_hybrid_search_matches_the_cpu_search(scene):
    target, atlases, labels = scene
    with pytest.raises(ValueError, match="mode must be"):
        g.joint_label_fusion(target, atlases, labels, mode="fpga")
    with pytest.raises(ValueError, match="adapter applies"):
        g.joint_label_fusion(target, atlases, labels, adapter="Arc")
    if not g.gpu_available():
        pytest.skip("no usable GPU adapter")
    cpu = g.joint_label_fusion(target, atlases, labels, patch_radius=1, search_radius=1)
    for mode in ("hybrid", "gpu"):
        gpu = g.joint_label_fusion(
            target, atlases, labels, patch_radius=1, search_radius=1, mode=mode
        )
        assert gpu["label_set"] == cpu["label_set"]
        assert np.count_nonzero(gpu["labels"] != cpu["labels"]) <= cpu["labels"].size // 100
        np.testing.assert_allclose(gpu["intensity"][0], cpu["intensity"][0], rtol=1e-2, atol=1e-2)


@pytest.mark.filterwarnings("ignore:fastwarp. linear stage")
def test_register_atlases_writes_files_for_label_fusion(tmp_path, scene):
    target, atlases, labels = scene
    with pytest.raises(ValueError, match="label images for"):
        g.register_atlases(target, atlases, labels[:1], tmp_path)
    big = _tuple(labels[0][0].astype(np.float64) + 2**25)
    with pytest.raises(ValueError, match="whole numbers"):
        g.register_atlases(target, atlases[:1], [big], tmp_path)
    if not g.gpu_available():
        pytest.skip("no usable GPU adapter")
    out = g.register_atlases(target, atlases[:2], labels[:2], tmp_path / "reg", iso_mm=2.0)
    assert [p.name for p in out["transforms"]] == ["atlas01_Composite.h5", "atlas02_Composite.h5"]
    assert [p.name for p in out["labels"]] == [
        "atlas01_WarpedLabels.nii.gz",
        "atlas02_WarpedLabels.nii.gz",
    ]
    for path in [*out["labels"], *out["transforms"], *(f for fs in out["atlases"] for f in fs)]:
        assert path.is_file() and path.stat().st_size > 0, path
    grid = (SHAPE, SPACING, ORIGIN, DIRECTION)
    ras = np.array(
        [
            [-SPACING[0], 0, 0, -ORIGIN[0]],
            [0, -SPACING[1], 0, -ORIGIN[1]],
            [0, 0, SPACING[2], ORIGIN[2]],
        ]
    )
    warped = []
    for i, composite in enumerate(out["transforms"]):
        expected = g.apply_transforms(atlases[i], grid, [composite])
        data, datatype, srow = _read_nifti(out["atlases"][i][0])
        assert datatype == 16 and data.shape == SHAPE
        np.testing.assert_allclose(srow, ras, atol=1e-5)
        np.testing.assert_array_equal(data, expected)
        lab = g.apply_transforms(labels[i], grid, [composite], interpolation="nearestNeighbor")
        data, datatype, srow = _read_nifti(out["labels"][i])
        assert datatype == 8 and data.shape == SHAPE
        np.testing.assert_allclose(srow, ras, atol=1e-5)
        np.testing.assert_array_equal(data, lab.astype(np.int32))
        warped.append(expected)
    fused = g.joint_label_fusion(target, [_tuple(w) for w in warped], patch_radius=1)
    assert fused["intensity"][0].shape == SHAPE


def test_images_of_another_size_are_refused(scene):
    target, atlases, _ = scene
    small = _tuple(np.zeros((4, 4, 4), np.float32))
    with pytest.raises(ValueError, match="voxels"):
        g.joint_label_fusion(target, [atlases[0], small])


@pytest.fixture(scope="module")
def ants():
    return pytest.importorskip("ants")


def _ants_scene(ants, scene, relabel=None, zeros=False):
    target, atlases, labels = scene
    if relabel is not None:
        labels = [(np.vectorize(relabel.get)(lab[0]).astype(np.uint32), *lab[1:]) for lab in labels]
    if zeros:
        first = labels[0][0].copy()
        first[:, 4:7, :] = 0
        labels = [(first, *labels[0][1:]), *labels[1:]]

    def image(t, pixeltype="float"):
        img = ants.from_numpy(t[0].astype(np.float32), spacing=SPACING, origin=ORIGIN)
        return img.clone(pixeltype)

    mask = np.zeros(SHAPE, np.float32)
    mask[1:-1, 2:-1, 1:-2] = 1
    return (
        image(target),
        ants.from_numpy(mask, spacing=SPACING, origin=ORIGIN),
        [image(a) for a in atlases],
        [image(lab, "unsigned int") for lab in labels],
    )


@pytest.mark.parametrize(
    ("kw", "variant"),
    [
        ({"rad": 1, "r_search": 1}, {}),
        ({"rad": [1, 1, 1], "r_search": 2, "usecor": True, "beta": 2, "rho": 0.1}, {}),
        ({"rad": 1, "r_search": 1, "nonnegative": True, "no_zeroes": True}, {"zeros": True}),
        ({"rad": 1, "r_search": 1}, {"zeros": True}),
        ({"rad": 1, "r_search": 1}, {"relabel": {1: 3, 2: 10, 3: 100, 4: 7}}),
    ],
)
def test_ants_joint_label_fusion_equals_antspy(ants, scene, kw, variant):
    target, mask, atlases, labels = _ants_scene(ants, scene, **variant)
    before = mask.numpy().copy()
    ours = fa.joint_label_fusion(target, mask, atlases, label_list=labels, **kw)
    np.testing.assert_array_equal(mask.numpy(), before)
    theirs = ants.joint_label_fusion(target, mask.clone(), atlases, label_list=labels, **kw)
    assert ours["segmentation_numbers"] == theirs["segmentation_numbers"]
    if "relabel" in variant:
        assert ours["segmentation_numbers"] == [3, 7, 10, 100]
    if kw.get("no_zeroes"):
        assert np.all(ours["segmentation"].numpy()[:, 4:7, :] == 0)
    np.testing.assert_array_equal(ours["segmentation"].numpy(), theirs["segmentation"].numpy())
    np.testing.assert_array_equal(ours["intensity"].numpy(), theirs["intensity"].numpy())
    for a, b in zip(ours["probabilityimages"], theirs["probabilityimages"], strict=True):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_ants_intensity_fusion_and_refusals(ants, scene, tmp_path):
    target, mask, atlases, labels = _ants_scene(ants, scene)
    ours = fa.joint_label_fusion(target, mask, atlases, rad=1, r_search=1)
    theirs = ants.joint_label_fusion(target, mask, atlases, rad=1, r_search=1)
    np.testing.assert_array_equal(ours.numpy(), theirs.numpy())
    with pytest.raises(fa.UnsupportedArgument, match="max_lab_plus_one"):
        fa.joint_label_fusion(target, mask, atlases, label_list=labels, max_lab_plus_one=True)
    with pytest.raises(ValueError, match="len\\(label_list\\)"):
        fa.joint_label_fusion(target, mask, atlases, label_list=labels[:2])
    out = fa.joint_label_fusion(
        target,
        mask,
        atlases,
        label_list=labels,
        rad=1,
        r_search=1,
        output_prefix=str(tmp_path / "jlf_"),
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"jlf_prob{v:02d}.nii.gz" for v in out["segmentation_numbers"]
    ]
