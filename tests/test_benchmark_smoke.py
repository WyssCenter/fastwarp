# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""benchmarks.accuracy end to end on a synthetic two-subject dataset: run (FastWarp and ANTs),
evaluate, report. Needs a GPU and ANTsPy; skips otherwise. No download, no real data."""

from __future__ import annotations

import copy
import csv
import json

import fastwarp as g
import numpy as np
import pytest
from test_register import SPACING, needs_gpu, phantom

from benchmarks.accuracy import datasets, evaluate, linear, multimodal, post, recipes, report, run

ants = pytest.importorskip("ants")

SHIFT = (4.0, -3.0, 2.0)
LEVELS = {"shrink_factors": [2, 1], "smoothing_sigmas": [1.0, 0.0]}


def _tiny_recipe() -> dict:
    r = copy.deepcopy(recipes.RECIPES["cohort"])
    for stage in ("rigid", "affine"):
        r[stage].update(iterations=[40, 20], **LEVELS)
    r["syn"].update(iterations=[10, 5], **LEVELS)
    return r


def _subject(root, sid: str, shift) -> datasets.Subject:
    vol, sp, origin, _ = phantom(shift)
    img = ants.from_numpy(vol.astype("float32"), spacing=sp, origin=origin)
    pts = np.indices(vol.shape, dtype=np.float64) * SPACING
    c = np.array([56.0, 40.0, 40.0]) + shift
    nucleus = sum(((pts[a] - c[a]) / (8, 6, 6)[a]) ** 2 for a in range(3)) <= 1.0
    brain = vol > 0.5
    labels = np.where(nucleus, 10, np.where(brain, 1002, 0)).astype("float32")
    folder = root / sid
    folder.mkdir(parents=True)
    ants.image_write(img, str(folder / "t1w.nii.gz"))
    ants.image_write(img.new_image_like(labels), str(folder / "labels.nii.gz"))
    ants.image_write(img.new_image_like(nucleus.astype("float32")), str(folder / "nucleus.nii.gz"))
    ras = [(-c[0], -c[1], c[2]), (-(c[0] + 20), -(c[1] + 30), c[2] + 5)]
    rows = "".join(
        f"n{i},{x},{y},{z},0,0,0,1,1,1,1,{i},P{i},vol\n" for i, (x, y, z) in enumerate(ras, 1)
    )
    (folder / "points.fcsv").write_text(
        "# Markups fiducial file version = 4.6\n# CoordinateSystem = 0\n"
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n" + rows
    )
    return datasets.Subject(
        id=sid,
        image=f"{sid}/t1w.nii.gz",
        labels=f"{sid}/labels.nii.gz",
        landmarks=f"{sid}/points.fcsv",
        masks=((7, f"{sid}/nucleus.nii.gz"),),
    )


@needs_gpu
def test_run_evaluate_report_on_a_phantom(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "work"
    root = data / "smoke"
    subjects = (_subject(root, "s-1", (0.0, 0.0, 0.0)), _subject(root, "s-2", SHIFT))
    (root / ".fetched.json").write_text("{}")
    ds = datasets.Dataset(
        key="smoke",
        title="phantom",
        licence="",
        downloads=(),
        subjects=subjects,
        protocol="pairwise",
        groups={"subcortical": (10, 7), "cortical": (1002,)},
        label_names={10: "nucleus", 7: "nucleus mask"},
    )
    monkeypatch.setattr(datasets, "registry", lambda: {"smoke": ds})
    monkeypatch.setitem(recipes.RECIPES, "tiny", _tiny_recipe())
    monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

    kw = {"data": data, "work": work, "keys": ["smoke"], "recipe_name": "tiny", "pairs": 1}
    rows = run.run(**kw, quick=False, iso_mm=SPACING, threads=1, log=lambda s: None)
    assert [(r["engine"], r["status"], r["ran"]) for r in rows] == [
        ("fastwarp", "ok", True),
        ("ants", "ok", True),
    ]
    again = run.run(**kw, quick=False, iso_mm=SPACING, threads=1, log=lambda s: None)
    assert not any(r["ran"] for r in again)

    pair = work / "smoke" / "tiny" / "s-1__s-2"
    ants_run = json.loads((pair / "ants" / "run.json").read_text())
    assert ants_run["argv"][ants_run["argv"].index("--convergence") + 1] == "[40x20,1e-06,10]"
    fw_run = json.loads((pair / "fastwarp" / "run.json").read_text())
    assert fw_run["recipe_matches"] and fw_run["environment"]["fastwarp"] == g.__version__
    assert fw_run["timings"]["total_s"] > 0 and fw_run["timings"]["syn_s"] > 0
    assert ants_run["timings"]["linear_s"] > 0 and ants_run["timings"]["syn_s"] > 0
    assert "note" not in ants_run["timings"]

    lin = linear.run_linear(**kw, quick=False, iso_mm=SPACING, threads=1, log=lambda s: None)
    assert [(r["engine"], r["status"]) for r in lin] == [("fastwarp", "ok"), ("ants", "ok")]
    comparison = json.loads(
        (work / "smoke" / "linear" / "s-1__s-2" / "comparison.json").read_text()
    )
    for fit in ("rigid", "affine"):
        assert comparison["fits"][fit]["fastwarp"]["landmark_mean_mm"] < 3.0
        assert comparison["fits"][fit]["agreement"]["mean_mm"] < 1.0
    posted = post.run_post(**kw, quick=False, iso_mm=SPACING, threads=1, log=lambda s: None)
    assert [r["status"] for r in posted] == ["ok"]
    post_record = json.loads((pair / "post.json").read_text())
    assert post_record["agreement"]["forward_field"]["field_diff_max_mm"] < 0.01
    assert not (pair / "post_tmp").exists()

    out = evaluate.evaluate(**kw, quick=False, log=lambda s: None)
    with open(out / "pairs.csv", newline="") as f:
        pairs = list(csv.DictReader(f))
    assert len(pairs) == 2
    for r in pairs:
        assert float(r["dice_subcortical"]) > 0.7, r
        assert float(r["landmark_mean_mm"]) < 2.0, r
        assert float(r["jac_nonpositive_frac"]) == 0.0, r
    with open(out / "labels.csv", newline="") as f:
        assert {r["label"] for r in csv.DictReader(f)} == {"7", "10", "1002"}
    with open(out / "agreement.csv", newline="") as f:
        agreement = list(csv.DictReader(f))
    assert len(agreement) == 1 and float(agreement[0]["field_diff_rms_mm"]) < 2.0

    md = report.report(work=work, recipe_name="tiny", plot=False)
    text = md.read_text()
    assert "FastWarp against ANTs" in text and "dice_subcortical" in text
    assert "## Registration time" in text and "## Linear fits" in text
    assert "## Post-processing" in text
    assert (out / "paired.csv").is_file() and (out / "structures.csv").is_file()


@needs_gpu
def test_an_antspy_call_and_a_second_ants_run_on_a_phantom(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "work"
    root = data / "smoke"
    subjects = (_subject(root, "s-1", (0.0, 0.0, 0.0)), _subject(root, "s-2", SHIFT))
    (root / ".fetched.json").write_text("{}")
    ds = datasets.Dataset(
        key="smoke",
        title="phantom",
        licence="",
        downloads=(),
        subjects=subjects,
        protocol="pairwise",
        groups={"subcortical": (10, 7), "cortical": (1002,)},
    )
    monkeypatch.setattr(datasets, "registry", lambda: {"smoke": ds})
    call = {"antspy": {"type_of_transform": "SyN", "reg_iterations": [6, 3, 0]}, "seed": 5}
    monkeypatch.setitem(recipes.RECIPES, "tiny-call", call)
    monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    kw = {"data": data, "work": work, "keys": ["smoke"], "recipe_name": "tiny-call", "pairs": 1}
    engines = ("fastwarp", "ants", "ants_repeat")
    rows = run.run(**kw, engines=engines, iso_mm=SPACING, threads=1, log=lambda s: None)
    assert [(r["engine"], r["status"]) for r in rows] == [(e, "ok") for e in engines]
    pair = work / "smoke" / "tiny-call" / "s-1__s-2"
    first = json.loads((pair / "ants" / "run.json").read_text())
    second = json.loads((pair / "ants_repeat" / "run.json").read_text())
    fw = json.loads((pair / "fastwarp" / "run.json").read_text())
    assert first["arguments"]["seed"] == 5 and second["arguments"]["seed"] == 20260911
    assert fw["call"] == "fastwarp.ants.registration"
    assert fw["arguments"] == {"kwargs": call["antspy"], "seed": 5}
    assert first["arguments"]["kwargs"] == call["antspy"] == second["arguments"]["kwargs"]
    assert first["ants_version"] and first["ants_version"] == second["ants_version"]
    assert first["timings"]["syn_s"] > 0 and first["timings"]["call_s"] > 0
    with pytest.raises(ValueError, match="needs a recipe with stages"):
        linear.run_linear(**kw, iso_mm=SPACING, log=lambda s: None)

    out = evaluate.evaluate(**kw, quick=False, log=lambda s: None)
    with open(out / "pairs.csv", newline="") as f:
        assert sorted(r["engine"] for r in csv.DictReader(f)) == ["ants", "fastwarp"]
    with open(out / "agreement.csv", newline="") as f:
        (agreement,) = list(csv.DictReader(f))
    assert float(agreement["field_diff_rms_mm"]) < 2.0
    assert float(agreement["ants_vs_ants_field_diff_rms_mm"]) < 2.0
    assert float(agreement["ants_vs_ants_agreement_dice_subcortical"]) > 0.8
    text = report.report(work=work, recipe_name="tiny-call", plot=False).read_text()
    assert "ants_vs_ants_field_diff_rms_mm" in text and "## Linear fits" not in text

    second["ants_version"] = "another"
    (pair / "ants_repeat" / "run.json").write_text(json.dumps(second))
    logged = []
    out = evaluate.evaluate(**kw, quick=False, force=True, log=logged.append)
    cached = json.loads((pair / "evaluation.json").read_text())
    assert "error" not in cached and "different ANTs" in cached["ants_repeat_skipped"]
    assert any("second ANTs run not compared" in line for line in logged)
    with open(out / "pairs.csv", newline="") as f:
        assert sorted(r["engine"] for r in csv.DictReader(f)) == ["ants", "fastwarp"]
    with open(out / "agreement.csv", newline="") as f:
        (agreement,) = list(csv.DictReader(f))
    assert float(agreement["field_diff_rms_mm"]) < 2.0
    assert not [c for c in agreement if c.startswith("ants_vs_ants")]


def _ixi_phantoms(root, template_shift, shift, contrast_shift):
    vol, sp, origin, _ = phantom(template_shift)
    tpl = ants.from_numpy(vol.astype("float32"), spacing=sp, origin=origin)
    root.mkdir(parents=True)
    ants.image_write(tpl, str(root / "template.nii.gz"))
    ants.image_write(tpl.new_image_like((vol > 0.5).astype("float32")), str(root / "mask.nii.gz"))
    t1, sp, origin, _ = phantom(shift)
    t2, _, _, _ = phantom(tuple(a + b for a, b in zip(shift, contrast_shift, strict=True)))
    t2 = np.where(t2 > 0.05, 1.2 - t2, 0.0)
    (root / "s").mkdir()
    for name, v in (("t1", t1), ("t2", t2)):
        img = ants.from_numpy(v.astype("float32"), spacing=sp, origin=origin)
        ants.image_write(img, str(root / "s" / f"{name}.nii.gz"))
    return datasets.Subject(id="tpl", image="template.nii.gz", mask="mask.nii.gz")


@needs_gpu
def test_the_multimodal_check_on_a_phantom(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "work"
    tpl = _ixi_phantoms(data / "ixi", (0.0, 0.0, 0.0), SHIFT, (1.5, 0.0, -1.0))
    common = {"licence": "", "downloads": (), "protocol": "template", "template": tpl}
    t1 = datasets.Dataset(
        key="ixi_t1",
        title="t1",
        subjects=(datasets.Subject(id="s", image="s/t1.nii.gz"),),
        data_folder="ixi",
        **common,
    )
    t2 = datasets.Dataset(
        key="ixi_t2",
        title="t2",
        subjects=(datasets.Subject(id="s", image="s/t2.nii.gz", align_to="s/t1.nii.gz"),),
        data_folder="ixi",
        **common,
    )
    for ds in (t1, t2):
        ds.fetched_marker(data).write_text("{}")
    monkeypatch.setattr(datasets, "registry", lambda: {"ixi_t1": t1, "ixi_t2": t2})
    tiny = _tiny_recipe()
    monkeypatch.setitem(recipes.RECIPES, "cohort", tiny)
    tiny_mi = copy.deepcopy(tiny)
    tiny_mi["syn"]["channel_metrics"] = ["mi"]
    monkeypatch.setitem(recipes.RECIPES, "cohort-mi", tiny_mi)
    monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    for key, name in (("ixi_t1", "cohort"), ("ixi_t2", "cohort-mi")):
        kw = {"data": data, "work": work, "keys": [key], "recipe_name": name, "pairs": 1}
        rows = run.run(**kw, iso_mm=SPACING, threads=1, log=lambda s: None)
        assert [r["status"] for r in rows] == ["ok", "ok"], rows
    pair = work / "ixi_t2" / "cohort-mi" / "tpl__s"
    ants_run = json.loads((pair / "ants" / "run.json").read_text())
    assert any(a.startswith("MI[") for a in ants_run["argv"])
    prep = run.prep_folder(t2, "s", work, SPACING)
    align = json.loads((prep / "align.json").read_text())
    assert align["type_of_transform"] == "Rigid" and align["random_seed"] == run.ALIGN_SEED

    out = multimodal.run_multimodal(work=work, pairs=1, log=lambda s: None)
    with open(out / "multimodal.csv", newline="") as f:
        table = list(csv.DictReader(f))
    assert sorted(r["engine"] for r in table) == ["ants", "fastwarp"]
    cached = json.loads((pair / "multimodal.json").read_text())
    for r in table:
        assert float(r["vs_own_field_diff_rms_mm"]) < 2.0, r
        own = r[f"vs_{r['engine']}_field_diff_rms_mm"]
        assert r["vs_own_field_diff_rms_mm"] == own, r
        for ref in ("ants", "fastwarp"):
            got = float(r[f"vs_{ref}_field_diff_rms_mm"])
            assert got == cached["engines"][r["engine"]][ref]["field_diff_rms_mm"], (r, ref)
    by = {r["engine"]: r for r in table}
    assert by["ants"]["vs_fastwarp_field_diff_rms_mm"] != by["ants"]["vs_ants_field_diff_rms_mm"]
    text = (out / "summary.md").read_text()
    assert "vs_own_field_diff_rms_mm" in text and "descriptive only" in text
    with open(out / "multimodal_paired.csv", newline="") as f:
        assert {r["metric"] for r in csv.DictReader(f)} == {
            f"vs_own_{k}" for k in multimodal.FIELD_KEYS
        }


def test_a_cached_alignment_of_another_seed_is_made_again(tmp_path, monkeypatch):
    root = tmp_path / "data" / "m"
    root.mkdir(parents=True)
    img = ants.from_numpy(np.ones((8, 8, 8), "float32"), spacing=(1.0, 1.0, 1.0))
    for name in ("t1.nii.gz", "t2.nii.gz"):
        ants.image_write(img, str(root / name))
    ds = datasets.Dataset(
        key="m",
        title="m",
        licence="",
        downloads=(),
        subjects=(datasets.Subject(id="a", image="t2.nii.gz", align_to="t1.nii.gz"),),
        protocol="pairwise",
    )
    calls = []

    def fake_align(moving, reference, folder):
        calls.append(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "align.json").write_text(json.dumps({"random_seed": run.ALIGN_SEED}))
        return reference

    monkeypatch.setattr(run, "align_rigidly", fake_align)
    work = tmp_path / "work"
    out = run.prepare_moving(ds, "a", tmp_path / "data", work, 1.0)
    assert out.is_file() and len(calls) == 1
    run.prepare_moving(ds, "a", tmp_path / "data", work, 1.0)
    assert len(calls) == 1
    (out.parent / "align.json").write_text(json.dumps({"random_seed": 7}))
    run.prepare_moving(ds, "a", tmp_path / "data", work, 1.0)
    assert len(calls) == 2
    (out.parent / "align.json").unlink()
    run.prepare_moving(ds, "a", tmp_path / "data", work, 1.0)
    assert len(calls) == 3


def test_the_moving_image_goes_onto_the_registration_grid(tmp_path, monkeypatch):
    root = tmp_path / "data" / "m"
    root.mkdir(parents=True)
    vol = np.random.default_rng(0).random((20, 16, 12)).astype("float32")
    img = ants.from_numpy(vol, spacing=(1.5, 1.5, 1.5), origin=(3.0, -2.0, 1.0))
    ants.image_write(img, str(root / "a.nii.gz"))
    ds = datasets.Dataset(
        key="m",
        title="m",
        licence="",
        downloads=(),
        subjects=(datasets.Subject(id="a", image="a.nii.gz"),),
        protocol="pairwise",
    )
    work = tmp_path / "work"
    out = run.prepare_moving(ds, "a", tmp_path / "data", work, 2.0)
    moved = ants.image_read(str(out))
    assert out.parent == run.prep_folder(ds, "a", work, 2.0)
    np.testing.assert_allclose(moved.spacing, (2.0, 2.0, 2.0))
    np.testing.assert_allclose(moved.origin, img.origin)
    assert run.prepare_moving(ds, "a", tmp_path / "data", work, 1.5) == root / "a.nii.gz"
