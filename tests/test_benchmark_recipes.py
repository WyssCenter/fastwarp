# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""benchmarks.accuracy: the recipes, their antsRegistration argv and the pair selection."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.accuracy import datasets, post, recipes, run

COHORT_ARGV = [
    "--dimensionality",
    "3",
    "--output",
    "[/w/p/]",
    "--winsorize-image-intensities",
    "[0.005,0.995]",
    "--use-histogram-matching",
    "0",
    "--initial-moving-transform",
    "[/f.nii.gz,/m.nii.gz,1]",
    "--transform",
    "Rigid[0.1]",
    "--metric",
    "MI[/f.nii.gz,/m.nii.gz,1,32,Regular,0.25]",
    "--convergence",
    "[1000x500x250x100,1e-06,10]",
    "--shrink-factors",
    "8x4x2x1",
    "--smoothing-sigmas",
    "3x2x1x0vox",
    "--transform",
    "Affine[0.1]",
    "--metric",
    "MI[/f.nii.gz,/m.nii.gz,1,32,Regular,0.25]",
    "--convergence",
    "[1000x500x250x100,1e-06,10]",
    "--shrink-factors",
    "8x4x2x1",
    "--smoothing-sigmas",
    "3x2x1x0vox",
    "--transform",
    "SyN[0.1,3,0]",
    "--metric",
    "CC[/f.nii.gz,/m.nii.gz,1,4]",
    "--convergence",
    "[100x70x50x20,1e-06,10]",
    "--shrink-factors",
    "8x4x2x1",
    "--smoothing-sigmas",
    "3x2x1x0vox",
    "--write-composite-transform",
    "1",
    "--random-seed",
    "20260910",
    "--verbose",
    "1",
]


def test_the_cohort_recipe_is_fastwarps_default():
    fastwarp = pytest.importorskip("fastwarp")
    assert recipes.recipe("cohort") == fastwarp.cohort_recipe(1)


def test_cohort_argv_is_the_documented_ants_command():
    argv = recipes.ants_argv(recipes.recipe("cohort"), "/f.nii.gz", "/m.nii.gz", "/w/p/")
    assert argv == COHORT_ARGV


def test_syn_sh_differs_only_in_the_syn_step_and_radius():
    a, b = recipes.recipe("cohort"), recipes.recipe("syn-sh")
    assert a["rigid"] == b["rigid"] and a["affine"] == b["affine"]
    assert {k for k in a["syn"] if a["syn"][k] != b["syn"][k]} == {"gradient_step", "cc_radius"}
    argv = recipes.ants_argv(b, "/f", "/m", "/o/")
    assert "SyN[0.2,3,0]" in argv and "CC[/f,/m,1,2]" in argv


def test_recipes_are_copies_and_bad_inputs_are_refused():
    r = recipes.recipe("cohort")
    r["syn"]["iterations"][0] = 1
    assert recipes.recipe("cohort")["syn"]["iterations"][0] == 100
    with pytest.raises(ValueError, match="unknown recipe"):
        recipes.recipe("nope")
    with pytest.raises(ValueError, match="cannot take a path"):
        recipes.ants_argv(recipes.recipe("cohort"), "/a,b.nii", "/m", "/o/")
    two = recipes.recipe("cohort")
    two["syn"]["channel_weights"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="one channel"):
        recipes.ants_argv(two, "/f", "/m", "/o/")
    gc = recipes.recipe("cohort")
    gc["syn"]["channel_metrics"] = ["gc"]
    with pytest.raises(ValueError, match="SyN on CC or MI"):
        recipes.ants_argv(gc, "/f", "/m", "/o/")
    with pytest.raises(ValueError, match="not an ANTsPy call"):
        recipes.ants_argv(recipes.recipe("antspy-syn"), "/f", "/m", "/o/")


def test_cohort_mi_is_the_cohort_command_with_mi_in_syn():
    argv = recipes.ants_argv(recipes.recipe("cohort-mi"), "/f.nii.gz", "/m.nii.gz", "/w/p/")
    want = list(COHORT_ARGV)
    want[want.index("CC[/f.nii.gz,/m.nii.gz,1,4]")] = "MI[/f.nii.gz,/m.nii.gz,1,32]"
    assert argv == want
    a, b = recipes.recipe("cohort"), recipes.recipe("cohort-mi")
    assert {k for k in a["syn"] if a["syn"][k] != b["syn"][k]} == {"channel_metrics"}
    seeded = recipes.ants_argv(b, "/f", "/m", "/o/", seed=7)
    assert seeded[seeded.index("--random-seed") + 1] == "7"


def test_the_antspy_call_recipes_carry_only_the_call_arguments():
    for name, t in (("antspy-syn", "SyN"), ("synquick-s", "antsRegistrationSyNQuick[s]")):
        rec = recipes.recipe(name)
        assert recipes.is_antspy(rec) and recipes.antspy_kwargs(rec) == {"type_of_transform": t}
        assert recipes.antspy_seed(rec) == 20260910
        assert recipes.normalized(rec) == rec and recipes.same_recipe(rec, recipes.recipe(name))
    assert not recipes.is_antspy(recipes.recipe("cohort"))
    with pytest.raises(ValueError, match="not an ANTsPy-call recipe"):
        recipes.antspy_kwargs(recipes.recipe("cohort"))


def test_the_ants_version_check_applies_only_where_antsregistration_runs():
    env = {"ants_version": "2.5.1.post12", "ants_program": "ANTsPy"}
    cohort, call = recipes.recipe("cohort-mi"), recipes.recipe("synquick-s")
    run.check_ants_version(env, None, cohort, ("fastwarp", "ants"))
    run.check_ants_version(env, "2.6.5", cohort, ("fastwarp",))
    run.check_ants_version(env, None, call, ("fastwarp", "ants"))
    run.check_ants_version({"ants_version": "2.6.5-gfdce4d2"}, "2.6.5", cohort, ("ants",))
    with pytest.raises(RuntimeError, match="not 2.6.5"):
        run.check_ants_version(env, "2.6.5", cohort, ("fastwarp", "ants_repeat"))
    with pytest.raises(ValueError, match="ANTsPy's own ANTs"):
        run.check_ants_version(env, "2.6.5", call, ("fastwarp", "ants"))


def test_an_antspy_call_recipe_refuses_another_ants_before_any_work(tmp_path):
    for kw in ({"ants_version": "2.6.5"}, {"ants_dir": tmp_path}):
        with pytest.raises(ValueError, match="ANTsPy's own ANTs"):
            run.run(
                data=tmp_path,
                work=tmp_path / "w",
                keys=["mindboggle"],
                recipe_name="synquick-s",
                **kw,
            )
    assert not (tmp_path / "w").exists()


def test_two_ants_runs_of_different_ants_are_not_compared():
    a = {"program": "C:/ants/antsRegistration.exe", "ants_version": "2.6.5-gfdce4d2"}
    old = {"program": a["program"], "environment": {"ants_version": "2.6.5-gfdce4d2"}}
    run.check_same_ants(a, old)
    for other in (
        {**a, "ants_version": "2.5.1"},
        {**a, "program": "ANTsPy"},
        {"program": a["program"], "environment": {"ants_version": "2.6.4"}},
    ):
        with pytest.raises(ValueError, match="different ANTs"):
            run.check_same_ants(a, other)


def _ds(n: int, protocol: str = "pairwise") -> datasets.Dataset:
    subjects = tuple(datasets.Subject(id=f"s-{i}", image=f"{i}.nii.gz") for i in range(1, n + 1))
    tpl = datasets.Subject(id="tpl", image="tpl.nii.gz")
    return datasets.Dataset(
        key="t",
        title="t",
        licence="",
        downloads=(),
        subjects=subjects,
        protocol=protocol,
        template=tpl,
    )


def test_pairwise_pairs_form_rings_in_natural_order():
    pairs = run.select_pairs(_ds(4), 6)
    assert pairs[:4] == [("s-1", "s-2"), ("s-2", "s-3"), ("s-3", "s-4"), ("s-4", "s-1")]
    assert pairs[4:] == [("s-1", "s-3"), ("s-2", "s-4")]
    assert len(run.select_pairs(_ds(4), 100)) == 12
    assert len(set(run.select_pairs(_ds(4), 100))) == 12
    assert run.select_pairs(_ds(12), 2) == [("s-1", "s-2"), ("s-2", "s-3")]
    assert all(f != m for f, m in run.select_pairs(_ds(5), 20))


def test_template_pairs_put_the_template_first():
    assert run.select_pairs(_ds(3, "template"), 2) == [("tpl", "s-1"), ("tpl", "s-2")]
    with pytest.raises(ValueError, match="protocol"):
        run.select_pairs(_ds(3, "other"), 2)


def test_engine_done_needs_ok_status_and_both_composites(tmp_path):
    folder = tmp_path / "e"
    folder.mkdir()
    assert not run.engine_done(folder)
    (folder / "run.json").write_text(json.dumps({"status": "ok"}))
    assert not run.engine_done(folder)
    (folder / "reg_Composite.h5").write_bytes(b"x")
    (folder / "reg_InverseComposite.h5").write_bytes(b"x")
    assert run.engine_done(folder)
    (folder / "run.json").write_text(json.dumps({"status": "failed"}))
    assert not run.engine_done(folder) and run.engine_status(folder) == "failed"
    rec = recipes.recipe("cohort")
    (folder / "run.json").write_text(json.dumps({"status": "ok", "recipe": rec, "iso_mm": 1.0}))
    assert run.engine_done(folder, rec, 1.0)
    assert not run.engine_done(folder, rec, 2.0)
    assert not run.engine_done(folder, recipes.recipe("syn-sh"), 1.0)
    (folder / "reg_InverseComposite.h5").unlink()
    assert not run.engine_done(folder, rec, 1.0)


def _old_style(rec: dict) -> dict:
    old = copy.deepcopy(rec)
    del old["syn"]["channel_metrics"], old["syn"]["mi_bins"]
    return old


def test_records_written_before_the_syn_metric_keys_still_count_as_done(tmp_path):
    rec = recipes.recipe("cohort")
    old = _old_style(rec)
    assert old != rec and recipes.same_recipe(old, rec)
    folder = tmp_path / "e"
    folder.mkdir()
    (folder / "reg_Composite.h5").write_bytes(b"x")
    (folder / "reg_InverseComposite.h5").write_bytes(b"x")
    (folder / "run.json").write_text(json.dumps({"status": "ok", "recipe": old, "iso_mm": 1.0}))
    assert run.engine_done(folder, rec, 1.0)
    assert not run.pair_stale({"recipe": old, "iso_mm": 1.0}, rec, 1.0)
    assert post.same_config(post.config(old, 1.0), post.config(rec, 1.0))


def test_a_genuinely_different_recipe_is_still_stale(tmp_path):
    rec = recipes.recipe("cohort")
    other = _old_style(recipes.recipe("syn-sh"))
    mi = recipes.recipe("cohort")
    mi["syn"]["channel_metrics"] = ["mi"]
    folder = tmp_path / "e"
    folder.mkdir()
    (folder / "reg_Composite.h5").write_bytes(b"x")
    (folder / "reg_InverseComposite.h5").write_bytes(b"x")
    for different in (other, mi):
        assert not recipes.same_recipe(different, rec)
        record = {"status": "ok", "recipe": different, "iso_mm": 1.0}
        (folder / "run.json").write_text(json.dumps(record))
        assert not run.engine_done(folder, rec, 1.0)
        assert run.pair_stale({"recipe": different, "iso_mm": 1.0}, rec, 1.0)
        assert not post.same_config(post.config(different, 1.0), post.config(rec, 1.0))
    assert run.pair_stale({"recipe": _old_style(rec), "iso_mm": 2.0}, rec, 1.0)


def test_plan_counts_what_is_left(tmp_path):
    p = run.plan(
        work=tmp_path,
        keys=["mindboggle"],
        recipe_name="cohort",
        pairs=None,
        quick=True,
        linear_keys=["snsx"],
    )
    d = p["datasets"]["mindboggle"]
    assert d["pairs"] == 1 and d["to_run"] == {"fastwarp": 1, "ants": 1}
    assert p["post"]["mindboggle"]["to_run"] == 1
    assert p["linear"]["snsx"]["to_run"] == {"fastwarp": 1, "ants": 1}
    assert p["total_s"] == pytest.approx(sum(run.DEFAULT_SECONDS.values()))
    without = run.plan(
        work=tmp_path, keys=["mindboggle"], recipe_name="cohort", pairs=None, quick=True, post=False
    )
    assert without["post"] == {} and without["linear"] == {}
    s = run.DEFAULT_SECONDS
    assert without["total_s"] == pytest.approx(s["ants"] + s["fastwarp"] + s["evaluate"])


def test_the_run_lock_refuses_a_second_run(tmp_path):
    with (
        run.RunLock(tmp_path),
        pytest.raises(RuntimeError, match="another run"),
        run.RunLock(tmp_path),
    ):
        pass
    assert not (tmp_path / ".run.lock").exists()


def test_a_pair_that_fails_to_evaluate_is_recorded_and_the_rest_go_on(tmp_path):
    from benchmarks.accuracy import evaluate

    rec = recipes.recipe("cohort")
    folder = tmp_path / "mindboggle" / "cohort" / "OASIS-TRT-20-1__OASIS-TRT-20-2" / "fastwarp"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"status": "ok", "recipe": rec, "iso_mm": 1.0}))
    (folder / "reg_Composite.h5").write_bytes(b"x")
    (folder / "reg_InverseComposite.h5").write_bytes(b"x")
    logged = []
    out = evaluate.evaluate(
        data=tmp_path / "data",
        work=tmp_path,
        keys=["mindboggle"],
        recipe_name="cohort",
        pairs=2,
        quick=False,
        log=logged.append,
    )
    ev = json.loads((folder.parent / "evaluation.json").read_text())
    assert ev["error"] and ev["engines"] == {}
    assert any("failed" in line for line in logged)
    assert (out / "pairs.csv").is_file()
