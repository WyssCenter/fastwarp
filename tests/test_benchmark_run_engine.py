# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import sys
import types

import fastwarp as g
import pytest

from benchmarks.accuracy import recipes, run

REC = recipes.recipe("cohort")


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    calls = {"fits": 0, "register": []}
    monkeypatch.setitem(
        sys.modules, "ants", types.SimpleNamespace(image_read=lambda path: object())
    )

    def fits_gpu(*args, **kw):
        calls["fits"] += 1
        return False

    def register_syn(*args, **kw):
        calls["register"].append(kw)
        return {"recipe": REC}

    monkeypatch.setattr(g, "fits_gpu", fits_gpu)
    monkeypatch.setattr(g, "register_syn", register_syn)
    monkeypatch.setattr(run.timing, "fastwarp_registration_timings", lambda info: {})
    monkeypatch.delenv("FASTWARP_ENGINE", raising=False)
    monkeypatch.setenv("FASTWARP_GPU_BACKEND", "none-such")
    return calls


def _run(tmp_path, adapter=None):
    return run.run_fastwarp(
        tmp_path / "f.nii.gz", tmp_path / "m.nii.gz", tmp_path, REC, iso_mm=1.0, adapter=adapter
    )


def test_the_cpu_engine_without_a_gpu_runs_the_pair(stubs, monkeypatch, tmp_path):
    monkeypatch.setenv("FASTWARP_ENGINE", "cpu")
    record = _run(tmp_path)
    assert record["status"] == "ok"
    assert record["fastwarp_engine"] == "cpu"
    assert stubs["fits"] == 0
    assert stubs["register"][0]["engine"] == "cpu"
    assert stubs["register"][0]["adapter"] is None


def test_auto_without_a_gpu_falls_back_to_the_cpu_and_runs_the_pair(stubs, monkeypatch, tmp_path):
    monkeypatch.setenv("FASTWARP_ENGINE", "auto")
    with pytest.warns(UserWarning, match="no usable GPU"):
        record = _run(tmp_path)
    assert record["status"] == "ok"
    assert stubs["fits"] == 0
    assert stubs["register"][0]["engine"] == "cpu"


def test_the_gpu_engine_without_a_gpu_skips_the_pair(stubs, tmp_path):
    record = _run(tmp_path)
    assert record["status"] == "skipped"
    assert stubs["fits"] == 1
    assert stubs["register"] == []


def test_a_gpu_skip_is_a_different_config_from_a_cpu_run(stubs, monkeypatch, tmp_path):
    skipped = _run(tmp_path)
    assert skipped["status"] == "skipped"
    assert skipped["fastwarp_engine"] == "gpu"
    assert run.same_config(skipped, REC, 1.0, "gpu")
    assert not run.same_config(skipped, REC, 1.0, "cpu")
    assert run.same_config(skipped, REC, 1.0)
    legacy = {k: v for k, v in skipped.items() if k != "fastwarp_engine"}
    assert run.same_config(legacy, REC, 1.0, "gpu")
    assert not run.same_config(legacy, REC, 1.0, "cpu")


def test_the_library_engine_resolution_the_benchmark_imports_keeps_its_signature(monkeypatch):
    from fastwarp._api import _settled_engine

    params = list(inspect.signature(_settled_engine).parameters)
    assert params == ["engine", "adapter", "recipe"]
    assert _settled_engine("cpu", None, REC) == ("cpu", None)
    monkeypatch.setenv("FASTWARP_ENGINE", "cpu")
    assert _settled_engine(None, None, REC) == ("cpu", None)


def test_an_ok_gpu_record_is_not_done_for_the_cpu_engine(stubs, tmp_path):
    (tmp_path / "reg_Composite.h5").write_bytes(b"x")
    (tmp_path / "reg_InverseComposite.h5").write_bytes(b"x")
    record = {"status": "ok", "recipe": REC, "iso_mm": 1.0, "fastwarp_engine": "gpu"}
    (tmp_path / "run.json").write_text(json.dumps(record))
    assert run.engine_done(tmp_path, REC, 1.0)
    assert run.engine_done(tmp_path, REC, 1.0, "gpu")
    assert not run.engine_done(tmp_path, REC, 1.0, "cpu")
    record.pop("fastwarp_engine")
    (tmp_path / "run.json").write_text(json.dumps(record))
    assert run.engine_done(tmp_path, REC, 1.0, "gpu")
    assert not run.engine_done(tmp_path, REC, 1.0, "cpu")


def test_the_run_settles_the_engine_once_and_reports_a_bad_setting(stubs, monkeypatch, tmp_path):
    monkeypatch.setenv("FASTWARP_ENGINE", "auto")
    with pytest.warns(UserWarning, match="no usable GPU") as caught:
        settled = run.settle_for_run(REC, None)
        for _ in range(2):
            _run_settled(tmp_path, settled)
    assert settled == ("cpu", None)
    assert len([w for w in caught if "no usable GPU" in str(w.message)]) == 1
    monkeypatch.setenv("FASTWARP_ENGINE", "tpu")
    with pytest.raises(SystemExit, match="FASTWARP_ENGINE='tpu'"):
        run.settle_for_run(REC, None)
    monkeypatch.setenv("FASTWARP_ENGINE", "cpu")
    with pytest.raises(SystemExit, match="FASTWARP_ENGINE='cpu'"):
        run.settle_for_run(REC, "Arc")


def _run_settled(tmp_path, settled):
    return run.run_fastwarp(
        tmp_path / "f.nii.gz",
        tmp_path / "m.nii.gz",
        tmp_path,
        REC,
        iso_mm=1.0,
        adapter=None,
        settled=settled,
    )
