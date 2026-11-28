# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""The opt-in CPU engine (``engine="cpu"``, ``"auto"``, ``FASTWARP_ENGINE``) on a small phantom.

None of these tests needs a GPU: they run in the check tier as they are.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import fastwarp as g
import numpy as np
import pytest
from test_register import SHIFT_MM, SPACING, phantom

LINEAR = {"iterations": [40, 20], "shrink_factors": [2, 1], "smoothing_sigmas": [1, 0]}
SMALL = {
    "rigid": LINEAR,
    "affine": LINEAR,
    "syn": {"iterations": [10, 5], "shrink_factors": [2, 1], "smoothing_sigmas": [1, 0]},
}
MISSING = "zzqx 9000 7f3c"


@pytest.fixture
def no_gpu(monkeypatch):
    """No usable GPU: an unknown graphics API makes every GPU open fail."""
    monkeypatch.setenv("FASTWARP_GPU_BACKEND", "none-such")


def _register(out, **kw):
    return g.register_syn(phantom(), phantom(SHIFT_MM), out, iso_mm=SPACING, recipe=SMALL, **kw)


@pytest.fixture(scope="module")
def cpu_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("cpu") / "sub_"
    return out, _register(out, engine="cpu", write_warped=True)


def test_the_cpu_engine_registers_and_records_itself(cpu_run):
    out, info = cpu_run
    assert info["engine"] == "cpu"
    assert info["adapter"].startswith("cpu (f64, ")
    assert info["peak_gpu_bytes"] is None
    assert info["memory_budget_bytes"] is None
    for suffix in ("Composite.h5", "InverseComposite.h5", "0GenericAffine.mat", "_Warped.nii.gz"):
        assert Path(f"{out}{suffix}").stat().st_size > 0, suffix
    p = np.asarray(info["affine_parameters"])
    np.testing.assert_allclose(p[9:], SHIFT_MM, atol=0.5)


def test_fastwarp_engine_selects_the_cpu_and_an_argument_overrides_it(
    cpu_run, tmp_path, monkeypatch
):
    monkeypatch.setenv("FASTWARP_ENGINE", "cpu")
    info = _register(tmp_path / "env_")
    assert info["engine"] == "cpu"
    assert info["affine_parameters"] == cpu_run[1]["affine_parameters"]
    monkeypatch.setenv("FASTWARP_ENGINE", "tpu")
    with pytest.raises(ValueError, match="FASTWARP_ENGINE"):
        _register(tmp_path / "bad_")
    info = _register(tmp_path / "arg_", engine="cpu")
    assert info["engine"] == "cpu"


def test_auto_falls_back_to_the_cpu_with_a_warning(cpu_run, tmp_path, no_gpu):
    with pytest.warns(UserWarning, match="no usable GPU") as caught:
        info = _register(tmp_path / "auto_", engine="auto")
    assert any("much slower" in str(w.message) for w in caught)
    assert info["engine"] == "cpu"
    assert info["warnings"] == []
    assert info["affine_parameters"] == cpu_run[1]["affine_parameters"]


def test_a_fallback_warning_turned_error_stops_before_any_work(tmp_path, no_gpu):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="no usable GPU"):
            _register(tmp_path / "strict_", engine="auto")
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("engine", ["auto", "gpu", None])
def test_a_missing_adapter_is_an_error_without_a_fallback(tmp_path, engine):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(g.AdapterNotFoundError, match="'zzqx 9000 7f3c' matches no GPU"):
            _register(tmp_path / "missing_", engine=engine, adapter=MISSING)
        with pytest.raises(g.AdapterNotFoundError):
            g.gpu_available(MISSING)
    assert issubclass(g.AdapterNotFoundError, g.GpuUnavailableError)
    assert not any(tmp_path.iterdir())


def test_a_misspelled_adapter_gets_a_suggestion(tmp_path):
    if not g.gpu_available():
        pytest.skip("no usable GPU adapter to misspell")
    word = g.gpu_info()["name"].split()[0]
    if len(word) < 3:
        pytest.skip(f"the adapter name {word!r} is too short to misspell")
    typo = word[0] + ("z" if word[1].lower() == "q" else "q") + word[2:]
    with pytest.raises(g.AdapterNotFoundError, match="; did you mean '"):
        _register(tmp_path / "typo_", engine="auto", adapter=typo)


def test_engine_mistakes_are_refused_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="'gpu', 'cpu' or 'auto'"):
        _register(tmp_path / "x_", engine="tpu")
    with pytest.raises(ValueError, match="engine 'cpu'"):
        _register(tmp_path / "x_", engine="cpu", adapter="Arc")
    assert not any(tmp_path.iterdir())


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS reports no available memory")
def test_a_grid_beyond_the_host_memory_is_refused_before_any_work(tmp_path):
    with pytest.raises(g.HostMemoryError, match="the CPU engine needs about"):
        g.register_syn(phantom(), phantom(SHIFT_MM), tmp_path / "big_", iso_mm=0.01, engine="cpu")
    assert not any(tmp_path.iterdir())


def test_a_linear_session_runs_on_the_cpu():
    with g.linear_session(phantom(), phantom(SHIFT_MM), engine="cpu") as s:
        assert s.engine == "cpu"
        assert s.adapter.startswith("cpu (f64, ")
        fit = s.register(stage=LINEAR, transform="rigid")
    np.testing.assert_allclose(fit["parameters"][9:], SHIFT_MM, atol=0.5)


def test_register_atlases_falls_back_once_and_runs_on_the_cpu(tmp_path, no_gpu):
    atlases = [phantom(SHIFT_MM), phantom((-1.0, 2.0, 0.0))]
    with pytest.warns(UserWarning, match="no usable GPU") as caught:
        out = g.register_atlases(
            phantom(),
            atlases,
            None,
            tmp_path,
            recipe=SMALL,
            iso_mm=SPACING,
            engine="auto",
        )
    assert len([w for w in caught if "no usable GPU" in str(w.message)]) == 1
    assert [p.is_file() for p in out["transforms"]] == [True, True]


def test_fastwarp_ants_registration_runs_on_the_cpu(tmp_path, capsys):
    ants = pytest.importorskip("ants")
    import fastwarp.ants as fa

    def image(shift):
        return ants.from_numpy(phantom(shift)[0].astype("float32"), spacing=(SPACING,) * 3)

    result = fa.registration(
        image((0.0, 0.0, 0.0)),
        image(SHIFT_MM),
        "SyN",
        outprefix=str(tmp_path / "o_"),
        reg_iterations=(10, 5),
        verbose=True,
        engine="cpu",
    )
    assert "compatibility mode on cpu (f64, " in capsys.readouterr().out
    assert all(Path(f).is_file() for f in result["fwdtransforms"])
    assert result["fastwarp_engine"] == "cpu"
    assert "engine" not in result
    with pytest.raises(g.AdapterNotFoundError):
        fa.registration(
            image((0.0, 0.0, 0.0)),
            image(SHIFT_MM),
            "SyN",
            outprefix=str(tmp_path / "m_"),
            reg_iterations=(10, 5),
            engine="auto",
            adapter=MISSING,
            fallback="ants",
        )
