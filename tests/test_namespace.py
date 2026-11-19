# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""The public namespace is exactly ``fastwarp.__all__``."""

from __future__ import annotations

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import fastwarp
import pytest

ERRORS = ("GpuCapacityError", "GpuUnavailableError", "RegistrationCancelled", "RegistrationError")


def test_dir_lists_only_the_public_api():
    assert set(dir(fastwarp)) == set(fastwarp.__all__)


def test_every_public_name_resolves():
    for name in fastwarp.__all__:
        assert getattr(fastwarp, name) is not None


def test_python_defined_functions_report_the_public_module():
    for name in ("register_syn", "apply_transforms", "linear_session", "fits_gpu"):
        assert getattr(fastwarp, name).__module__ == "fastwarp"
    assert fastwarp.LinearSession.__module__ == "fastwarp._api"


@pytest.mark.parametrize("name", ERRORS)
def test_errors_report_the_public_module_and_pickle(name):
    error = getattr(fastwarp, name)
    assert error.__module__ == "fastwarp" and error.__qualname__ == name
    copy = pickle.loads(pickle.dumps(error("boom")))
    assert type(copy) is error and copy.args == ("boom",)


def test_extension_objects_report_an_importable_module():
    assert fastwarp.gpu_info.__module__ == "fastwarp"
    assert fastwarp.CancelToken.__module__ == "fastwarp"
    assert pickle.loads(pickle.dumps(fastwarp.gpu_info)) is fastwarp.gpu_info
    assert pickle.loads(pickle.dumps(fastwarp.CancelToken)) is fastwarp.CancelToken
    core = fastwarp._core.LinearSession
    assert core.__module__ == "fastwarp._core"
    assert pickle.loads(pickle.dumps(core)) is core


def _raise(name: str) -> None:
    raise getattr(fastwarp, name)(f"{name} in a worker")


def test_errors_cross_a_process_pool():
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        for name in ERRORS:
            with pytest.raises(getattr(fastwarp, name), match=f"{name} in a worker"):
                pool.submit(_raise, name).result(timeout=120)


def test_private_modules_stay_importable():
    import fastwarp._api
    import fastwarp._core

    assert fastwarp._api.register_syn is fastwarp.register_syn


def test_removed_names_are_not_public():
    for name in ("RigidSession", "rigid_session", "register_rigid"):
        assert name not in fastwarp.__all__ and name not in dir(fastwarp)


def test_the_removed_rigid_names_are_gone():
    import fastwarp._api
    import fastwarp._core

    for name in ("RigidSession", "rigid_session", "register_rigid"):
        with pytest.raises(AttributeError):
            getattr(fastwarp, name)
        with pytest.raises(ImportError):
            exec(f"from fastwarp import {name}")
        assert not hasattr(fastwarp._api, name)
    assert not hasattr(fastwarp._core, "check_linear_stage")
    with pytest.raises(AttributeError):
        _ = fastwarp.no_such_name
