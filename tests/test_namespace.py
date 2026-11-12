# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""The public namespace is exactly ``fastwarp.__all__``."""

from __future__ import annotations

import fastwarp
import pytest


def test_dir_lists_only_the_public_api():
    assert set(dir(fastwarp)) == set(fastwarp.__all__)


def test_every_public_name_resolves():
    for name in fastwarp.__all__:
        assert getattr(fastwarp, name) is not None


def test_python_defined_functions_report_the_public_module():
    for name in ("register_syn", "apply_transforms", "linear_session", "fits_gpu"):
        assert getattr(fastwarp, name).__module__ == "fastwarp"
    assert fastwarp.LinearSession.__module__ == "fastwarp._api"


def test_private_modules_stay_importable():
    import fastwarp._api
    import fastwarp._core

    assert fastwarp._api.register_syn is fastwarp.register_syn


def test_deprecated_names_are_not_public():
    for name in ("RigidSession", "rigid_session", "register_rigid"):
        assert name not in fastwarp.__all__ and name not in dir(fastwarp)


def test_the_deprecated_names_still_work_but_warn():
    with pytest.warns(DeprecationWarning, match="LinearSession") as record:
        assert fastwarp.RigidSession is fastwarp.LinearSession
    assert record[0].filename == __file__
    with pytest.warns(DeprecationWarning, match="linear_session"):
        assert fastwarp.rigid_session is fastwarp.linear_session
    with pytest.warns(DeprecationWarning, match="register"):
        assert callable(fastwarp.register_rigid)
    with pytest.warns(DeprecationWarning) as record:
        from fastwarp import RigidSession
    assert record[0].filename == __file__
    assert RigidSession is fastwarp.LinearSession
    with pytest.raises(AttributeError):
        _ = fastwarp.no_such_name
