# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""The public namespace is exactly ``fastwarp.__all__``."""

from __future__ import annotations

import fastwarp


def test_dir_lists_only_the_public_api():
    assert set(dir(fastwarp)) == set(fastwarp.__all__)


def test_every_public_name_resolves():
    for name in fastwarp.__all__:
        assert getattr(fastwarp, name) is not None


def test_python_defined_objects_report_the_public_module():
    for name in ("register_syn", "apply_transforms", "linear_session", "LinearSession"):
        assert getattr(fastwarp, name).__module__ == "fastwarp"


def test_private_modules_stay_importable():
    import fastwarp._api
    import fastwarp._core

    assert fastwarp._api.register_syn is fastwarp.register_syn


def test_deprecated_names_are_not_public():
    for name in ("RigidSession", "rigid_session", "register_rigid"):
        assert name not in fastwarp.__all__ and name not in dir(fastwarp)
