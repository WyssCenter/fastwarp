# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

_cargo = (Path(__file__).resolve().parent.parent / "Cargo.toml").read_text(encoding="utf-8")
_version = re.search(r'^version\s*=\s*"([^"]+)"', _cargo, re.MULTILINE)

project = "FastWarp"
author = "Richard Köhler and Jan Philipp Payonk"
copyright = "2026 Wyss Center for Bio and Neuroengineering"
release = _version.group(1) if _version else ""
version = release

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "requirements.txt"]
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist"]

autoapi_dirs = ["../python/fastwarp"]
autoapi_file_patterns = ["*.pyi", "*.py"]
autoapi_options = ["members", "undoc-members", "imported-members"]
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_generate_api_docs = False
autoapi_add_toctree_entry = False
python_maximum_signature_line_length = 88
python_use_unqualified_type_names = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "ants": ("https://antspyx.readthedocs.io/en/latest", None),
}

copybutton_exclude = ".linenos, .gp, .go"

html_theme = "pydata_sphinx_theme"
html_title = f"FastWarp {release}"
html_static_path = ["_static"]
html_css_files = ["fastwarp.css"]
html_show_sourcelink = False
html_context = {
    "github_user": "WyssCenter",
    "github_repo": "fastwarp",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}
html_theme_options = {
    "logo": {"text": "FastWarp"},
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/WyssCenter/fastwarp",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/fastwarp/",
            "icon": "fa-brands fa-python",
        },
    ],
}
html_sidebars = {
    "index": [],
}

_PRIVATE_MODULES = ("fastwarp._core.", "fastwarp._api.")


def _public_name(app, env, node, contnode):
    target = node.get("reftarget", "")
    if node.get("refdomain") != "py" or not target.startswith(_PRIVATE_MODULES):
        return None
    public = "fastwarp." + target.split(".", 2)[2]
    resolved = env.get_domain("py").resolve_xref(
        env, node["refdoc"], app.builder, node["reftype"], public, node, contnode
    )
    if resolved is not None:
        node["reftarget"] = public
    return resolved


def setup(app):
    app.connect("missing-reference", _public_name)
