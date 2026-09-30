"""Every public name has a place in the API reference (``docs/api/*.md``).

The API pages list their objects in tables whose first column is an ``{obj}``
role; ``docs/api_objects.py`` collects those tables into the page the
per-object stubs are generated from.
"""

import ast
import importlib.util
import json
from importlib import import_module
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
API = DOCS / "api"

#: Public modules and the names of theirs that are modules or data, not entries.
MODULES = {
    "bartorch": {
        "__version__",
        "apps",
        "cli",
        "interop",
        "io",
        "learning",
        "linop",
        "nlop",
        "optim",
        "priors",
        "tools",
    },
    "bartorch.learning": set(),
    "bartorch.linop": set(),
    "bartorch.nlop": set(),
    "bartorch.optim": set(),
    "bartorch.priors": set(),
    "bartorch.tools": set(),
    "bartorch.apps": set(),
    "bartorch.cli": set(),
    "bartorch.io": set(),
    "bartorch.interop": set(),
}


def _public(module: str) -> set[str]:
    return set(import_module(module).__all__)


def _api_objects():
    """``docs/api_objects.py``, which is not a package module."""
    spec = importlib.util.spec_from_file_location("api_objects", DOCS / "api_objects.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _listed() -> dict[str, set[str]]:
    """Names listed in the API pages' object tables, by module."""
    return {module: set(names) for module, names in _api_objects().collect(API).items()}


@pytest.mark.parametrize("module", sorted(MODULES))
def test_every_public_name_is_in_the_reference(module):
    public = _public(module) - MODULES[module]
    missing = public - _listed().get(module, set())
    assert not missing, f"{module}: not in docs/api: {sorted(missing)}"


@pytest.mark.parametrize("module", sorted(MODULES))
def test_the_reference_lists_nothing_that_is_not_public(module):
    extra = _listed().get(module, set()) - _public(module)
    assert not extra, f"{module}: listed in docs/api but not public: {sorted(extra)}"


def test_the_object_index_generates_a_page_for_every_listed_object():
    """The holder page carries every table entry under its module's autosummary."""
    objects = _api_objects()
    blocks = objects.collect(API)
    text = objects.render(blocks)
    assert text.startswith(":orphan:")
    for module, names in blocks.items():
        assert f".. currentmodule:: {module}" in text
        for name in names:
            assert f"\n   {name}\n" in text, f"{module}.{name} has no stub"


def test_api_pages_carry_no_visible_autosummary():
    """Object lists are human-written tables; autosummary runs on the hidden holder page."""
    for page in API.glob("*.md"):
        assert ".. autosummary::" not in page.read_text(), page.name


@pytest.mark.parametrize(
    "page",
    sorted(p.name for p in (DOCS / "explanation").glob("*.md") if p.name != "index.md"),
)
def test_every_explanation_page_opens_with_a_tldr(page):
    """The page's title, then a TL;DR block before anything else."""
    lines = (DOCS / "explanation" / page).read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# "), page
    following = [line for line in lines[1:] if line.strip()]
    assert following[:2] == ["```{admonition} TL;DR", ":class: tldr"], page


def _colab():
    """``docs/colab.py``, which is not a package module."""
    spec = importlib.util.spec_from_file_location("colab", DOCS / "colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXAMPLE_PAGE = """.. _sphx_glr_auto_examples_01-basics_01-from-kspace-to-image.py:


=====================
From k-space to image
=====================

Reconstruction of an undersampled Cartesian acquisition.
"""


def test_an_example_page_carries_the_colab_badge_under_its_title():
    colab = _colab()
    text = colab.with_badge(
        _EXAMPLE_PAGE, "auto_examples/01-basics/01-from-kspace-to-image", "latest"
    )
    title_end = text.index("=====================\n\n", text.index("From k-space")) + 22
    badge = text.index("colab-badge.svg")
    assert title_end < badge < text.index("Reconstruction of")
    assert "blob/gh-pages/latest/_colab/01-basics/01-from-kspace-to-image.ipynb" in text
    section = colab.with_badge(_EXAMPLE_PAGE, "auto_examples/01-basics/index", "latest")
    assert section == _EXAMPLE_PAGE


def test_the_colab_notebook_is_the_gallery_notebook_after_a_setup_cell(tmp_path):
    """The downloadable notebook is left alone; the Colab copy installs first."""
    colab = _colab()
    notebook = {"cells": [{"cell_type": "code", "source": ["import bartorch"]}], "nbformat": 4}
    source = tmp_path / "docs" / "auto_examples" / "06-learning" / "01-modl.ipynb"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(notebook))

    assert colab.write(tmp_path / "docs", tmp_path / "site", "v1.2.3") == 1
    copy = json.loads((tmp_path / "site" / "_colab" / "06-learning" / "01-modl.ipynb").read_text())
    assert json.loads(source.read_text()) == notebook
    install = "".join(copy["cells"][1]["source"])
    assert install.startswith("%pip install")
    assert "bartorch==1.2.3" in install and "deepinv" in install
    assert copy["cells"][2:] == notebook["cells"]
    assert "bartorch " in colab.setup_cells("01-basics", "latest")[1]["source"][0] + " "


def _toctree(page: Path) -> list[str]:
    """The entries of the toctrees of a Markdown page."""
    entries, inside = [], False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("```{toctree}"):
            inside = True
        elif inside and line.startswith("```"):
            inside = False
        elif inside and line.strip() and not line.startswith(":"):
            entries.append(line.strip())
    return entries


def test_the_user_guide_holds_installation_and_support_pages_only():
    """Concepts and conventions are explanation pages, not user-guide pages."""
    user = DOCS / "guides" / "user"
    pages = ["prerequisites", "installation", "issues", "discussions", "security"]
    assert _toctree(user / "index.md") == pages
    assert sorted(p.stem for p in user.glob("*.md")) == sorted([*pages, "index"])


def _gallery_sections() -> list[str]:
    """``GALLERY_SECTIONS`` of ``docs/conf.py``, read without executing it."""
    tree = ast.parse((DOCS / "conf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        target = getattr(node, "targets", [None])[0]
        if isinstance(node, ast.Assign) and getattr(target, "id", "") == "GALLERY_SECTIONS":
            return [Path(s).name for s in ast.literal_eval(node.value)]
    raise AssertionError("docs/conf.py defines no GALLERY_SECTIONS")


def test_the_examples_page_is_the_gallery_root_and_links_every_section():
    """Examples > section > example: the sidebar enters the gallery at its root."""
    sections = _gallery_sections()
    on_disk = sorted(p.parent.name for p in (DOCS / "examples").glob("*/README.rst"))
    assert sorted(sections) == on_disk
    assert "auto_examples/index" in _toctree(DOCS / "index.md")
    assert not (DOCS / "examples" / "index.md").exists()
    root = (DOCS / "examples" / "README.rst").read_text(encoding="utf-8")
    linked = [line.split("`")[1] for line in root.splitlines() if ":doc:`0" in line]
    assert linked == [f"{section}/index" for section in sections]
