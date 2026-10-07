"""Which ``libbartorch`` loads: an installed CUDA build, or the package's own.

``bartorch[cu12]`` and ``bartorch[cu13]`` install the library compiled with
CUDA as a package of its own, ``bartorch_cuda12`` or ``bartorch_cuda13``.  The
one that loads has to be for torch's CUDA major version and from the same
release as the package; anything else is refused by name rather than loaded.
The installed builds and the versions are stood in for here, so every case
runs without a CUDA build installed.
"""

import re
from pathlib import Path

import pytest
import torch

from bartorch import __version__, _lib

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def installed(monkeypatch, tmp_path):
    """Stand in for the CUDA builds installed and the versions they report."""

    def install(majors, versions=None, cuda=None):
        builds = {m: tmp_path / f"bartorch_cuda{m}" for m in majors}
        versions = versions or {m: __version__ for m in majors}
        monkeypatch.setattr(_lib, "_cuda_builds", lambda: builds)
        monkeypatch.setattr(torch.version, "cuda", cuda)

        def version(name):
            major = int(name.removeprefix("bartorch-cuda"))
            if major not in versions:
                raise _lib.metadata.PackageNotFoundError(name)
            return versions[major]

        monkeypatch.setattr(_lib.metadata, "version", version)
        return builds

    return install


def test_without_a_cuda_build_the_package_loads_its_own(installed):
    installed([], cuda="12.6")
    assert _lib._cuda_build() is None


def test_the_build_for_torchs_cuda_major_version_loads(installed):
    builds = installed([12, 13], cuda="12.6")
    assert _lib._cuda_build() == builds[12]
    builds = installed([12, 13], cuda="13.0")
    assert _lib._cuda_build() == builds[13]


def test_a_build_for_another_cuda_major_version_is_refused(installed):
    installed([12], cuda="13.0")
    with pytest.raises(ImportError, match=re.escape("install bartorch[cu13]")):
        _lib._cuda_build()


def test_with_a_cpu_torch_the_newest_build_loads(installed):
    """No card to run on, so the build only has to load and report none."""
    builds = installed([12, 13], cuda=None)
    assert _lib._cuda_build() == builds[13]


def test_a_build_from_another_release_is_refused(installed):
    """Its library is another release's ABI."""
    installed([12], versions={12: "0.0.1"}, cuda="12.6")
    with pytest.raises(ImportError, match=re.escape(f"bartorch[cu12]=={__version__}")):
        _lib._cuda_build()


def test_bartorch_library_overrides_the_search(installed, monkeypatch, tmp_path):
    installed([12], versions={12: "0.0.1"}, cuda="13.0")
    monkeypatch.setenv("BARTORCH_LIBRARY", str(tmp_path / "libbartorch.so"))
    assert _lib._find_library() == tmp_path / "libbartorch.so"


@pytest.mark.parametrize("major", _lib.CUDA_MAJORS)
def test_each_cuda_build_is_pinned_to_its_release_and_offered_as_an_extra(major):
    """The pin is what makes `bartorch[cuNN]` resolve to the build of that bartorch."""
    tomllib = pytest.importorskip("tomllib", reason="pyproject is read with tomllib, a 3.11 module")
    build = tomllib.loads((ROOT / "src" / "cuda" / str(major) / "pyproject.toml").read_text())
    assert build["project"]["name"] == f"bartorch-cuda{major}"
    assert build["tool"]["scikit-build"]["cmake"]["define"]["BARTORCH_CUDA_PACKAGE"] == str(major)
    (dependencies,) = [d for d in build["tool"]["dynamic-metadata"] if d["field"] == "dependencies"]
    assert "bartorch=={project[version]}" in dependencies["result"]

    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    (spec,) = package["project"]["optional-dependencies"][f"cu{major}"]
    assert spec.startswith(f"bartorch-cuda{major};")
