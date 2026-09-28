"""What the package asks to be installed with, and why.

FINUFFT, and cuFINUFFT in a CUDA build, are compiled into the library from the
pinned submodule, so neither Python package is a requirement or an extra, and
a NUFFT runs in an environment that has neither.

deepinv is an extra: denoisers, losses and samplers come from it through
`priors.ImplicitPrior` and `bartorch.interop`, and nothing else imports it.

torchsim is a dependency because `nlop.SignalModel`
turns any of its simulators into a BART nonlinear operator, and
`nlop.InversionRecovery`, `nlop.MultiEcho` and `nlop.Bloch` are `moba`'s
families written on it, so every quantitative reconstruction here imports it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib", reason="pyproject is read with tomllib, a 3.11 module")
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _requirements(specs: list[str], name: str) -> list:
    """Every requirement on *name* in *specs*, parsed."""
    packaging = pytest.importorskip("packaging.requirements")
    return [r for r in (packaging.Requirement(s) for s in specs) if r.name == name]


@pytest.mark.parametrize("name", ["finufft", "cufinufft"])
def test_neither_finufft_package_is_asked_for(name):
    """The transforms are compiled in; a requirement would install a second copy."""
    project = _pyproject()["project"]
    assert not _requirements(project["dependencies"], name)
    for extra, specs in project["optional-dependencies"].items():
        assert not _requirements(specs, name), f"the {extra} extra asks for {name}"
    assert name not in project["optional-dependencies"]


def test_deepinv_is_an_extra_and_not_a_dependency():
    project = _pyproject()["project"]
    assert not _requirements(project["dependencies"], "deepinv"), (
        "only the adapter and the denoisers a caller brings use deepinv; "
        "it belongs in optional-dependencies"
    )
    assert _requirements(project["optional-dependencies"]["deepinv"], "deepinv")


def test_torchsim_is_a_dependency_and_not_an_extra():
    """Every quantitative reconstruction here goes through it."""
    project = _pyproject()["project"]
    assert _requirements(project["dependencies"], "torchsim"), (
        "nlop.SignalModel turns a TorchSim simulator into a BART operator and "
        "the moba families are written on it; torchsim belongs in dependencies, "
        "not in optional-dependencies"
    )
    assert "torchsim" not in project["optional-dependencies"], (
        "an extra named torchsim says the signal models are optional, and they are not"
    )


def test_the_models_really_do_reach_torchsim():
    """The claim the requirement rests on, rather than the requirement alone."""
    from torchsim.recon import ModelOperator

    from bartorch import nlop

    model = nlop.MultiEcho((10.0, 40.0), (2,))
    assert isinstance(model.model, ModelOperator)


def test_the_installed_metadata_asks_for_no_finufft_package():
    """What pip was actually told, read off the installed distribution."""
    from importlib import metadata

    packaging = pytest.importorskip("packaging.requirements")
    try:
        specs = metadata.requires("bartorch") or []
    except metadata.PackageNotFoundError:
        pytest.skip("bartorch is on the path but not installed, so it has no metadata")
    names = {packaging.Requirement(s).name for s in specs}
    assert not names & {"finufft", "cufinufft"}


def test_a_nufft_runs_where_no_finufft_package_can_be_imported():
    """In a fresh interpreter whose import system refuses both packages."""
    code = (
        "import sys\n"
        "class Refuse:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('finufft', 'cufinufft'):\n"
        "            raise ImportError(name + ' is refused in this test')\n"
        "sys.meta_path.insert(0, Refuse())\n"
        "import bartorch, bartorch.tools as bt\n"
        "from bartorch import _finufft\n"
        "traj = bt.traj(x=32, y=16, r=True)\n"
        "img = bt.phantom([32, 32]).reshape(1, 32, 32)\n"
        "_finufft.reset_counters()\n"
        "y = bartorch.nufft(img, traj)\n"
        "assert _finufft.operators_built() == (1, 0), _finufft.operators_built()\n"
        "assert float(y.abs().max()) > 0\n"
        "assert not {'finufft', 'cufinufft'} & set(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
