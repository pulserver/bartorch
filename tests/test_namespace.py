"""Each namespace's public interface is flat: its names, and no public module beneath them."""

import pkgutil
import subprocess
import sys
from importlib import import_module

import pytest

import bartorch

#: The public modules of ``bartorch`` itself: the subpackages and two modules.
NAMESPACES = (
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
)

SUBPACKAGES = sorted(info.name for info in pkgutil.iter_modules(bartorch.__path__) if info.ispkg)


def _python_modules(path):
    """The Python modules and packages under ``path``.

    An installed wheel places the compiled library and its FINUFFT modules in
    the package directory, where ``pkgutil`` lists them as extension modules;
    they are shared libraries loaded by path, not modules.
    """
    for info in pkgutil.iter_modules(path):
        spec = info.module_finder.find_spec(info.name)
        if info.ispkg or (spec is not None and spec.origin.endswith(".py")):
            yield info


def test_bartorch_exposes_no_public_module_but_its_namespaces():
    public = {
        info.name for info in _python_modules(bartorch.__path__) if not info.name.startswith("_")
    }
    assert public == set(NAMESPACES)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_no_subpackage_of_bartorch_exposes_a_public_submodule(name):
    package = import_module(f"bartorch.{name}")
    public = [
        info.name for info in _python_modules(package.__path__) if not info.name.startswith("_")
    ]
    assert not public, f"bartorch.{name}: public submodules {public}"


@pytest.mark.parametrize("name", NAMESPACES)
def test_every_name_a_namespace_exports_is_reachable_from_it(name):
    namespace = import_module(f"bartorch.{name}")
    for public in namespace.__all__:
        if public in ("Reconstruction", "RandomGain") and name == "learning":
            pytest.importorskip("lightning")
            pytest.importorskip("torchio")
        assert getattr(namespace, public) is not None


def test_importing_learning_imports_no_training_library():
    script = (
        "import sys, bartorch.learning\n"
        "print(sorted({'lightning', 'torchio'} & set(sys.modules)))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]"
