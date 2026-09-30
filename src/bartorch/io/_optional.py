import importlib
from types import ModuleType


def require(name: str) -> ModuleType:
    """Import an optional dependency of :mod:`bartorch.io`, naming the extra that installs it."""
    try:
        return importlib.import_module(name)
    except ImportError as error:
        package = name.split(".")[0]
        raise ImportError(
            f"{package} is needed to read and write this format: pip install 'bartorch[io]'"
        ) from error
