"""Finding and loading ``libbartorch``.

``BARTORCH_LIBRARY`` overrides the search.  Otherwise a CUDA build installed
beside the package (``bartorch[cu12]``, ``bartorch[cu13]``) is taken when it
matches torch's CUDA major version, and the package's own library when none is
installed.  The signatures come from :mod:`bartorch._abi`, which
``scripts/gen_abi.py`` generates from the C header.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path

from bartorch._abi import (  # noqa: F401  (re-exported: these are the ABI's vocabulary)
    ALLOC_FN,
    APPLY_FN,
    DIMS,
    FREE_FN,
    GENERIC_APPLY_FN,
    LOG_FN,
    LOG_LEVELS,
    PAIR_APPLY_FN,
    RELEASE_FN,
    SYMBOLS,
    bind,
)


def _library_names() -> list[str]:
    if sys.platform == "darwin":
        return ["libbartorch.dylib"]
    if sys.platform == "win32":
        return ["libbartorch.dll"]
    return ["libbartorch.so"]


def _load(path: Path) -> ctypes.CDLL:
    # The library imports the OpenMP runtime torch carries -- libgomp.so.1 on
    # Linux, @rpath/libomp.dylib on macOS, libiomp5md.dll on Windows -- rather
    # than one of its own.  Importing torch first makes the copy torch loaded
    # the one the loader binds to; torch's directory is where it is found
    # otherwise.
    import torch

    if sys.platform == "win32":
        os.add_dll_directory(str(Path(torch.__file__).resolve().parent / "lib"))
    return ctypes.CDLL(str(path))


def _package_dirs() -> list[Path]:
    """Every directory the package's files are in.

    A wheel puts the library beside the Python sources.  An editable install
    leaves the sources where they are and adds the directory it built into to
    the package's ``__path__``, so that is where the library is then.
    """
    roots = [Path(__file__).resolve().parent]
    package = sys.modules.get(__package__ or "bartorch")
    for entry in getattr(package, "__path__", ()):
        root = Path(entry).resolve()
        if root not in roots:
            roots.append(root)
    return roots


# The CUDA major versions a bartorch-cudaNN wheel is built for (src/cuda/NN).
CUDA_MAJORS = (12, 13)


def _cuda_builds() -> dict[int, Path]:
    """The directory of each CUDA build installed, by CUDA major version."""
    found = {}
    for major in CUDA_MAJORS:
        spec = importlib.util.find_spec(f"bartorch_cuda{major}")
        if spec is not None and spec.origin is not None:
            found[major] = Path(spec.origin).resolve().parent
    return found


def _cuda_build() -> Path | None:
    """The directory of the CUDA build to load, or None for the package's own library.

    The build has to be for torch's CUDA major version, whose runtime torch has
    loaded, and from the same release as the package, whose ABI it is.  With a
    CPU build of torch there is no card to run on and the newest build loads.
    """
    builds = _cuda_builds()
    if not builds:
        return None
    import torch

    from bartorch import __version__

    if torch.version.cuda is None:
        major = max(builds)
    else:
        major = int(torch.version.cuda.split(".")[0])
    names = ", ".join(f"bartorch-cuda{m}" for m in sorted(builds))
    if major not in builds:
        raise ImportError(
            f"bartorch: torch is built for CUDA {torch.version.cuda}, and the CUDA build "
            f"installed is {names}; install bartorch[cu{major}]."
        )
    try:
        built = metadata.version(f"bartorch-cuda{major}")
    except metadata.PackageNotFoundError:
        built = None
    if built != __version__:
        raise ImportError(
            f"bartorch: bartorch {__version__} is installed with bartorch-cuda{major} {built}, "
            f"whose library is another release's; install bartorch[cu{major}]=={__version__}."
        )
    return builds[major]


def _find_library() -> Path:
    override = os.environ.get("BARTORCH_LIBRARY")
    if override:
        return Path(override)
    cuda = _cuda_build()
    roots = [cuda] if cuda is not None else _package_dirs()
    for root in roots:
        for name in _library_names():
            candidate = root / name
            if candidate.exists():
                return candidate
    raise ImportError(
        "bartorch: compiled library not found next to the package. "
        "Build it with `pip install .` or point BARTORCH_LIBRARY at libbartorch."
    )


_lib: ctypes.CDLL | None = None
_path: Path | None = None


def library() -> ctypes.CDLL:
    """Return the loaded library, loading it on first use."""
    global _lib, _path
    if _lib is None:
        _path = _find_library()
        _lib = bind(_load(_path))
    return _lib


def library_path() -> Path:
    library()
    assert _path is not None
    return _path
