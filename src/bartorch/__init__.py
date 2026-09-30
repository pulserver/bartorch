"""BART, the Berkeley Advanced Reconstruction Toolbox, in-process on torch tensors.

Shapes are C order: an axis argument is an index into a tensor's shape, never
a BART bitmask.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from bartorch import (
    _fourier,
    _interp,
    _kspace,
    _settings,
    _thresh,
    _util,
    _wavelet,
    apps,
    cli,
    interop,
    io,
    learning,
    linop,
    nlop,
    optim,
    priors,
    tools,
)
from bartorch._dispatch import BartError
from bartorch._fourier import *  # noqa: F401,F403
from bartorch._interp import *  # noqa: F401,F403
from bartorch._kspace import *  # noqa: F401,F403
from bartorch._settings import *  # noqa: F401,F403
from bartorch._thresh import *  # noqa: F401,F403
from bartorch._util import *  # noqa: F401,F403
from bartorch._wavelet import *  # noqa: F401,F403

try:
    __version__ = version("bartorch")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "BartError",
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
    *_settings.__all__,
    *_fourier.__all__,
    *_interp.__all__,
    *_kspace.__all__,
    *_thresh.__all__,
    *_util.__all__,
    *_wavelet.__all__,
]
