"""BART's applications, and the corrections a reconstruction is surrounded by.

Five sections of BART commands: simulation, sampling and trajectories, coil
calibration, preprocessing and measurement, and low-rank completion.  Two more
hold what BART has no command for: corrections of data and images outside the
reconstruction, and rigid motion estimation from navigators.  Reconstruction is
:mod:`bartorch.apps`, or an encoding from :mod:`bartorch.linop` under a solver
from :mod:`bartorch.optim`.

Commands without a hand-written wrapper are built from BART's own declaration
of the command and take its options under their long names.
"""

from __future__ import annotations

from bartorch.tools import (
    _calib,
    _correct,
    _lowrank,
    _motion,
    _process,
    _sampling,
    _simulate,
)
from bartorch.tools._calib import *  # noqa: F401,F403
from bartorch.tools._correct import *  # noqa: F401,F403
from bartorch.tools._lowrank import *  # noqa: F401,F403
from bartorch.tools._motion import *  # noqa: F401,F403
from bartorch.tools._process import *  # noqa: F401,F403
from bartorch.tools._sampling import *  # noqa: F401,F403
from bartorch.tools._simulate import *  # noqa: F401,F403

__all__ = sorted(
    {*_simulate.__all__, *_sampling.__all__, *_calib.__all__, *_process.__all__}
    | {*_lowrank.__all__, *_correct.__all__, *_motion.__all__}
)


def __getattr__(name: str):
    # A BART command reached here by its name says where bartorch exposes it.
    from bartorch._coverage import PRIVATE, curated_wrappers

    if name in PRIVATE:
        raise AttributeError(f"bartorch.tools has no {name!r}: {PRIVATE[name]}")
    found = curated_wrappers().get(name, ())
    places = sorted(
        {
            ".".join(p for p in w.__module__.split(".") if not p.startswith("_")) + f".{w.__name__}"
            for w in found
        }
    )
    if places:
        raise AttributeError(f"bartorch.tools has no {name!r}; use {', '.join(places)}")
    raise AttributeError(f"module 'bartorch.tools' has no attribute {name!r}")
