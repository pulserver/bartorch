"""BART's applications.

Five sections: simulation, sampling and trajectories, coil calibration,
preprocessing and measurement, and low-rank completion.  Reconstruction is
:mod:`bartorch.apps`, or an encoding from :mod:`bartorch.linop` under a solver
from :mod:`bartorch.optim`.

Commands without a hand-written wrapper are built from BART's own declaration
of the command and take its options under their long names.
"""

from __future__ import annotations

from bartorch.tools import calib, lowrank, process, sampling, simulate
from bartorch.tools.calib import *  # noqa: F401,F403
from bartorch.tools.lowrank import *  # noqa: F401,F403
from bartorch.tools.process import *  # noqa: F401,F403
from bartorch.tools.sampling import *  # noqa: F401,F403
from bartorch.tools.simulate import *  # noqa: F401,F403

__all__ = sorted(
    {*simulate.__all__, *sampling.__all__, *calib.__all__, *process.__all__, *lowrank.__all__}
)
