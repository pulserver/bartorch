"""BART's applications.

Five sections: simulation, sampling and trajectories, coil calibration,
reconstruction, and the array operations a reconstruction is surrounded by.
Two more hold what BART has no command for: corrections of data and images
outside the reconstruction, and rigid motion estimation from navigators.

Commands without a hand-written wrapper are built from BART's own declaration
of the command and take its options under their long names.
"""

from __future__ import annotations

from bartorch.tools import calib, correct, motion, process, recon, sampling, simulate
from bartorch.tools.calib import *  # noqa: F401,F403
from bartorch.tools.correct import *  # noqa: F401,F403
from bartorch.tools.motion import *  # noqa: F401,F403
from bartorch.tools.process import *  # noqa: F401,F403
from bartorch.tools.recon import *  # noqa: F401,F403
from bartorch.tools.sampling import *  # noqa: F401,F403
from bartorch.tools.simulate import *  # noqa: F401,F403

__all__ = sorted(
    {*simulate.__all__, *sampling.__all__, *calib.__all__, *recon.__all__, *process.__all__}
    | {*correct.__all__, *motion.__all__}
)
