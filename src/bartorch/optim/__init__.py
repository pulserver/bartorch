"""BART's iterative algorithms: a block per step, and a solver and a function per algorithm.

A block is one step as a torch module, which a network stacks; a solver loops
a block to BART's schedule, called as ``solver(y, A, x0=None)``; the function
``optim.fista(y, A, term)`` is that call in one expression.
"""

from __future__ import annotations

from bartorch.optim._blocks import ADMMBlock, FISTABlock, ISTBlock, PRIDUBlock
from bartorch.optim._fixed_point import FixedPoint
from bartorch.optim._functional import (
    admm,
    cg,
    fista,
    ist,
    pocs,
    pridu,
)
from bartorch.optim._linear import (
    ADMM,
    CG,
    FISTA,
    IST,
    PRIDU,
    Tikhonov,
    maxeigen,
)
from bartorch.optim._pocs import POCS, POCSBlock
from bartorch.optim._scaling import data_scaling

__all__ = [
    "ADMM",
    "ADMMBlock",
    "CG",
    "FISTA",
    "FISTABlock",
    "FixedPoint",
    "IST",
    "ISTBlock",
    "POCS",
    "POCSBlock",
    "PRIDU",
    "PRIDUBlock",
    "Tikhonov",
    "admm",
    "cg",
    "data_scaling",
    "fista",
    "ist",
    "pocs",
    "maxeigen",
    "pridu",
]
