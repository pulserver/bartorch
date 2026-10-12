"""Trajectories and sampling patterns."""

from __future__ import annotations

from bartorch import _call
from bartorch._call import curated
from bartorch._dispatch import dispatch

__all__ = ["traj"]


@curated("traj")
def traj(
    *,
    readout: int | None = None,
    spokes: int | None = None,
    frames: int | None = None,
    radial: bool = False,
    golden: bool = False,
    turns: int | None = None,
    oversampling: bool = False,
    **extra,
):
    """k-space trajectory in grid units, Cartesian or radial.

    Parameters
    ----------
    readout : int, default=None
        Samples per readout (``-x``); BART's default is 128.
    spokes : int, default=None
        Readouts per frame (``-y``); BART's default is 128.
    frames : int, default=None
        Number of frames along BART's time axis, each rotated by a fraction of
        the angular increment (``-t``, BART's ``turns``); radial only.
    radial : bool, default=False
        Radial spokes rather than Cartesian lines (``-r``).
    golden : bool, default=False
        Golden-angle increment between spokes (``-G``); implies ``radial``.
    turns : int, default=None
        Simultaneous multi-slice factor: partitions along BART's slice axis
        (``-m``); radial only.
    oversampling : bool, default=False
        Correct the transverse gradient error of a radial trajectory (``-O``).
        Readout oversampling is ``traj -o``, passed as ``o=factor``.
    **extra
        Further BART ``traj`` options, by name.

    Returns
    -------
    torch.Tensor
        Coordinates ``(..., spokes, readout, 3)``, ``kx, ky, kz`` in grid
        units, with leading axes for ``frames`` and ``turns`` when they exceed
        one.  The layout every non-Cartesian transform in this package takes.

    Examples
    --------
    >>> trajectory = traj(readout=256, spokes=64, radial=True, golden=True)
    """
    flags: dict = dict(extra)
    for keyword, value in (("x", readout), ("y", spokes), ("t", frames), ("m", turns)):
        if value is not None:
            flags[keyword] = value
    if radial:
        flags["r"] = True
    if golden:
        flags["G"] = True
    if oversampling:
        flags["O"] = True
    return dispatch("traj", [], None, **flags)


#: Commands in this section without a hand-written wrapper, built from the catalogue.
_DERIVED = (
    "bin",
    "estdelay",
    "estdims",
    "grid",
    "grog",
    "pattern",
    "poisson",
    "psf",
    "raga",
    "rmfreq",
    "ssa",
    "trajcor",
    "upat",
    "wavepsf",
)

for _name in _DERIVED:
    globals()[_name] = _call.build(_name, __name__)
del _name

__all__ = [*__all__, *_DERIVED]
