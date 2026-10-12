"""Phantoms and signal models."""

from __future__ import annotations

import torch

from bartorch import _call
from bartorch._call import curated
from bartorch._dispatch import dispatch

__all__ = ["phantom"]


@curated("phantom")
def phantom(
    shape: int | tuple[int, ...] | None = None,
    *,
    kspace: bool = False,
    coils: int | None = None,
    traj: torch.Tensor | None = None,
    geometry: str | None = None,
    rotation_angle: float | None = None,
    rotation_steps: int | None = None,
    **extra,
) -> torch.Tensor:
    """Analytical phantom, as an image or as its k-space.

    Parameters
    ----------
    shape : int or tuple of int, default=None
        Spatial size (``-x``).  The phantom is square or cubic: a tuple repeats
        one size, and three entries select the three-dimensional phantom.
        BART's default is 128.
    kspace : bool, default=False
        Return the analytical k-space rather than the image (``-k``).
    coils : int, default=None
        Number of simulated coils (``-s``); adds a leading coil axis.
    traj : torch.Tensor, default=None
        Trajectory ``(..., samples, 3)`` in grid units to sample the k-space
        along (``-t``) rather than a grid; implies ``kspace``.
    geometry : {'circle', 'brain', 'nist', 'sonar', 'tubes', 'bart'}, default=None
        Phantom geometry, each one of BART's flags; BART's default is the
        Shepp-Logan phantom.
    rotation_angle, rotation_steps : float, int, default=None
        Rotation angle in degrees (``--rotation-angle``), and the number of
        frames rotated by it, for ``"nist"``, ``"sonar"`` and ``"tubes"``
        (``--rotation-steps``).
    **extra
        Further BART ``phantom`` options, by name.

    Returns
    -------
    torch.Tensor
        ``(coils, z, y, x)`` with ``coils``, otherwise ``(y, x)`` or
        ``(z, y, x)``, with leading axes for frames; with ``traj``, the samples
        as :func:`bartorch.nufft` returns them.

    Raises
    ------
    ValueError
        If ``shape`` has unequal sizes or ``geometry`` is unknown.

    Examples
    --------
    >>> image = phantom(128)
    >>> kspace = phantom(128, coils=8, kspace=True)
    """
    geometries = {
        "circle": "c",
        "brain": "BRAIN",
        "nist": "NIST",
        "sonar": "SONAR",
        "tubes": "T",
        "bart": "B",
    }
    flags: dict = dict(extra)
    if shape is not None:
        if isinstance(shape, int):
            flags["x"] = shape
        else:
            spatial = tuple(shape)
            if len(set(spatial)) != 1:
                raise ValueError(
                    "BART's phantom is square or cubic; give one size, or resize after"
                )
            flags["x"] = spatial[0]
            if len(spatial) == 3:
                flags["flag_3"] = True
    if kspace:
        flags["k"] = True
    if coils is not None:
        flags["s"] = coils
    if traj is not None:
        flags["t"] = traj
    if geometry is not None:
        if geometry not in geometries:
            raise ValueError(f"geometry must be one of {sorted(geometries)}, not {geometry!r}")
        flags[geometries[geometry]] = True
    if rotation_angle is not None:
        flags["rotation_angle"] = rotation_angle
    if rotation_steps is not None:
        flags["rotation_steps"] = rotation_steps
    return dispatch("phantom", [], None, **flags)


#: Commands in this section without a hand-written wrapper, built from the catalogue.
_DERIVED = ("coils", "fakeksp", "noise")

for _name in _DERIVED:
    globals()[_name] = _call.build(_name, __name__)
del _name

__all__ = [*__all__, *_DERIVED]
