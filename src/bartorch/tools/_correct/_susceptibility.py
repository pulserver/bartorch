"""Susceptibility distortion correction from a reversed phase-encoding pair, by PyHySCO.

PyHySCO is GPL-3.0-only.  It is an optional extra (``bartorch[pyhysco]``) that
is not distributed with bartorch, and it is imported inside
:func:`correct_susceptibility` when called.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = ["SusceptibilityCorrection", "correct_susceptibility"]

Optimizer = Literal["gauss-newton", "lbfgs", "admm"]

#: PyHySCO's axis order, phase encoding last, and its inverse, by
#: ``(ndim, phase-encoding axis)``.
_PERMUTATIONS: dict[tuple[int, int], tuple[list[int], list[int]]] = {
    (2, 0): ([1, 0], [1, 0]),
    (2, 1): ([0, 1], [0, 1]),
    (3, 0): ([2, 1, 0], [2, 1, 0]),
    (3, 1): ([2, 0, 1], [1, 2, 0]),
}


@dataclass(frozen=True)
class SusceptibilityCorrection:
    """Result of :func:`correct_susceptibility`, in the input's axis order.

    Parameters
    ----------
    field_map : torch.Tensor
        Estimated displacement field along the phase-encoding axis, defined on
        cell edges: one sample longer than the images along that axis.
    blip_up, blip_down : torch.Tensor
        The two corrected images.
    """

    field_map: torch.Tensor
    blip_up: torch.Tensor
    blip_down: torch.Tensor


def _require_pyhysco() -> dict[str, Any]:
    try:
        from EPI_MRI.EPIMRIDistortionCorrection import (
            DataObject,
            EPIMRIDistortionCorrection,
        )
        from EPI_MRI.ImageModels import Interp1D
        from EPI_MRI.LinearOperators import myLaplacian2D, myLaplacian3D
        from EPI_MRI.utils import m_plus, normalize
        from optimization.ADMM import ADMM
        from optimization.GaussNewton import GaussNewton
        from optimization.LBFGS import LBFGS
    except ImportError as error:
        raise ImportError(
            "susceptibility correction needs PyHySCO, which is GPL-3.0-only and not "
            "distributed with bartorch: pip install 'bartorch[pyhysco]'"
        ) from error
    return {
        "DataObject": DataObject,
        "correction": EPIMRIDistortionCorrection,
        "Interp1D": Interp1D,
        "laplacian": {2: myLaplacian2D, 3: myLaplacian3D},
        "m_plus": m_plus,
        "normalize": normalize,
        "optimizers": {"gauss-newton": GaussNewton, "lbfgs": LBFGS, "admm": ADMM},
    }


def _domain(
    shape: tuple[int, ...],
    voxel_size: tuple[float, ...],
    permutation: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    """PyHySCO's image domain ``omega``, cell counts ``m`` and cell size ``h``."""
    extent = torch.zeros(2 * len(shape), dtype=dtype, device=device)
    extent[1::2] = torch.tensor(
        [size * count for size, count in zip(voxel_size, shape, strict=True)],
        dtype=dtype,
        device=device,
    )
    omega = torch.zeros_like(extent)
    for axis, source in enumerate(permutation):
        omega[2 * axis : 2 * axis + 2] = extent[2 * source : 2 * source + 2]
    counts = torch.tensor(shape, dtype=torch.int, device=device)[permutation]
    spacing = (omega[1::2] - omega[:-1:2]) / counts
    return omega, counts, spacing


def _data_object(
    blip_up: torch.Tensor,
    blip_down: torch.Tensor,
    voxel_size: tuple[float, ...],
    permutation: list[int],
    inverse: list[int],
    parts: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    """PyHySCO's ``DataObject`` populated from tensors.

    ``DataObject.__init__`` loads NIfTI files by path, so the object is created
    without it and given the attributes that constructor sets.
    """
    holder = parts["DataObject"].__new__(parts["DataObject"])
    holder.device = device
    holder.dtype = dtype
    holder.omega, holder.m, holder.h = _domain(
        tuple(blip_up.shape), voxel_size, permutation, dtype=dtype, device=device
    )
    holder.p = inverse
    up = blip_up.to(device=device, dtype=dtype).permute(permutation)
    down = blip_down.to(device=device, dtype=dtype).permute(permutation)
    holder.im1, holder.im2 = up, down
    normalised_up, normalised_down = parts["normalize"](up, down)
    holder.I1 = parts["Interp1D"](normalised_up, holder.omega, holder.m, dtype=dtype, device=device)
    holder.I2 = parts["Interp1D"](
        normalised_down, holder.omega, holder.m, dtype=dtype, device=device
    )
    return holder


def correct_susceptibility(
    blip_up: torch.Tensor,
    blip_down: torch.Tensor,
    *,
    voxel_size: tuple[float, ...],
    phase_encoding_axis: int = 0,
    alpha: float = 300.0,
    beta: float = 1e-4,
    optimizer: Optimizer = "gauss-newton",
    max_iter: int = 10,
    device: torch.device | str | None = None,
) -> SusceptibilityCorrection:
    """Correct susceptibility distortion from a pair with reversed phase encoding.

    The displacement field is PyHySCO's estimate [1]_: the field that brings
    the two images, displaced in opposite directions along the phase-encoding
    axis, into register, under a smoothness penalty and an invertibility
    constraint.  The computation stays in memory and on tensors; nothing is
    read from or written to disk.

    Parameters
    ----------
    blip_up, blip_down : torch.Tensor
        Real images of one shape, 2D or 3D, with opposite phase-encoding
        polarity.
    voxel_size : tuple of float
        Voxel size per axis of the images, in millimetres.
    phase_encoding_axis : int, default=0
        Phase-encoding axis: one of the first two axes of the images.
    alpha : float, default=300.0
        Weight of the smoothness penalty on the field.
    beta : float, default=0.0001
        Weight of the invertibility constraint.
    optimizer : {"gauss-newton", "lbfgs", "admm"}, default='gauss-newton'
        PyHySCO optimizer.
    max_iter : int, default=10
        Optimizer iterations.
    device : torch.device or str, default=None
        Device of the computation, which runs in float64; that of ``blip_up``
        if ``None``.

    Returns
    -------
    SusceptibilityCorrection

    Raises
    ------
    ImportError
        If PyHySCO is not installed.
    ValueError
        If the images differ in shape, ``voxel_size`` does not give one value
        per axis, the phase-encoding axis is not one of the first two, or the
        optimizer is unknown.
    NotImplementedError
        If PyHySCO's two-dimensional regularizer fails on a 2D pair.

    References
    ----------
    .. [1] Julian A, Ruthotto L. PyHySCO: GPU-enabled susceptibility artifact
       distortion correction in seconds. Front Neurosci 2024.
    """
    if blip_up.shape != blip_down.shape:
        raise ValueError(
            f"the pair must have one shape, got {tuple(blip_up.shape)} and {tuple(blip_down.shape)}"
        )
    if len(voxel_size) != blip_up.ndim:
        raise ValueError(f"voxel_size has {len(voxel_size)} entries for a {blip_up.ndim}D image")
    key = (blip_up.ndim, phase_encoding_axis % max(blip_up.ndim, 1))
    if key not in _PERMUTATIONS:
        raise ValueError(
            f"no PyHySCO axis order for a {blip_up.ndim}D image with phase encoding along "
            f"axis {phase_encoding_axis}; images are 2D or 3D and the axis is 0 or 1"
        )
    if optimizer not in ("gauss-newton", "lbfgs", "admm"):
        raise ValueError(f"unknown optimizer {optimizer!r}")
    permutation, inverse = _PERMUTATIONS[key]
    parts = _require_pyhysco()
    resolved = torch.device(device) if device is not None else blip_up.device

    holder = _data_object(
        blip_up,
        blip_down,
        tuple(voxel_size),
        permutation,
        inverse,
        parts,
        dtype=torch.float64,
        device=resolved,
    )
    try:
        objective = parts["correction"](
            holder, alpha, beta, regularizer=parts["laplacian"][blip_up.ndim]
        )
        initial = objective.initialize(blur_result=blip_up.ndim == 3)
    except IndexError as error:
        if blip_up.ndim == 2:
            raise NotImplementedError(
                "PyHySCO cannot correct a two-dimensional pair: its two-dimensional "
                "regularizer indexes a third axis; pass the volume instead"
            ) from error
        raise

    solver = parts["optimizers"][optimizer](objective, max_iter=max_iter, verbose=False, path=None)
    # PyHySCO writes an iteration log file unless pointed elsewhere.
    solver.log.log_file = os.devnull
    solver.run_correction(initial)

    field = solver.Bc.detach().reshape(list(parts["m_plus"](holder.m)))
    shape = list(holder.m)
    return SusceptibilityCorrection(
        field_map=field.permute(holder.p),
        blip_up=objective.corr1.reshape(shape).permute(holder.p),
        blip_down=objective.corr2.reshape(shape).permute(holder.p),
    )
