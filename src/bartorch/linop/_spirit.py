"""The SPIRiT self-consistency of multichannel k-space as a linear operator."""

from __future__ import annotations

import torch

from bartorch.linop._base import LinearOperator
from bartorch.linop._basic import FFT, Identity
from bartorch.linop._shape import Reshape
from bartorch.linop._signal import Matrix

__all__ = ["spirit_consistency"]


def spirit_consistency(kernel: torch.Tensor, grid: tuple[int, ...]) -> LinearOperator:
    """Return ``G - I`` on ``(coils, *grid)`` centred, unitary k-space: the SPIRiT inconsistency.

    ``G`` applies a calibrated SPIRiT kernel, cyclically, to every sample.
    ``|(G - I) k|^2`` is small for k-space that agrees with the calibration,
    so it serves as a quadratic penalty in any solver that takes one: a
    :class:`~bartorch.optim.Tikhonov` term of conjugate gradients on a SENSE
    encoding (composed with the encoding's coil model and Fourier transform),
    or the data-consistency subproblem of ADMM or a plug-and-play scheme.
    ``G`` is applied in the image domain, as a coil mixing per voxel.

    Parameters
    ----------
    kernel : torch.Tensor
        ``(coils, coils, *size)``, from :func:`bartorch.tools.spirit_kernel`.
    grid : tuple of int
        The k-space grid.

    Returns
    -------
    LinearOperator
        From and to ``(coils, *grid)``.
    """
    from bartorch.tools import spirit_weights

    grid = tuple(grid)
    coils = kernel.shape[0]
    shape = (coils, *grid)
    axes = tuple(range(1, len(shape)))
    weights = spirit_weights(kernel, grid)
    mix = (
        Reshape(shape, (coils, 1, *grid))
        @ Matrix(weights, (coils, 1, *grid), (1, coils, *grid))
        @ Reshape((1, coils, *grid), shape)
    )
    return FFT(shape, axes) @ mix @ FFT(shape, axes, inverse=True) - Identity(shape)
