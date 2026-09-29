"""Voxel-wise uncertainty from repeated reconstructions, calibrated to a coverage."""

from __future__ import annotations

import math

import torch

__all__ = ["calibrate", "moments"]


def moments(reconstruct, samples: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of a randomized reconstruction, voxel by voxel.

    ``reconstruct`` is called ``samples`` times without gradient, and each
    result is folded into running moments (Welford's update), so memory holds
    the mean, the sum of squares and one result whatever ``samples`` is.  The
    spread measures whatever randomness the reconstruction draws: the patch
    grid of a :class:`~bartorch.learning.Patchwise` network with
    ``shift=True``, dropout left active in a network, or the acquired samples
    a reconstruction from a :func:`~bartorch.learning.split` of them keeps.
    Each is a distinct and partial measure of the error, and none is a
    posterior; :func:`calibrate` relates the spread to the error actually
    made on references.

    Parameters
    ----------
    reconstruct : callable
        Called with no argument; returns the reconstructed image, complex or
        real, the same shape every time.
    samples : int, default=8
        Number of reconstructions.

    Returns
    -------
    mean, variance : torch.Tensor
        On the device and with the shape of a reconstruction; the variance is
        real, ``E|x - mean|^2`` with the unbiased ``samples - 1`` divisor.
    """
    if samples < 2:
        raise ValueError(f"a spread takes at least two reconstructions, not {samples}")
    with torch.no_grad():
        mean = reconstruct().clone()
        squares = torch.zeros(mean.shape, dtype=mean.real.dtype, device=mean.device)
        for count in range(2, samples + 1):
            made = reconstruct()
            delta = made - mean
            mean += delta / count
            squares += (delta.conj() * (made - mean)).real
    return mean, squares / (samples - 1)


def calibrate(error: torch.Tensor, spread: torch.Tensor, coverage: float = 0.9) -> float:
    """The factor making ``|error| <= factor * spread`` hold with the given coverage.

    Split conformal calibration: over held-out reconstructions whose
    reference is known, the score ``|error| / spread`` is taken voxel by
    voxel, and the factor is its ``ceil((n + 1) coverage) / n`` empirical
    quantile.  On a new reconstruction from the same distribution,
    ``factor * spread`` is then an interval that contains the error at the
    stated rate, whether or not the spread was the error's standard
    deviation.  The guarantee is marginal, over voxels and subjects drawn
    alike, not voxel by voxel.

    Parameters
    ----------
    error : torch.Tensor
        Reconstruction minus reference, over the calibration voxels: every
        voxel of a few held-out subjects, or a mask of them.
    spread : torch.Tensor
        The same voxels' standard deviation, the square root of the variance
        :func:`moments` returns.
    coverage : float, default=0.9
        Fraction of voxels the interval is to contain.

    Returns
    -------
    float
        The factor; infinite when the calibration set is too small for the
        coverage asked.

    References
    ----------
    Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction.
    Found Trends Mach Learn 2023;16:494-591.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage is strictly between 0 and 1, not {coverage}")
    if error.shape != spread.shape:
        raise ValueError(
            f"an error and a spread per voxel: {tuple(error.shape)} and {tuple(spread.shape)}"
        )
    tiny = torch.finfo(torch.float32).tiny
    scores = (error.abs().double() / spread.double().clamp_min(tiny)).reshape(-1)
    n = scores.numel()
    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        return math.inf
    return float(scores.kthvalue(rank).values)
