"""Normalization of the data before a solve."""

from __future__ import annotations

import torch

from bartorch import _call
from bartorch._dispatch import _ensure_ready, _lock
from bartorch._lib import library
from bartorch._operator import as_operand

_estscaling = _call.build("estscaling", __name__)

__all__ = ["data_scaling"]


def data_scaling(
    y: torch.Tensor,
    *,
    A=None,
    percentile: float | None = None,
    compat: bool = False,
) -> float:
    """Estimate the factor by which BART's reconstructions divide the data.

    Dividing the data by it makes a regularization weight independent of the
    data's overall scale.

    Parameters
    ----------
    y : torch.Tensor
        Data as the solve takes it: for a Cartesian encoding, after the sampling
        pattern and ``fftmod(..., inverse=True)``.  Without ``A`` the coils are
        on their own axis in BART's layout, not squeezed as an operator takes
        them, since the estimate is read from the k-space centre.
    A : LinearOperator, default=None
        Encoding operator.  With one, the estimate is taken from ``|A^H y|``,
        as for a non-Cartesian acquisition; without one, from the k-space
        centre, as BART's ``estscaling`` does.
    percentile : float, default=None
        Percentile of the sorted magnitudes to take in place of BART's rule.
    compat : bool, default=False
        Take the median, BART's legacy estimate.  Only with ``A``.

    Returns
    -------
    float
        The scaling.  Zero when the estimate failed, which ``pics`` replaces
        by one.

    Notes
    -----
    Without ``A`` the estimate is BART's ``estimate_scaling``: the
    root-sum-of-squares image of the fully sampled central region of k-space,
    at most 32 samples along each axis, corrected for the region's size, and the
    90th percentile of its voxel magnitudes -- or their maximum, when the
    maximum exceeds the 90th percentile by at least twice the difference
    between the 90th percentile and the median.  With ``A`` the same rule is
    applied to the magnitudes of ``A^H y``.

    A solve on the divided data returns the image divided by the same factor;
    multiplying by the scaling restores the data's units, as ``pics -S`` and
    :func:`bartorch.apps.pics` do.

    Examples
    --------
    >>> y = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
    >>> scale = optim.data_scaling(y)
    >>> x = optim.FISTA(priors.Wavelet((-1, -2), 0.01))((y / scale).squeeze(1), A)
    """
    if A is None:
        estimate = _estscaling(y, **({} if percentile is None else {"percentile": percentile}))
        return float(estimate.reshape(-1)[0].abs())

    adjoint = A.adjoint(y)
    image = as_operand(adjoint, tuple(adjoint.shape), "the adjoint image")
    _ensure_ready()
    with _lock:
        return float(
            library().bartorch_scaling_norm(
                image.numel(),
                image.data_ptr(),
                1.0,
                int(compat),
                -1.0 if percentile is None else float(percentile),
            )
        )
