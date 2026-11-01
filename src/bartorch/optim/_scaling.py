"""Normalization of the data before a solve."""

from __future__ import annotations

import torch

from bartorch import _call
from bartorch._operator import as_operand

_estscaling = _call.build("estscaling", __name__)

__all__ = ["data_scaling"]


def data_scaling(
    y: torch.Tensor,
    *,
    A=None,
    percentile: float | None = None,
    compat: bool = False,
    adjoint: torch.Tensor | None = None,
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
    adjoint : torch.Tensor, default=None
        ``A^H y`` when the caller has it already, as a solve's start does;
        ``A`` is then not applied again.  Only with ``A``.

    Returns
    -------
    float
        The scaling.  Zero when the estimate failed, which ``pics`` replaces
        by one.

    Notes
    -----
    The order statistics are found by selection (``torch.kthvalue``), on the
    operator's card when it has one, rather than by BART's sort of the whole
    image on the host; they are the same values.

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

    if adjoint is None:
        adjoint = A.adjoint(y)
    image = as_operand(adjoint, tuple(adjoint.shape), "the adjoint image")
    magnitudes = image.reshape(-1).abs()
    card = getattr(A, "device", None)
    if card is not None and torch.device(card).type == "cuda" and magnitudes.device.type == "cpu":
        # Selection over tens of millions of values is seconds on the host
        # and milliseconds on the card the operator already uses.
        magnitudes = magnitudes.to(card)
    return _scaling_norm(magnitudes, compat, percentile)


def _scaling_norm(magnitudes: torch.Tensor, compat: bool, percentile: float | None) -> float:
    """BART's ``estimate_scaling_norm`` over ``magnitudes``, by selection where they lie.

    The same order statistics BART sorts the whole array for -- the median at
    ``n // 2``, the 90th percentile at ``trunc(0.9 n)``, the maximum, or the
    asked-for percentile -- found with ``kthvalue``, on the card when the
    image is there.
    """
    n = magnitudes.numel()
    if n == 0:
        return 0.0
    m = magnitudes.to(torch.float32)

    def at(i: int) -> float:
        return float(torch.kthvalue(m, min(n - 1, max(0, i)) + 1).values)

    if percentile is not None and percentile > 0:
        if percentile > 1:
            raise ValueError(f"the percentile is a fraction, at most one, not {percentile}")
        return at(int(n * percentile))
    median = at(n // 2)
    if compat:
        return median
    p90 = at(int(n * 0.9))
    top = float(m.max())
    return p90 if (top - p90) < 2 * (p90 - median) else top
