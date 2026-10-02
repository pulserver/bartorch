"""Coil calibration: sensitivities, compression, whitening and noise estimates."""

from __future__ import annotations

import torch

from bartorch import _call
from bartorch._call import curated
from bartorch._dispatch import dispatch

__all__ = ["caldir", "ecalib", "nlinv", "whiten"]


@curated("ecalib")
def ecalib(
    kspace: torch.Tensor,
    *,
    maps: int | None = None,
    calib_size: int | tuple[int, ...] | None = None,
    threshold: float | None = None,
    crop: float | None = None,
    kernel_size: int | None = None,
    softsense: bool = False,
    intensity_correction: bool = False,
    return_eigenvalues: bool = False,
    **extra,
):
    """Coil sensitivities by ESPIRiT.

    Parameters
    ----------
    kspace : torch.Tensor
        Fully sampled calibration data, or k-space with a sampled centre,
        ``(coils, z, y, x)``, or ``(coils, y, x)`` for one slice.
    maps : int, default=None
        How many sets of sensitivities to produce (``-m``).
    calib_size : int or tuple of int, default=None
        The calibration region's size (``-r``), the same on every axis or one
        per axis.
    threshold : float, default=None
        The singular-value threshold for the calibration matrix (``-t``).
    crop : float, default=None
        The eigenvalue below which a sensitivity is set to zero (``-c``).
    kernel_size : int, default=None
        The calibration kernel's size (``-k``).
    softsense : bool, default=False
        Return the maps without the eigenvalue crop, for soft-SENSE (``-S``).
    intensity_correction : bool, default=False
        Correct for intensity rather than normalising (``-I``).
    return_eigenvalues : bool, default=False
        Also return the eigenvalue map, which BART writes as a second array
        only when asked.
    **extra
        Further BART ``ecalib`` options, by name.  ``e``, the axis the second
        step is split along, takes an axis of ``kspace``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The sensitivities, and the eigenvalues when asked for, without a ``z``
        axis when ``kspace`` has none.

    Examples
    --------
    >>> maps = ecalib(kspace, maps=1, crop=0.8)
    """
    flags: dict = _call.translate("ecalib", dict(extra), [kspace])
    if maps is not None:
        flags["m"] = maps
    if calib_size is not None:
        flags["r"] = (
            calib_size
            if isinstance(calib_size, int)
            else tuple(int(n) for n in reversed(tuple(calib_size)))
        )
    if threshold is not None:
        flags["t"] = threshold
    if crop is not None:
        flags["c"] = crop
    if kernel_size is not None:
        flags["k"] = kernel_size
    if softsense:
        flags["S"] = True
    if intensity_correction:
        flags["I"] = True
    planar = kspace.ndim == 3
    if planar:
        kspace = kspace.unsqueeze(1)
    found = dispatch("ecalib", [kspace], None, _n_out=2 if return_eigenvalues else 1, **flags)
    if not planar:
        return found
    if return_eigenvalues:
        return tuple(out.squeeze(-3) for out in found)
    return found.squeeze(-3)


@curated("caldir")
def caldir(kspace: torch.Tensor, calib_size: int, **extra) -> torch.Tensor:
    """Coil sensitivities from the centre of k-space directly.

    Parameters
    ----------
    kspace : torch.Tensor
        k-space with a sampled centre.
    calib_size : int
        The size of the calibration region to use.
    **extra
        Further BART ``caldir`` flags, by name.

    Returns
    -------
    torch.Tensor
        The sensitivities.
    """
    return dispatch("caldir", [kspace], None, _pos=[int(calib_size)], **extra)


@curated("nlinv")
def nlinv(
    kspace: torch.Tensor,
    *,
    maxiter: int | None = None,
    maps: int | None = None,
    traj: torch.Tensor | None = None,
    pattern: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    initial: torch.Tensor | None = None,
    alpha: float | None = None,
    real: bool = False,
    normalize: bool = True,
    return_sensitivities: bool = False,
    **extra,
):
    """Nonlinear inversion: the image and the sensitivities together.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space, C order.
    maxiter : int, default=None
        Gauss-Newton steps (``-i``).
    maps : int, default=None
        How many sets of sensitivities to estimate (``-m``).
    traj : tensor, default=None
        Non-Cartesian trajectory (``-t``).
    pattern : tensor, default=None
        Sampling pattern (``-p``).
    basis : tensor, default=None
        Subspace basis (``-B``).
    initial : tensor, default=None
        Warm start (``-I``).
    alpha : float, default=None
        The ``a`` of the Sobolev coil weighting ``(1 + a |k|^2)^(-b/2)``
        (``-a``), not the first step's regularization weight -- that is
        ``nlinv --alpha``, reachable through ``**extra``.
    real : bool, default=False
        Constrain the image to be real (``-c``).
    normalize : bool, default=True
        Divide the image by the root sum of squares of the sensitivities, as
        ``nlinv`` does unless told not to.  BART spells this the
        other way round, as ``-N`` for "do not normalize".
    return_sensitivities : bool, default=False
        Also return the sensitivities, which BART writes as a second array.
    **extra
        Further BART ``nlinv`` options, by name.  ``s``, the axes the
        sensitivities are constant along, takes axes of ``kspace``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The image, and the sensitivities when asked for.
    """
    flags: dict = _call.translate("nlinv", dict(extra), [kspace])
    if maxiter is not None:
        flags["i"] = maxiter
    if maps is not None:
        flags["m"] = maps
    for keyword, value in (("t", traj), ("p", pattern), ("B", basis), ("I", initial)):
        if value is not None:
            flags[keyword] = value
    if alpha is not None:
        flags["a"] = alpha
    if real:
        flags["c"] = True
    if not normalize:
        # BART's -N is "do not normalize", and its own default is to do it.
        flags["N"] = True
    return dispatch("nlinv", [kspace], None, _n_out=2 if return_sensitivities else 1, **flags)


@curated("whiten")
def whiten(
    input: torch.Tensor,
    ndata: torch.Tensor,
    *,
    return_matrix: bool = False,
    return_covariance: bool = False,
    **extra,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Noise prewhitening of multi-channel data with a noise-only measurement.

    Multiplies the channels of ``input`` by :math:`W = L^{-1}`, where :math:`L`
    is the lower triangular Cholesky factor of the channel noise covariance
    :math:`\Psi = L L^H` estimated from ``ndata``, so that
    :math:`W \Psi W^H = I`.

    Parameters
    ----------
    input : torch.Tensor
        Data to whiten, ``(coils, z, y, x)``: the channels are axis ``-4``, and
        one slice has a ``z`` axis of length one.
    ndata : torch.Tensor
        Noise-only measurement of the same channels in the same layout.  Every
        axis but the channel axis indexes noise samples.
    return_matrix : bool, default=False
        Also return the whitening matrix :math:`W`, BART's optional second
        output.
    return_covariance : bool, default=False
        Also return the noise covariance, BART's optional third output.
    **extra
        Further BART ``whiten`` options, by name.  ``o`` and ``c`` take a
        whitening matrix and a noise covariance in the layout returned here, to
        use in place of the ones estimated from ``ndata``; ``n`` normalizes the
        variance to one using ``ndata``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The whitened data, in the shape of ``input``; then, when asked for, the
        whitening matrix and the noise covariance, in that order, each
        ``(coils, coils, 1, 1, 1)``.

    Notes
    -----
    The covariance is :math:`\Psi = (K - 1)^{-1} \sum_k n_k n_k^H` over the
    :math:`K` samples of ``ndata``, :math:`n_k` being the channel vector of the
    :math:`k`-th, with no mean removed, so that ``ndata`` is taken to be
    zero-mean.  The returned matrix is :math:`W`, acting on the channel vector
    from the left, and the returned covariance is the transpose
    :math:`\Psi^T = \overline{\Psi}`.

    Examples
    --------
    >>> white = whiten(data, noise)
    >>> white, matrix = whiten(data, noise, return_matrix=True)
    >>> other = whiten(more_data, noise, o=matrix)
    """
    # BART's outputs are positional: the covariance is the third, so asking for
    # it alone still takes the matrix's slot, which is dropped from the result.
    n_out = 3 if return_covariance else 2 if return_matrix else 1
    found = dispatch("whiten", [input, ndata], None, _n_out=n_out, **extra)
    if n_out == 1:
        return found
    white, matrix, *covariance = found
    return (white, *([matrix] if return_matrix else []), *covariance)


#: Commands in this section without a hand-written wrapper, built from the catalogue.
_DERIVED = (
    "calmat",
    "cc",
    "ccapply",
    "ecaltwo",
    "estvar",
    "ncalib",
    "phasepole",
    "rovir",
    "walsh",
)

for _name in _DERIVED:
    globals()[_name] = _call.build(_name, __name__)
del _name

__all__ = [*__all__, *_DERIVED]
