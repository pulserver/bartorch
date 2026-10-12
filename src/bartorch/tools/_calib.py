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
        k-space with a fully sampled centre, ``(coils, z, y, x)``, or
        ``(coils, y, x)`` for one slice.
    maps : int, default=None
        Number of sets of sensitivities (``-m``); BART's default is 2.
    calib_size : int or tuple of int, default=None
        Size of the calibration region (``-r``), one for every axis or one per
        axis in C order; BART's default is 24 on every axis.
    threshold : float, default=None
        Singular-value threshold of the calibration matrix, which sets the
        size of its null space (``-t``); BART's default is 0.001.
    crop : float, default=None
        Eigenvalue below which a sensitivity is set to zero (``-c``); BART's
        default is 0.8.
    kernel_size : int, default=None
        Size of the calibration kernel (``-k``); BART's default is 6.
    softsense : bool, default=False
        Weight the maps by a smooth function of the eigenvalue about ``crop``
        rather than cropping them (Soft-SENSE, ``-S``).
    intensity_correction : bool, default=False
        Normalise the maps to unit l1 norm over the coils, times the square
        root of the coil count, rather than unit l2 norm (``-I``).
    return_eigenvalues : bool, default=False
        Also return the eigenvalue maps, BART's optional second output.
    **extra
        Further BART ``ecalib`` options, by name.  ``e``, the axis the second
        step is split along, takes an axis of ``kspace``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The sensitivities, ``(maps, coils, z, y, x)`` with the ``maps`` axis
        dropped when it has size one, and the eigenvalue maps when asked for;
        without a ``z`` axis when ``kspace`` has none.

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
    """Coil sensitivities by direct calibration from the centre of k-space.

    Parameters
    ----------
    kspace : torch.Tensor
        k-space with a fully sampled centre, ``(coils, z, y, x)``.
    calib_size : int
        Size of the calibration region, the same on every axis.
    **extra
        Further BART ``caldir`` options, by name.

    Returns
    -------
    torch.Tensor
        The sensitivities, in the shape of ``kspace``.
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
    r"""Image and coil sensitivities estimated jointly by nonlinear inversion.

    Parameters
    ----------
    kspace : torch.Tensor
        Undersampled k-space, ``(..., coils, z, y, x)``.
    maxiter : int, default=None
        Number of Gauss-Newton steps (``-i``); BART's default is 8.
    maps : int, default=None
        Number of sets of sensitivities, ENLIVE (``-m``).
    traj : torch.Tensor, default=None
        Non-Cartesian trajectory ``(..., samples, 3)`` in grid units (``-t``).
    pattern : torch.Tensor, default=None
        Sampling pattern (``-p``).
    basis : torch.Tensor, default=None
        Temporal subspace basis (``-B``).
    initial : torch.Tensor, default=None
        Initial image and sensitivities (``-I``).
    alpha : float, default=None
        ``a`` of the Sobolev weighting ``(1 + a |k|^2)^(-b/2)`` of the
        sensitivities (``-a``).  The regularization weight of the first
        Gauss-Newton step is ``nlinv --alpha``, passed through ``**extra``.
    real : bool, default=False
        Constrain the image to be real (``-c``).
    normalize : bool, default=True
        Multiply the image by the root sum of squares of the sensitivities
        over the coils (see Notes); ``False`` is BART's ``-N``.
    return_sensitivities : bool, default=False
        Also return the sensitivities, BART's optional second output.
    **extra
        Further BART ``nlinv`` options, by name.  ``s``, the axes the
        sensitivities are constant along, takes axes of ``kspace``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The image, without the coil axis and with the sets of ``maps``
        combined, and the sensitivities when asked for.

    Notes
    -----
    With ``normalize=True`` and one set of sensitivities, the returned image is
    :math:`\rho \sqrt{\sum_c |s_c|^2}`: the magnitude of the coil images
    :math:`\rho s_c` combined by root sum of squares, with the phase of
    :math:`\rho`.  With several sets combined it is
    :math:`\sqrt{\sum_c |\sum_m \rho_m s_{m,c}|^2}`, which is real.
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
