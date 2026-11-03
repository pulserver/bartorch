"""Coil calibration: sensitivities, compression, whitening and noise estimates."""

from __future__ import annotations

import torch

from bartorch import _call
from bartorch._call import curated
from bartorch._dispatch import dispatch

__all__ = ["caldir", "ecalib", "nlinv", "rovir_fov", "whiten"]


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
    if return_eigenvalues:
        # BART's trailing axes of one come back dropped; the eigenvalues have
        # the sensitivities' shape with one entry along the coils.
        sensitivities, eigenvalues = found
        shape = (*sensitivities.shape[:-4], 1, *sensitivities.shape[-3:])
        found = (sensitivities, eigenvalues.reshape(shape))
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


def rovir_fov(
    kspace: torch.Tensor,
    traj: torch.Tensor,
    coils: int,
    *,
    keep: int | None = None,
    size: int = 48,
    extent: float = 2.0,
    iterations: int = 15,
    return_energy: bool = False,
):
    r"""ROVir coil compression from non-Cartesian samples, rejecting what is past the field of view.

    A grid exactly the size of the field of view, as gridding without
    oversampling (GROG) gives, folds whatever lies past its edge -- the neck
    below a head -- back into the image.  Region-optimized virtual coils
    (ROVir, Kim et al. [1]_) remove it before the reconstruction.  Each
    channel's samples near the centre of k-space are gridded by the adjoint
    of :class:`bartorch.linop.NUFFT`, weighted by
    :func:`bartorch.estimate_density`, onto a ``size`` image covering
    ``extent`` times the field of view on every axis.  The wanted region is
    the field of view and the interference region is the rest of that image:
    geometry only, no mask of the anatomy.  :func:`rovir` solves the
    generalized eigenproblem of their channel correlations and
    orthonormalizes its vectors, so that whitened noise stays white.

    ROVir's directions of highest wanted-to-rejected ratio are often weak
    ones.  So ``keep`` of them are kept, which drops the few that carry the
    interference, and the SVD of the image in them keeps the ``coils`` of
    most energy, as :func:`cc` would.

    Parameters
    ----------
    kspace : torch.Tensor
        ``(channels, *frames, samples)`` whitened samples; every readout
        starts at the centre of k-space, so a prefix of each is near it.
    traj : torch.Tensor
        ``(*frames, samples, ndim)`` in grid units of the field of view,
        ``kx, ky[, kz]``, as for :class:`bartorch.linop.NUFFT`.
    coils : int
        Virtual coils kept.
    keep : int, default=None
        ROVir directions the SVD chooses from; ``None`` keeps ``coils``, which
        is ROVir alone.
    size : int, default=48
        Edge of the low-resolution image the regions are read on.
    extent : float, default=2.0
        Its field of view over the reconstruction's, on every axis.
    iterations : int, default=15
        Pipe-Menon iterations of the density.
    return_energy : bool, default=False
        Also return the share of the field of view's energy the virtual coils
        keep, and their energy outside it over inside it.

    Returns
    -------
    matrix : torch.Tensor
        ``(channels, channels, 1, 1, 1)``, as :func:`ccapply` takes it with
        ``p=coils``: its first ``coils`` rows are the virtual coils.
    energy : tuple of float
        ``(kept, outside_over_inside)``, when ``return_energy``.

    Notes
    -----
    The interference correlation includes the noise of the larger image, so
    noiseless samples make it singular and the eigenproblem fails.

    References
    ----------
    .. [1] Kim D, Cauley SF, Nayak KS, Leahy RM, Haldar JP. Region-optimized
       virtual (ROVir) coils: localization and/or suppression of spatial
       regions using sensor-domain beamforming. Magn Reson Med 2021;86:197-212.

    Examples
    --------
    >>> matrix = rovir_fov(kspace, traj, coils=6, keep=36)
    >>> virtual = ccapply(kspace[..., None], matrix, p=6)
    """
    from bartorch._fourier import estimate_density
    from bartorch.linop import NUFFT
    from bartorch.tools import rovir  # built from the catalogue below

    channels, ndim = kspace.shape[0], traj.shape[-1]
    k = traj.to(torch.float32) * extent  # cycles per field of view -> per the larger one
    near = k.abs().amax(-1) < size / 2 - 0.5
    samples = near.shape[-1]
    first = int(near.reshape(-1, samples).sum(-1).max())  # readouts run centre-out
    k = k[..., :first, :].reshape(-1, first, ndim)
    k = k.clamp(-size / 2 + 0.5, size / 2 - 0.5).contiguous()
    near = near[..., :first].reshape(-1, first)
    y = kspace[..., :first].reshape(channels, -1, first)
    grid = (size,) * ndim
    # One trajectory: every readout shares the density, as one image is gridded from them all.
    density = estimate_density(k.reshape(-1, ndim), grid, iterations=iterations).reshape(near.shape)
    weights = (density * near).sqrt().contiguous()
    nufft = NUFFT(k, (channels, *grid), weights=weights, toeplitz=False)
    images = nufft.adjoint((y * weights).to(torch.complex64).contiguous()).to(torch.complex64)

    inner = round(size / extent)
    lo = (size - inner) // 2
    wanted = torch.zeros(grid, dtype=torch.bool)
    wanted[(slice(lo, lo + inner),) * ndim] = True
    transform = rovir((images * wanted).contiguous(), (images * ~wanted).contiguous())
    # ccapply's virtual coil j is sum_c conj(transform[j, c]) x_c: its rows, as columns here.
    q = transform.reshape(channels, channels).mT.to(torch.complex128)[:, : keep or coils]
    x = images.reshape(channels, -1).to(torch.complex128)
    u = torch.linalg.svd(q.conj().mT @ x, full_matrices=False)[0][:, :coils]
    w = q @ u
    matrix = torch.zeros(channels, channels, 1, 1, 1, dtype=torch.complex64)
    matrix[:coils, :, 0, 0, 0] = w.mT.to(torch.complex64)
    if not return_energy:
        return matrix
    inside = wanted.flatten()
    v = w.conj().mT @ x
    kept = float(v[:, inside].abs().pow(2).sum() / x[:, inside].abs().pow(2).sum())
    ratio = float(v[:, ~inside].abs().pow(2).sum() / v[:, inside].abs().pow(2).sum())
    return matrix, (kept, ratio)


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
