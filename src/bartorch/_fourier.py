"""Fourier transforms along C-order axes."""

from __future__ import annotations

import math

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = [
    "estimate_density",
    "fft",
    "fftmod",
    "fftshift",
    "ifft",
    "nufft",
    "nufft_adjoint",
]


@curated("fft")
def fft(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    inverse: bool = False,
    unitary: bool = False,
    uncentred: bool = False,
) -> torch.Tensor:
    """Fourier transform along ``axes``, centred by default.

    Parameters
    ----------
    input : torch.Tensor
    axes : int or tuple of int
        Axes to transform, as indices into ``input.shape``.
    inverse : bool, default=False
        Transform with the positive exponent.
    unitary : bool, default=False
        Scale by one over the square root of the transformed size;
        otherwise unnormalized.
    uncentred : bool, default=False
        Keep the zero frequency at index zero rather than at ``n // 2``.

    Examples
    --------
    >>> spectrum = fft(image, axes=(-2, -1), unitary=True)
    """
    return dispatch(
        "fft",
        [input],
        None,
        _pos=[axes_flags(axes, input.ndim)],
        i=inverse,
        u=unitary,
        n=uncentred,
    )


def ifft(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    unitary: bool = False,
    uncentred: bool = False,
) -> torch.Tensor:
    """Inverse Fourier transform along ``axes``: :func:`fft` with ``inverse=True``."""
    return fft(input, axes, inverse=True, unitary=unitary, uncentred=uncentred)


@curated("fftmod")
def fftmod(
    input: torch.Tensor, axes: int | tuple[int, ...], *, inverse: bool = False
) -> torch.Tensor:
    """Multiply by the alternating phase that centres an uncentred transform, along ``axes``."""
    return dispatch("fftmod", [input], None, _pos=[axes_flags(axes, input.ndim)], i=inverse)


@curated("fftshift")
def fftshift(
    input: torch.Tensor, axes: int | tuple[int, ...], *, inverse: bool = False
) -> torch.Tensor:
    """Move the zero frequency to the middle along ``axes``, or back with ``inverse``."""
    return dispatch("fftshift", [input], None, _pos=[axes_flags(axes, input.ndim)], i=inverse)


@curated("nufft")
def nufft(
    input: torch.Tensor,
    traj: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    **extra,
) -> torch.Tensor:
    """Non-uniform Fourier transform of an image to samples along ``traj``.

    Computed by FINUFFT, with a negative exponent and scaled by one over the
    square root of the image's voxel count.

    Parameters
    ----------
    input : torch.Tensor
        Image, C order ``(..., z, y, x)``; ``z`` is one for a two-dimensional
        trajectory.
    traj : torch.Tensor
        Trajectory ``(..., samples, 3)`` in grid units, ``kx, ky, kz``, as
        :func:`bartorch.tools.traj` produces.
    weights : torch.Tensor, default=None
        Diagonal in k-space applied to the samples.
    basis : torch.Tensor, default=None
        Temporal subspace basis, coefficients then frames, in BART's axis
        order: ``(coeffs, frames, 1, 1, 1, 1, 1)``.
    **extra
        Further ``bart nufft`` options, by name.
    """
    flags: dict = dict(extra)
    if weights is not None:
        flags["p"] = weights
    if basis is not None:
        flags["B"] = basis
    return dispatch("nufft", [traj, input], None, **flags)


@curated("nufft")
def nufft_adjoint(
    input: torch.Tensor,
    traj: torch.Tensor,
    image_shape: tuple[int, ...] | None = None,
    *,
    weights: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    **extra,
) -> torch.Tensor:
    """Adjoint of :func:`nufft`: samples along ``traj`` back to an image.

    Parameters
    ----------
    input : torch.Tensor
        Samples, as :func:`nufft` returns them.
    traj : torch.Tensor
        Trajectory ``(..., samples, 3)`` in grid units, ``kx, ky, kz``, as
        :func:`bartorch.tools.traj` produces.
    image_shape : tuple of int, default=None
        Spatial shape of the image, C order ``(z, y, x)`` or ``(y, x)``.  By
        default BART estimates it from the trajectory.
    weights : torch.Tensor, default=None
        Diagonal in k-space applied to the samples; its conjugate is applied
        here.
    basis : torch.Tensor, default=None
        Temporal subspace basis, coefficients then frames, in BART's axis
        order: ``(coeffs, frames, 1, 1, 1, 1, 1)``.
    **extra
        Further ``bart nufft`` options, by name.
    """
    flags: dict = dict(extra)
    flags["a"] = True
    if image_shape is not None:
        spatial = [int(n) for n in reversed(tuple(image_shape))]
        flags["d"] = tuple((spatial + [1, 1, 1])[:3])
    if weights is not None:
        flags["p"] = weights
    if basis is not None:
        flags["B"] = basis
    return dispatch("nufft", [traj, input], None, **flags)


def _check_trajectory(traj: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """``traj`` as a real tensor whose components lie within the grid of ``shape``.

    Component ``i`` pairs with axis ``-1 - i`` of the image; a two-dimensional
    grid accepts a third component only when it is zero.
    """
    if len(shape) not in (2, 3):
        raise ValueError(f"the grid is two- or three-dimensional, got {shape}")
    if traj.ndim < 2 or traj.shape[-1] not in (2, 3):
        raise ValueError(f"a trajectory is (..., samples, 2 or 3), got {tuple(traj.shape)}")
    if traj.shape[-1] < len(shape):
        raise ValueError(f"a {len(shape)}D grid needs {len(shape)} trajectory components")
    if traj.shape[-1] > len(shape):
        if bool(torch.any(traj[..., 2] != 0)):
            raise ValueError("a two-dimensional grid takes no kz")
        traj = traj[..., :2]
    edge = torch.tensor([n / 2 for n in reversed(shape)], dtype=traj.dtype, device=traj.device)
    if bool(torch.any(traj.abs() > edge * (1 + 1e-6))):
        raise ValueError(
            f"the trajectory is in grid units and must lie within +-n/2 of the grid {shape}"
        )
    return traj


def _fejer_window(shape: tuple[int, ...], device) -> torch.Tensor:
    """Triangle ``(n - |r|) / n`` per axis of the doubled grid, about index ``n``.

    Its transform is the squared Dirichlet kernel of the grid ``shape``: the
    point spread function of that field of view, squared, and so non-negative.
    """
    window = torch.ones((), dtype=torch.float32, device=device)
    for axis, n in enumerate(shape):
        r = torch.arange(2 * n, dtype=torch.float32, device=device) - n
        view = [1] * len(shape)
        view[axis] = 2 * n
        window = window * ((n - r.abs()).clamp(min=0) / n).reshape(view)
    return window


def estimate_density(
    traj: torch.Tensor,
    shape: tuple[int, ...],
    *,
    iterations: int = 20,
) -> torch.Tensor:
    r"""Pipe-Menon density compensation weights for a trajectory.

    The fixed point :math:`w \leftarrow w / (C * w)` of Pipe and Menon [1]_
    is iterated from :math:`w = 1`, with the convolution :math:`C * w`
    evaluated at the samples by the non-uniform transforms of this package
    (:class:`bartorch.linop.NUFFT`).  The kernel :math:`C` is the squared
    Dirichlet kernel of the grid ``shape``, the squared point spread function
    of its field of view: a non-negative kernel one grid cell wide.  It is
    applied as a triangle window on a grid of twice ``shape``.  The weights
    are then scaled so that the density-compensated adjoint of the transform of
    a uniform image has unit mean modulus.

    Parameters
    ----------
    traj : torch.Tensor
        Trajectory ``(*frames, samples, ndim)`` in grid units of ``shape``,
        ``kx, ky[, kz]`` with ``kx`` along the last image axis, as for
        :func:`nufft`.  Each frame is a trajectory of its own and receives its
        own weights.
    shape : tuple of int
        Image grid, ``(y, x)`` or ``(z, y, x)``, relative to which the density
        is estimated.
    iterations : int, default=20
        Fixed-point iterations.

    Returns
    -------
    torch.Tensor
        Real, non-negative weights of shape ``traj.shape[:-1]``, for the
        ``weights`` of :func:`nufft_adjoint` or :class:`bartorch.linop.NUFFT`,
        or the ``density`` of :func:`bartorch.tools.reconstruct_navigator`.

    Raises
    ------
    ValueError
        If the trajectory does not match the grid's dimensionality or leaves
        :math:`\pm n/2` along an axis.

    References
    ----------
    .. [1] Pipe JG, Menon P. Sampling density compensation in MRI: rationale
       and an iterative numerical solution. Magn Reson Med 1999;41:179-186.

    Examples
    --------
    >>> weights = estimate_density(traj, (64, 64))  # traj (spokes * samples, 2)
    """
    from bartorch.linop import NUFFT

    shape = tuple(int(n) for n in shape)
    traj = _check_trajectory(torch.as_tensor(traj).to(torch.float32), shape)
    frames, samples = tuple(traj.shape[:-2]), int(traj.shape[-2])
    count = math.prod(frames)
    flat = traj.reshape(count, 1, samples, traj.shape[-1])

    doubled = NUFFT(2 * flat, (count, *(2 * n for n in shape)), toeplitz=False)
    window = _fejer_window(shape, traj.device)
    weights = torch.ones((count, 1, samples), dtype=torch.complex64, device=traj.device)
    for _ in range(int(iterations)):
        convolved = doubled(window * doubled.H(weights)).abs()
        weights = weights / convolved.clamp(min=torch.finfo(torch.float32).tiny)

    grid = NUFFT(flat, (count, *shape), toeplitz=False)
    uniform = torch.ones((count, *shape), dtype=torch.complex64, device=traj.device)
    response = grid.H(weights * grid(uniform)).abs()
    scale = response.reshape(count, -1).mean(dim=1).reshape(count, 1, 1)
    return (weights.abs() / scale).reshape(*frames, samples)
