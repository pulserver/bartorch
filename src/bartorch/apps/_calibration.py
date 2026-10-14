"""Coil sensitivity maps fitted by ``nlinv`` to the centre of k-space."""

from __future__ import annotations

import math

import torch

import bartorch
from bartorch import tools

__all__ = ["nlinv_maps"]


def _normalized(maps: torch.Tensor) -> torch.Tensor:
    return maps / maps.abs().square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def _centre(n: int, m: int) -> slice:
    """The ``m`` points of an axis of ``n`` whose centre, ``m // 2``, is the axis's, ``n // 2``."""
    return slice(n // 2 - m // 2, n // 2 - m // 2 + m)


#: ``a`` of BART's Sobolev weighting ``(1 + a |k / n|^2)^(-b/2)`` of the
#: sensitivities, ``k / n`` the frequency relative to the grid they are on.
SOBOLEV_A = 220.0


def _upsampled(maps: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """``(coils, *grid)`` maps zero-filled in k-space to ``(coils, *shape)``, normalized."""
    axes = tuple(-i for i in range(1, len(shape) + 1) if shape[-i] > 1)
    centre = (slice(None), *(_centre(n, m) for n, m in zip(shape, maps.shape[1:])))
    spectrum = maps.new_zeros((maps.shape[0], *shape))
    spectrum[centre] = bartorch.fft(maps, axes)
    return _normalized(bartorch.fft(spectrum, axes, inverse=True))


def _scaled_sobolev(grid: list[int], shape: tuple[int, ...], options: dict) -> dict:
    """``options`` with the Sobolev ``a`` scaled from ``shape`` to the smaller ``grid``.

    By the square of the ratio of the two along the shortest cropped axis, so
    that the weighting is, there, the one ``shape`` gives.
    """
    cropped = [m / n for n, m in zip(shape, grid) if m < n]
    if not cropped:
        return options
    return {"alpha": SOBOLEV_A * max(cropped) ** 2, **options}


def _cartesian_maps(volume: torch.Tensor, size: int, **options) -> torch.Tensor:
    """Sensitivities of ``(coils, z, y, x)`` k-space, fitted to its central ``size`` lines.

    ``nlinv`` runs on the grid of the central lines alone, and the maps are
    brought to the grid of ``volume`` by zero-filling their k-space.
    """
    if volume.shape[0] == 1:
        return torch.ones_like(volume)
    shape = tuple(volume.shape[1:])
    grid = [min(size, n) for n in shape]
    centre = (slice(None), *(_centre(n, m) for n, m in zip(shape, grid)))
    low = volume[centre].contiguous()
    options = _scaled_sobolev(grid, shape, options)
    _, maps = tools.nlinv(low, maps=1, return_sensitivities=True, **options)
    return _upsampled(maps.reshape(low.shape), shape)


def _radial_maps(
    kspace: torch.Tensor,
    traj: torch.Tensor,
    radius: float,
    shape: tuple[int, ...] | None,
    **options,
) -> torch.Tensor:
    """Sensitivities of non-Cartesian k-space, fitted to the samples within ``radius``.

    ``nlinv -t`` runs on those samples alone, on a grid of ``2 ceil(radius)``
    points an axis, and the maps are brought to ``shape``, by default the
    grid BART estimates from the whole trajectory, by zero-filling their
    k-space.
    """
    if shape is None:
        x, y, z = (int(n) for n in tools.estdims(traj.real.cpu()).split())
        shape = (y, x) if z == 1 else (z, y, x)
    shape = tuple(shape)
    if kspace.shape[0] == 1:
        return torch.ones((1, *shape), dtype=kspace.dtype, device=kspace.device)
    grid = [min(2 * math.ceil(radius), n) for n in shape]
    centre = traj.real.square().sum(dim=-1).sqrt() <= radius
    samples = kspace[:, centre][:, None, :, None]
    options = _scaled_sobolev(grid, shape, options)
    _, maps = tools.nlinv(
        samples,
        traj=traj[centre][None],
        maps=1,
        return_sensitivities=True,
        x=(*grid[::-1], 1)[:3],
        **options,
    )
    return _upsampled(maps.reshape(kspace.shape[0], *grid), shape)


def nlinv_maps(
    kspace: torch.Tensor,
    traj: torch.Tensor | None = None,
    *,
    size: int = 24,
    radius: float = 12.0,
    shape: tuple[int, ...] | None = None,
    phase_pole: int | None = None,
) -> torch.Tensor:
    """Coil sensitivity maps fitted to the low-resolution centre of k-space.

    One set of maps from ``bart nlinv -m 1``, normalized to unit root sum of
    squares over the coils (unit sensitivity for one coil). The maps are
    fitted on a grid of the low-resolution centre alone, with the Sobolev
    weighting of the sensitivities scaled to that grid, and zero-filled in
    k-space to the image grid.

    Parameters
    ----------
    kspace : torch.Tensor
        Complex k-space. Cartesian: ``(coils, [z,] y, x)`` on the
        reconstruction grid. Non-Cartesian: ``(coils, shots, samples)``.
    traj : torch.Tensor, default=None
        Trajectory ``(shots, samples, 3)`` in units of the image grid; ``None``
        for Cartesian data. Its kz may be nonzero.
    size : int, default=24
        Cartesian: lines of every encoded axis, around the centre, the maps
        are fitted to.
    radius : float, default=12.0
        Non-Cartesian: distance from the k-space centre, in grid units, within
        which samples are fitted, on a grid of ``2 ceil(radius)`` points an
        axis.
    shape : tuple of int, default=None
        Non-Cartesian: image grid of the maps, ``(y, x)`` or ``(z, y, x)``;
        ``None`` for the one BART estimates from ``traj``.
    phase_pole : int, default=None
        Gauss-Newton step after which the phase poles the maps share are
        detected and moved into the image (``nlinv --phase-pole``); 0 after
        every step, ``None`` never. At a shared phase pole the smoothness of
        the maps drives every coil's sensitivity towards zero, and a coil
        combination over them goes dark there. BART refuses the detection on
        a grid with an axis no longer than a twentieth of its longest, rounded
        up: the grid of the central lines, so a volume of fewer than 3
        partitions at the default ``size``.

    Returns
    -------
    torch.Tensor
        Complex maps: of the shape of ``kspace`` for Cartesian data,
        ``(coils, *shape)`` otherwise.

    Raises
    ------
    ~bartorch.BartError
        If ``phase_pole`` is set for a grid too thin for BART's detection.
    """
    kspace = kspace.to(torch.complex64)
    options = {} if phase_pole is None else {"phase_pole": int(phase_pole)}
    if traj is not None:
        return _radial_maps(kspace, traj, radius, shape, **options)
    flat = kspace.ndim == 3
    maps = _cartesian_maps(kspace[:, None] if flat else kspace, size, **options)
    return maps[:, 0] if flat else maps
