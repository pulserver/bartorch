"""Coil sensitivity maps fitted by ``nlinv`` to the centre of k-space."""

from __future__ import annotations

import torch

from bartorch import tools

__all__ = ["nlinv_maps"]


def _normalized(maps: torch.Tensor) -> torch.Tensor:
    return maps / maps.abs().square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def _cartesian_maps(volume: torch.Tensor, size: int, **options) -> torch.Tensor:
    """Sensitivities of ``(coils, z, y, x)`` k-space, fitted to its central ``size`` lines."""
    if volume.shape[0] == 1:
        return torch.ones_like(volume)
    centre = tuple(
        slice((n - min(size, n)) // 2, (n + min(size, n)) // 2) for n in volume.shape[1:]
    )
    low = torch.zeros_like(volume)
    low[(slice(None), *centre)] = volume[(slice(None), *centre)]
    _, maps = tools.nlinv(low, maps=1, return_sensitivities=True, **options)
    return _normalized(maps.reshape(volume.shape))


def _radial_maps(
    kspace: torch.Tensor, traj: torch.Tensor, radius: float, **options
) -> torch.Tensor:
    """Sensitivities of non-Cartesian k-space, fitted to the samples within ``radius``."""
    if kspace.shape[0] == 1:
        size = [int(n) for n in tools.estdims(traj.real.cpu()).split()]
        return torch.ones((1, size[1], size[0]), dtype=kspace.dtype, device=kspace.device)
    centre = traj.real.square().sum(dim=-1).sqrt() <= radius
    _, maps = tools.nlinv(
        (kspace * centre)[..., None], traj=traj, maps=1, return_sensitivities=True, **options
    )
    return _normalized(maps.reshape(kspace.shape[0], *maps.shape[-2:]))


def nlinv_maps(
    kspace: torch.Tensor,
    traj: torch.Tensor | None = None,
    *,
    size: int = 24,
    radius: float = 12.0,
    phase_pole: int | None = None,
) -> torch.Tensor:
    """Coil sensitivity maps fitted to the low-resolution centre of k-space.

    One set of maps from ``bart nlinv -m 1``, normalized to unit root sum of
    squares over the coils (unit sensitivity for one coil).

    Parameters
    ----------
    kspace : torch.Tensor
        Complex k-space. Cartesian: ``(coils, [z,] y, x)`` on the
        reconstruction grid. Non-Cartesian: ``(coils, shots, samples)``.
    traj : torch.Tensor, default=None
        Trajectory ``(shots, samples, 3)`` in units of the image grid; ``None``
        for Cartesian data.
    size : int, default=24
        Cartesian: lines of every encoded axis, around the centre, the maps
        are fitted to.
    radius : float, default=12.0
        Non-Cartesian: distance from the k-space centre, in grid units, within
        which samples are fitted.
    phase_pole : int, default=None
        Gauss-Newton step after which the phase poles the maps share are
        detected and moved into the image (``nlinv --phase-pole``); 0 after
        every step, ``None`` never. At a shared phase pole the smoothness of
        the maps drives every coil's sensitivity towards zero, and a coil
        combination over them goes dark there. BART refuses the detection on
        a volume with an axis no longer than a twentieth of its longest,
        rounded up.

    Returns
    -------
    torch.Tensor
        Complex maps: of the shape of ``kspace`` for Cartesian data,
        ``(coils, y, x)`` on the image grid of the trajectory otherwise.

    Raises
    ------
    ~bartorch.BartError
        If ``phase_pole`` is set for a volume too thin for BART's detection.
    """
    kspace = kspace.to(torch.complex64)
    options = {} if phase_pole is None else {"phase_pole": int(phase_pole)}
    if traj is not None:
        return _radial_maps(kspace, traj, radius, **options)
    flat = kspace.ndim == 3
    maps = _cartesian_maps(kspace[:, None] if flat else kspace, size, **options)
    return maps[:, 0] if flat else maps
