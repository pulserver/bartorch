"""Self-calibrated wavelet-regularized SENSE: ``nlinv`` sensitivities, then ``pics``."""

from __future__ import annotations

import numpy as np
import torch

from bartorch import apps, priors, tools

__all__ = ["nlinv_pics"]


def _normalized(maps: torch.Tensor) -> torch.Tensor:
    return maps / maps.abs().square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def _partial_fourier(sampled: np.ndarray, axis: int) -> tuple[float, bool]:
    """The acquired fraction of ``axis`` and whether its high indices are the acquired side.

    ``(1.0, False)`` unless the lines missing at one end outnumber the widest
    gap between sampled lines, which undersampling alone leaves.
    """
    profile = sampled.any(axis=tuple(a for a in range(sampled.ndim) if a != axis))
    lines = np.flatnonzero(profile)
    n = profile.size
    if lines.size < 2:
        return 1.0, False
    gap = int(np.diff(lines).max()) - 1
    low, high = int(lines[0]), n - 1 - int(lines[-1])
    missing = max(low, high)
    if missing <= gap:
        return 1.0, False
    return (n - missing) / n, low > high


def _cartesian_maps(volume: torch.Tensor, size: int) -> torch.Tensor:
    """Sensitivities of ``(coils, z, y, x)`` k-space, fitted to its central ``size`` lines."""
    if volume.shape[0] == 1:
        return torch.ones_like(volume)
    centre = tuple(
        slice((n - min(size, n)) // 2, (n + min(size, n)) // 2) for n in volume.shape[1:]
    )
    low = torch.zeros_like(volume)
    low[(slice(None), *centre)] = volume[(slice(None), *centre)]
    _, maps = tools.nlinv(low, maps=1, return_sensitivities=True)
    return _normalized(maps.reshape(volume.shape))


def _radial_maps(kspace: torch.Tensor, traj: torch.Tensor, radius: float) -> torch.Tensor:
    """Sensitivities of non-Cartesian k-space, fitted to the samples within ``radius``."""
    if kspace.shape[0] == 1:
        size = [int(n) for n in tools.estdims(traj.real.cpu()).split()]
        return torch.ones((1, size[1], size[0]), dtype=kspace.dtype, device=kspace.device)
    centre = traj.real.square().sum(dim=-1).sqrt() <= radius
    _, maps = tools.nlinv(
        (kspace * centre)[..., None], traj=traj, maps=1, return_sensitivities=True
    )
    return _normalized(maps.reshape(kspace.shape[0], *maps.shape[-2:]))


def nlinv_pics(
    kspace: torch.Tensor,
    traj: torch.Tensor | None = None,
    *,
    wavelet: float = 0.005,
    iterations: int = 30,
    size: int = 24,
    radius: float = 12.0,
) -> torch.Tensor:
    """Wavelet-regularized SENSE reconstruction with coil sensitivities fitted to the data.

    The sensitivities are one set from ``bart nlinv -m 1`` on the
    low-resolution centre of k-space, normalised to unit root sum of squares
    (unit sensitivity for one coil). The image minimises
    ``|P F S x - y|^2 + lambda |W x|_1`` (``bart pics -R W``).

    A Cartesian axis sampled on one side only, by more lines than the widest
    gap between sampled lines, is partial Fourier, and the image is completed
    along it by ``bart homodyne -I -C``. Unsampled k-space is zero.

    Parameters
    ----------
    kspace : torch.Tensor
        Complex k-space. Cartesian: ``(coils, [z,] y, x)`` on the
        reconstruction grid. Non-Cartesian: ``(coils, shots, samples)``.
    traj : torch.Tensor, default=None
        Trajectory ``(shots, samples, 3)`` in units of the image grid; ``None``
        for Cartesian data.
    wavelet : float, default=0.005
        ``lambda``, relative to the data scaling ``pics`` estimates.
    iterations : int, default=30
        Iterations of the solve.
    size : int, default=24
        Cartesian: lines of every encoded axis, around the centre, the
        sensitivities are fitted to.
    radius : float, default=12.0
        Non-Cartesian: distance from the k-space centre, in grid units, within
        which samples are fitted.

    Returns
    -------
    torch.Tensor
        Complex image: ``([z,] y, x)`` as ``kspace`` for Cartesian data, the
        trajectory's image grid for non-Cartesian data.
    """
    kspace = kspace.to(torch.complex64)
    if traj is not None:
        maps = _radial_maps(kspace, traj, radius)
        image = apps.pics(
            kspace,
            maps,
            traj=traj,
            regularizers=priors.Wavelet((-1, -2), wavelet),
            maxiter=iterations,
            eigen_step=True,
        )
        return image.reshape(maps.shape[-2:])

    flat = kspace.ndim == 3
    volume = kspace[:, None] if flat else kspace
    sampled = volume.abs().sum(dim=0).cpu().numpy() > 0
    maps = _cartesian_maps(volume, size)
    encoded = (-1, -2) if volume.shape[1] == 1 else (-1, -2, -3)
    image = apps.pics(
        volume,
        maps,
        regularizers=priors.Wavelet(encoded, wavelet),
        maxiter=iterations,
    ).reshape(volume.shape[1:])
    for axis in range(image.ndim):
        acquired, high = _partial_fourier(sampled, axis)
        if acquired == 1.0:
            continue
        # bart homodyne takes the acquired side at the low indices.
        if high:
            image = torch.flip(image, (axis,))
        image = tools.homodyne(axis, acquired, image.contiguous(), I=True, C=True).reshape(
            image.shape
        )
        if high:
            image = torch.flip(image, (axis,))
    return image[0] if flat else image
