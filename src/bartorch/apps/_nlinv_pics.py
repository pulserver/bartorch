"""Self-calibrated wavelet-regularized SENSE: ``nlinv_maps``, ``pics`` and ``partial_fourier``."""

from __future__ import annotations

import torch

from bartorch import apps, priors

__all__ = ["nlinv_pics"]


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

    The sensitivities are those of :func:`nlinv_maps`. The image is the
    :func:`pics` minimizer of ``|P F S x - y|^2 + lambda |W x|_1``
    (``bart pics -R W``). For Cartesian data, an axis acquired on one side
    only is then completed by :func:`partial_fourier`. Unsampled k-space is
    zero: the sampling mask is the support of ``kspace``.

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
    maps = apps.nlinv_maps(kspace, traj, size=size, radius=radius)
    if traj is not None:
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
    sampled = volume.abs().sum(dim=0) > 0
    encoded = (-1, -2) if volume.shape[1] == 1 else (-1, -2, -3)
    image = apps.pics(
        volume,
        maps.reshape(volume.shape),
        regularizers=priors.Wavelet(encoded, wavelet),
        maxiter=iterations,
    ).reshape(volume.shape[1:])
    image = apps.partial_fourier(image, sampled)
    return image[0] if flat else image
