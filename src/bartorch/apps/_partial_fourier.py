"""Partial-Fourier completion of an image by ``homodyne``."""

from __future__ import annotations

import numpy as np
import torch

from bartorch import tools

__all__ = ["partial_fourier"]


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


def partial_fourier(image: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
    """Homodyne completion of an image reconstructed from partial-Fourier k-space.

    An axis is partial Fourier when the lines missing at one end of ``sampled``
    outnumber the widest gap between sampled lines, which undersampling alone
    leaves. Each such axis is completed by ``bart homodyne -I -C`` with the
    acquired fraction of its lines, on the flipped image when the acquired
    side is the high indices. Other axes are untouched.

    Parameters
    ----------
    image : torch.Tensor
        Complex image ``([z,] y, x)`` reconstructed from the sampled k-space.
        The Fourier transform relating the two is centred: the zero frequency
        is at index ``n // 2`` of each axis.
    sampled : torch.Tensor
        Boolean sampling mask of that k-space, with the shape of ``image``;
        True where a sample was acquired.

    Returns
    -------
    torch.Tensor
        Complex image with the shape of ``image``. A completed image is
        real-valued: homodyne reconstruction removes the phase estimated from
        the central lines of k-space and keeps the real part.

    Raises
    ------
    ValueError
        If ``sampled`` and ``image`` differ in shape.
    """
    if sampled.shape != image.shape:
        raise ValueError(
            f"sampled has shape {tuple(sampled.shape)}, not the image's {tuple(image.shape)}"
        )
    image = image.to(torch.complex64)
    mask = sampled.cpu().numpy()
    for axis in range(image.ndim):
        acquired, high = _partial_fourier(mask, axis)
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
    return image
