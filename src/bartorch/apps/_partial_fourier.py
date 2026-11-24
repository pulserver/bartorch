"""Partial-Fourier completion of an image by ``homodyne``."""

from __future__ import annotations

import numpy as np
import torch

from bartorch import tools

__all__ = ["partial_fourier", "partial_fourier_phase", "partial_fourier_range"]


def partial_fourier_range(sampled: torch.Tensor | np.ndarray, axis: int) -> tuple[int, int] | None:
    """First and last line acquired along a partial-Fourier ``axis``; ``None`` where it is not one.

    An axis is partial Fourier when the lines missing at one end of
    ``sampled`` outnumber the widest gap between sampled lines, which
    undersampling alone leaves. Line 0 of an even axis has no conjugate on
    the grid: a scan that reverses its phase encode leaves it out without
    being partial Fourier, so that one line does not count as missing.

    Parameters
    ----------
    sampled : torch.Tensor or numpy.ndarray
        Boolean sampling mask, True where a sample was acquired.
    axis : int
        Axis of ``sampled``.

    Returns
    -------
    tuple of int or None
        The acquired range, inclusive.
    """
    mask = sampled.cpu().numpy() if isinstance(sampled, torch.Tensor) else np.asarray(sampled)
    axis = axis % mask.ndim
    profile = mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis))
    lines = np.flatnonzero(profile)
    if lines.size < 2:
        return None
    gap = int(np.diff(lines).max()) - 1
    low, high = int(lines[0]), profile.size - 1 - int(lines[-1])
    unpaired = int(profile.size % 2 == 0 and low == high + 1)
    if max(low, high) - unpaired <= gap:
        return None
    return int(lines[0]), int(lines[-1])


def _partial_fourier(sampled: np.ndarray, axis: int) -> tuple[float, bool]:
    """The acquired fraction of ``axis`` and whether its high indices are the acquired side."""
    acquired = partial_fourier_range(sampled, axis)
    if acquired is None:
        return 1.0, False
    n = sampled.shape[axis]
    low, high = acquired[0], n - 1 - acquired[1]
    return (n - max(low, high)) / n, low > high


def partial_fourier_phase(image: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
    """Homodyne phase of ``image``: its k-space where a sample and its conjugate were both acquired.

    Along each partial-Fourier axis (:func:`partial_fourier_range`) the
    k-space of ``image`` is kept where both a sample and its conjugate lie in
    the acquired range: the phase a homodyne reconstruction removes.

    Parameters
    ----------
    image : torch.Tensor
        Complex image, the transform centred: the zero frequency is at index
        ``n // 2`` of each axis.
    sampled : torch.Tensor
        Boolean sampling mask, with the shape of ``image``.

    Returns
    -------
    torch.Tensor
        Complex64, the shape of ``image``.

    Raises
    ------
    ValueError
        If ``sampled`` and ``image`` differ in shape.
    """
    from bartorch._fourier import fft

    if sampled.shape != image.shape:
        raise ValueError(
            f"sampled has shape {tuple(sampled.shape)}, not the image's {tuple(image.shape)}"
        )
    mask = sampled.cpu().numpy()
    kept = np.ones(mask.shape, dtype=bool)
    for axis in range(mask.ndim):
        lines = partial_fourier_range(mask, axis)
        if lines is None:
            continue
        inside = np.zeros(mask.shape[axis], dtype=bool)
        inside[lines[0] : lines[1] + 1] = True
        shape = [1] * mask.ndim
        shape[axis] = mask.shape[axis]
        kept = kept & inside.reshape(shape)
    axes = tuple(range(-1, -mask.ndim - 1, -1))
    # Index i of a centred axis of n points holds the conjugate of index
    # 2 (n // 2) - i, the mirror image shifted by one point where n is even.
    mirrored = np.roll(np.flip(kept), [1 - n % 2 for n in mask.shape], axis=axes)
    symmetric = torch.from_numpy((kept & mirrored).astype(np.float32)).to(image.device)
    kspace = fft(image.to(torch.complex64), axes, unitary=True).reshape(image.shape)
    low = fft(kspace * symmetric, axes, inverse=True, unitary=True).reshape(image.shape)
    return torch.exp(1j * torch.angle(low)).to(torch.complex64)


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
