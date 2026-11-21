"""Calibrated k-space kernels: SPIRiT self-consistency and split-slice GRAPPA band separation."""

from __future__ import annotations

import math

import torch

from bartorch._fourier import fft
from bartorch._util import casorati

__all__ = [
    "slice_grappa",
    "slice_grappa_kernel",
    "spirit_kernel",
    "spirit_weights",
]


def _patches(kspace: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Return every whole ``size`` block of ``(coils, *grid)`` k-space as a row.

    ``(blocks, coils * prod(size))``.
    """
    axes = tuple(range(-len(size), 0))
    return casorati(kspace, size, axes).transpose(0, 1)


def _ridge(sources: torch.Tensor, targets: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Return the ridge solution ``W`` of ``sources W = targets``.

    It minimises ``|sources W - targets|^2 + mu |W|^2``, with ``mu`` equal to
    ``lambda_`` times the mean squared column norm of ``sources``.
    """
    normal = sources.mH @ sources
    scale = lambda_ * torch.linalg.matrix_norm(sources) ** 2 / sources.shape[1]
    normal = normal + scale * torch.eye(normal.shape[0], dtype=normal.dtype, device=normal.device)
    return torch.linalg.solve(normal, sources.mH @ targets)


def spirit_kernel(
    calibration: torch.Tensor, size: int | tuple[int, ...] = 5, lambda_: float = 1e-3
) -> torch.Tensor:
    """Return the SPIRiT kernel: each sample predicted from its neighbourhood in every coil.

    Fitted by regularised least squares on every whole block of the
    calibration region: coil ``c``'s sample at a block's centre from all
    samples of the block but itself.

    Parameters
    ----------
    calibration : torch.Tensor
        Fully sampled ``(coils, *grid)`` k-space, the calibration region.
    size : int or tuple of int, default=5
        Kernel extent along each grid axis, odd.
    lambda_ : float, default=1e-3
        Tikhonov weight, relative to the mean squared source.

    Returns
    -------
    torch.Tensor
        ``(coils, coils, *size)``: entry ``[c, d, *offset]`` weighs coil
        ``d``'s sample at ``offset - size // 2`` from the target. The target
        itself has weight zero.

    Raises
    ------
    ValueError
        If a kernel extent is even or exceeds the calibration region.
    """
    ndim = calibration.ndim - 1
    size = (size,) * ndim if isinstance(size, int) else tuple(size)
    if any(n % 2 == 0 for n in size):
        raise ValueError(f"a SPIRiT kernel is centred on its target: extents {size} must be odd")
    if any(n > g for n, g in zip(size, calibration.shape[1:])):
        raise ValueError(
            f"a {size} kernel does not fit the calibration region {tuple(calibration.shape[1:])}"
        )
    coils, taps = calibration.shape[0], math.prod(size)
    rows = _patches(calibration, size)
    centre = taps // 2
    kernel = torch.zeros(coils, coils * taps, dtype=rows.dtype, device=rows.device)
    for coil in range(coils):
        target = coil * taps + centre
        keep = torch.arange(coils * taps, device=rows.device) != target
        kernel[coil, keep] = _ridge(rows[:, keep], rows[:, target : target + 1], lambda_)[:, 0]
    return kernel.reshape(coils, coils, *size)


def spirit_weights(kernel: torch.Tensor, grid: tuple[int, ...]) -> torch.Tensor:
    """Return the image-domain form of a SPIRiT kernel on a grid: the coil mixing at every voxel.

    Applying the kernel to centred, unitary k-space of shape
    ``(coils, *grid)`` equals transforming to the image, mixing the coils at
    each voxel by these weights, and transforming back.

    Parameters
    ----------
    kernel : torch.Tensor
        ``(coils, coils, *size)``, as :func:`spirit_kernel` returns it.
    grid : tuple of int
        The k-space grid, each axis at least the kernel's extent.

    Returns
    -------
    torch.Tensor
        ``(coils, coils, *grid)``: output coil, input coil, voxel.
    """
    size = tuple(kernel.shape[2:])
    padded = torch.zeros(*kernel.shape[:2], *grid, dtype=kernel.dtype, device=kernel.device)
    flipped = torch.flip(kernel, dims=tuple(range(2, kernel.ndim)))
    index = tuple(slice(g // 2 - n // 2, g // 2 - n // 2 + n) for g, n in zip(grid, size))
    padded[(slice(None), slice(None), *index)] = flipped
    axes = tuple(range(2, kernel.ndim))
    return fft(padded, axes, inverse=True, unitary=True).reshape(padded.shape) * math.sqrt(
        math.prod(grid)
    )


def slice_grappa_kernel(
    bands: torch.Tensor,
    encoding: torch.Tensor,
    size: int | tuple[int, ...] = (3, 5),
    lambda_: float = 1e-3,
) -> torch.Tensor:
    """Return split-slice GRAPPA kernels that separate the bands of collapsed multiband k-space.

    Each band is encoded into the collapsed k-space by a weight per line
    (the blipped-CAIPI phase). The kernels are fitted on one training set
    per band, that band's calibration alone, collapsed: band ``b``'s kernel
    reproduces it from its own set and nothing from the others' sets (the
    split-slice leak block), so it rejects the other bands' signal rather
    than only fitting the sum.

    Parameters
    ----------
    bands : torch.Tensor
        ``(bands, coils, lines, *rest)`` single-band calibration k-space, on
        the lattice of the acquired lines.
    encoding : torch.Tensor
        ``(bands, lines)``: the weight band ``b`` takes on each line of the
        collapsed k-space.
    size : int or tuple of int, default=(3, 5)
        Kernel extent along lines and the remaining axes, odd.
    lambda_ : float, default=1e-3
        Tikhonov weight, relative to the mean squared source.

    Returns
    -------
    torch.Tensor
        ``(bands, coils, coils * prod(size))``: the weights of each band's
        centre sample on a block of collapsed k-space.
    """
    count, coils = bands.shape[:2]
    ndim = bands.ndim - 2
    size = (size,) * ndim if isinstance(size, int) else tuple(size)
    shape = (1, -1) + (1,) * (ndim - 1)
    encoded = bands * encoding.reshape(count, 1, *shape[1:]).to(bands.dtype)
    centre = math.prod(size) // 2
    sources = torch.cat([_patches(encoded[b], size) for b in range(count)])
    kernels = []
    for band in range(count):
        targets = torch.cat(
            [
                _patches(encoded[b], size)[:, centre :: math.prod(size)]
                if b == band
                else torch.zeros_like(_patches(encoded[b], size)[:, centre :: math.prod(size)])
                for b in range(count)
            ]
        )
        kernels.append(_ridge(sources, targets, lambda_).T)
    return torch.stack(kernels)


def slice_grappa(
    collapsed: torch.Tensor, kernels: torch.Tensor, size: int | tuple[int, ...] = (3, 5)
) -> torch.Tensor:
    """Return each band's k-space separated from collapsed multiband k-space by split-slice GRAPPA.

    Parameters
    ----------
    collapsed : torch.Tensor
        ``(coils, lines, *rest)`` k-space on the lattice of the acquired lines.
    kernels : torch.Tensor
        ``(bands, coils, coils * prod(size))`` from :func:`slice_grappa_kernel`.
    size : int or tuple of int, default=(3, 5)
        The kernels' extent.

    Returns
    -------
    torch.Tensor
        ``(bands, coils, lines, *rest)``, each band carrying the weight per
        line it was encoded with; zero-padded at the edges.
    """
    ndim = collapsed.ndim - 1
    size = (size,) * ndim if isinstance(size, int) else tuple(size)
    pad = []
    for n in reversed(size):
        pad += [n // 2, n // 2]
    padded = torch.nn.functional.pad(collapsed.unsqueeze(0), pad).squeeze(0)
    rows = _patches(padded, size)
    out = torch.einsum("bcs,ns->bcn", kernels, rows)
    return out.reshape(kernels.shape[0], kernels.shape[1], *collapsed.shape[1:])
