"""A multi-coil EPI pair of opposite phase-encode polarity, encoded line by line off resonance."""

from __future__ import annotations

import math

import torch

from bartorch import linop


def coil_maps(n: int, coils: int) -> torch.Tensor:
    """Smooth sensitivities of coils around the field of view, unit root sum of squares."""
    rows, columns = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij"
    )
    maps = []
    for coil in range(coils):
        angle = 2 * math.pi * coil / coils
        centre = (1.2 * math.sin(angle), 1.2 * math.cos(angle))
        magnitude = torch.exp(-((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2) / 0.8)
        maps.append(magnitude * torch.exp(1j * (0.7 * coil + 0.5 * columns)))
    maps = torch.stack(maps).to(torch.complex64)
    return maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()


def disc(n: int) -> torch.Tensor:
    """An ellipse with structure inside it, and a smooth phase."""
    rows, columns = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij"
    )
    inside = ((rows / 0.75) ** 2 + (columns / 0.6) ** 2) < 1
    image = inside * (1 + 0.5 * torch.sin(6 * rows) * torch.cos(5 * columns))
    return (image * torch.exp(0.5j * columns)).to(torch.complex64)


def field(n: int, peak: float) -> torch.Tensor:
    """A smooth off-resonance in Hz, peaking off centre."""
    rows, columns = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij"
    )
    return (peak * torch.exp(-((rows + 0.3) ** 2 + columns**2) / 0.15)).float()


def times(n: int, line_time: float, polarity: int, echo_time: float = 0.03) -> torch.Tensor:
    """Sample time of every line, the centre line at ``echo_time``."""
    return echo_time + polarity * line_time * (torch.arange(n, dtype=torch.float32) - n // 2)


def acquire(image, maps, off_resonance, line_times, lines) -> torch.Tensor:
    """``(coils, y, x)`` k-space of ``lines``, each from the image at that line's time."""
    n = image.shape[-1]
    encode = linop.CartesianSense(maps, (n, n))
    kspace = torch.zeros((maps.shape[0], n, n), dtype=torch.complex64)
    for line in lines:
        phase = torch.exp(2j * math.pi * off_resonance * float(line_times[line]))
        kspace[:, line] = encode(image * phase).reshape(maps.shape[0], n, n)[:, line]
    return kspace
