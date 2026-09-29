"""Off-resonance field maps from the phase of single-echo coil images."""

from __future__ import annotations

import math

import torch

__all__ = ["field_map_from_phase"]


def _smooth(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian smoothing of a complex volume, with replicated edges."""
    if sigma <= 0:
        return volume
    width = int(3 * sigma) | 1
    offsets = torch.arange(-(width // 2), width // 2 + 1, device=volume.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    result = volume
    for axis in range(3):
        view = [1, 1, 1]
        view[axis] = width
        padding = [0] * 6
        padding[2 * (2 - axis)] = width // 2
        padding[2 * (2 - axis) + 1] = width // 2
        parts = []
        for component in (result.real, result.imag):
            padded = torch.nn.functional.pad(component[None, None], padding, mode="replicate")
            parts.append(torch.nn.functional.conv3d(padded, kernel.reshape(1, 1, *view))[0, 0])
        result = torch.complex(*parts)
    return result


def _sensitivities(coil_images: torch.Tensor, width: float) -> torch.Tensor:
    """Coil images low-pass filtered by a Hann taper of half-width ``width`` per axis."""
    spectrum = torch.fft.fftn(coil_images, dim=(1, 2, 3))
    for axis, length in enumerate(coil_images.shape[1:]):
        coordinate = torch.fft.fftfreq(length, device=coil_images.device) * 2.0
        taper = torch.where(
            coordinate.abs() < width,
            0.5 + 0.5 * torch.cos(math.pi * coordinate / width),
            torch.zeros((), device=coil_images.device),
        )
        view = [1, 1, 1, 1]
        view[axis + 1] = length
        spectrum = spectrum * taper.reshape(view)
    return torch.fft.ifftn(spectrum, dim=(1, 2, 3))


def field_map_from_phase(
    coil_images: torch.Tensor,
    *,
    echo_time: float,
    smoothing: float = 2.0,
    calibration_width: float = 0.06,
) -> torch.Tensor:
    r"""Off-resonance frequency from the phase of single-echo coil images.

    Coil sensitivities :math:`s_c` are the coil images low-pass filtered to
    ``calibration_width`` of k-space.  The images are combined as
    :math:`m = \sum_c x_c s_c^* / \sum_c |s_c|^2`, which removes the coil and
    object phase varying as slowly as the sensitivities, the complex result is
    Gaussian-smoothed, and the field is :math:`f = \arg(m) / (2\pi\,
    \mathrm{TE})`.  It is unambiguous for :math:`|f| < 1 / (2\,\mathrm{TE})`
    and includes the chemical shift of fat.

    Parameters
    ----------
    coil_images : torch.Tensor
        Complex uncombined images ``(coils, z, y, x)``.
    echo_time : float
        Echo time, in seconds.
    smoothing : float, default=2.0
        Standard deviation of the Gaussian applied to the combined image, in
        voxels.
    calibration_width : float, default=0.06
        Half-width of the Hann taper that low-pass filters the coil images into
        sensitivities, as a fraction of each k-space axis's Nyquist extent.

    Returns
    -------
    torch.Tensor
        Off-resonance frequency in Hz, real, of shape ``(z, y, x)``.

    Raises
    ------
    ValueError
        If the images are not complex or not ``(coils, z, y, x)``, or
        ``echo_time`` is not positive.
    """
    if not coil_images.is_complex():
        raise ValueError("coil images must be complex")
    if coil_images.ndim != 4:
        raise ValueError(
            f"expected (coils, *spatial) with three spatial axes, got shape "
            f"{tuple(coil_images.shape)}"
        )
    if echo_time <= 0:
        raise ValueError("echo_time must be positive")

    sensitivities = _sensitivities(coil_images, calibration_width)
    combined = (coil_images * sensitivities.conj()).sum(0) / (
        sensitivities.abs().pow(2).sum(0) + 1e-12
    )
    return _smooth(combined, smoothing).angle() / (2 * math.pi * echo_time)
