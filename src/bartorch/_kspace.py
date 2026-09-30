"""K-space preprocessing: apodization windows and readout-oversampling removal."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

__all__ = ["apodize", "fermi_window", "hann_window", "remove_readout_oversampling"]

#: Default Fermi transition width, in samples of the longest axis: Bernstein et
#: al.'s product setting ``T = 10 / (N / 2)``.
_FERMI_TRANSITION_SAMPLES = 10.0


def _coordinates(shape: tuple[int, ...], device) -> list[torch.Tensor]:
    """Per-axis coordinates, zero at index ``n // 2`` and one at the Nyquist edge.

    Each axis is normalized on its own, so that a radial window over an
    anisotropic grid is an ellipsoid matched to the grid's axes.
    """
    axes = []
    for axis, size in enumerate(shape):
        if size < 1:
            raise ValueError(f"every axis must have at least one sample, got {shape}")
        index = torch.arange(size, device=device, dtype=torch.float64)
        view = [1] * len(shape)
        view[axis] = size
        axes.append(((index - size // 2) / max(size // 2, 1)).reshape(view))
    return axes


def _extend(
    kernel: Callable[[torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
    geometry: str,
    device,
) -> torch.Tensor:
    """A one-dimensional kernel over a grid: of the radius, or its product over the axes."""
    coordinates = _coordinates(shape, device)
    if geometry == "radial":
        window = kernel(sum(coordinate**2 for coordinate in coordinates).sqrt())
    elif geometry == "separable":
        window = math.prod(kernel(coordinate.abs()) for coordinate in coordinates)
    else:
        raise ValueError(f"geometry must be radial or separable, got {geometry!r}")
    return torch.broadcast_to(window, shape).to(torch.float32).contiguous()


def fermi_window(
    shape: tuple[int, ...],
    *,
    radius: float = 1.0,
    width: float | None = None,
    geometry: str = "radial",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Fermi apodization window over a centred k-space grid.

    The kernel is

    .. math:: W(u) = \frac{1}{1 + e^{(u - r) / w}},

    with :math:`u` the k-space coordinate normalized per axis to zero at index
    ``n // 2`` and one at the Nyquist edge, :math:`r` the ``radius`` and
    :math:`w` the ``width`` (Bernstein et al. [1]_).

    Parameters
    ----------
    shape : tuple of int
        Grid shape, any number of axes.
    radius : float, default=1.0
        Coordinate of the half-height, as a fraction of the Nyquist edge.  At
        the default a radial window is exactly 0.5 at ``(k_max, 0)``.
    width : float, default=None
        Transition width in the same coordinate.  ``None`` is ten samples of
        the longest axis, ``10 / (max(shape) // 2)``.
    geometry : {"radial", "separable"}, default='radial'
        ``"radial"`` evaluates the kernel on the Euclidean norm of the
        coordinates, an ellipsoid over the grid; ``"separable"`` multiplies the
        kernel along each axis, which retains more of the corners.
    device : torch.device or str, default=None
        Device of the window.

    Returns
    -------
    torch.Tensor
        Real ``float32`` window of shape ``shape``.

    Raises
    ------
    ValueError
        If ``width`` is not positive, ``geometry`` is neither value, or an axis
        is empty.

    References
    ----------
    .. [1] Bernstein MA, Fain SB, Riederer SJ. Effect of windowing and
       zero-filled reconstruction of MRI data on spatial resolution and
       acquisition strategy. J Magn Reson Imaging 2001;14:270-280.

    Examples
    --------
    >>> window = fermi_window((256, 256))
    >>> float(window[128, 128]), round(float(window[128, 0]), 3)
    (1.0, 0.5)
    """
    shape = tuple(int(n) for n in shape)
    if width is None:
        width = _FERMI_TRANSITION_SAMPLES / max(max(shape) // 2, 1)
    if width <= 0.0:
        raise ValueError(f"width must be positive, got {width}")

    def kernel(u: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-(u - radius) / width)

    return _extend(kernel, shape, geometry, device)


def hann_window(
    shape: tuple[int, ...],
    *,
    radius: float = 1.0,
    geometry: str = "radial",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Hann (raised-cosine) apodization window over a centred k-space grid.

    The kernel is :math:`W(u) = \tfrac12 (1 + \cos(\pi u / r))` for
    :math:`u < r` and zero beyond, with :math:`u` normalized as in
    :func:`fermi_window` and :math:`r` the ``radius``.

    Parameters
    ----------
    shape : tuple of int
        Grid shape, any number of axes.
    radius : float, default=1.0
        Coordinate at which the window reaches zero, as a fraction of the
        Nyquist edge.
    geometry : {"radial", "separable"}, default='radial'
        As in :func:`fermi_window`.
    device : torch.device or str, default=None
        Device of the window.

    Returns
    -------
    torch.Tensor
        Real ``float32`` window of shape ``shape``.

    Raises
    ------
    ValueError
        If ``radius`` is not positive, ``geometry`` is neither value, or an
        axis is empty.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")

    def kernel(u: torch.Tensor) -> torch.Tensor:
        scaled = u / radius
        taper = 0.5 * (1.0 + torch.cos(math.pi * scaled.clamp(max=1.0)))
        return taper * (scaled < 1.0)

    return _extend(kernel, tuple(int(n) for n in shape), geometry, device)


def apodize(
    kspace: torch.Tensor,
    *,
    kind: str = "fermi",
    axes: tuple[int, ...] = (-2, -1),
    **kwargs,
) -> torch.Tensor:
    """Multiply centred k-space by an apodization window over ``axes``.

    The window is built over ``axes`` and broadcast along the other axes, so
    every coil, frame or slice is weighted identically.

    Parameters
    ----------
    kspace : torch.Tensor
        Centred k-space: the zero frequency at index ``n // 2`` along ``axes``.
    kind : {"fermi", "hann"}, default='fermi'
        :func:`fermi_window` or :func:`hann_window`.
    axes : tuple of int, default=(-2, -1)
        Axes spanning the k-space grid.
    **kwargs
        Passed to the window: ``radius``, ``width`` and ``geometry`` for
        ``"fermi"``, ``radius`` and ``geometry`` for ``"hann"``.

    Returns
    -------
    torch.Tensor
        The apodized k-space, on the device of ``kspace``.

    Raises
    ------
    ValueError
        If ``kind`` is neither value.
    """
    builders = {"fermi": fermi_window, "hann": hann_window}
    if kind not in builders:
        raise ValueError(f"kind must be fermi or hann, got {kind!r}")
    kspace = torch.as_tensor(kspace)
    axes = tuple(axis % kspace.ndim for axis in axes)
    shape = tuple(kspace.shape[axis] for axis in axes)
    window = builders[kind](shape, device=kspace.device, **kwargs)
    view = [1] * kspace.ndim
    for axis, size in zip(axes, shape, strict=True):
        view[axis] = size
    real = kspace.real.dtype if kspace.is_complex() else kspace.dtype
    if not real.is_floating_point:
        real = torch.float32
    return kspace * window.to(real).reshape(view)


def remove_readout_oversampling(
    kspace: torch.Tensor, target_size: int, *, axis: int = -1
) -> torch.Tensor:
    """Crop the readout's field of view to ``target_size`` samples, in the image domain.

    The readout is transformed to the image domain with the centred unitary
    transform (:func:`bartorch.fft`), cropped symmetrically about index
    ``n // 2`` and transformed back: the result is k-space over the same
    extent, sampled ``n / target_size`` times more coarsely.

    Parameters
    ----------
    kspace : torch.Tensor
        Centred k-space with the readout along ``axis``.
    target_size : int
        Number of readout samples to keep, the reconstructed matrix size.
    axis : int, default=-1
        Readout axis.

    Returns
    -------
    torch.Tensor
        Complex64 k-space with ``target_size`` samples along ``axis``, or ``kspace``
        itself when it already has ``target_size``.

    Raises
    ------
    ValueError
        If ``target_size`` is not in ``[1, n]``.
    """
    from bartorch._fourier import fft
    from bartorch._util import resize

    kspace = torch.as_tensor(kspace)
    current = kspace.shape[axis]
    if not 0 < target_size <= current:
        raise ValueError(f"target_size must be in [1, {current}], got {target_size}")
    if target_size == current:
        return kspace
    image = fft(kspace, axis, inverse=True, unitary=True).reshape(kspace.shape)
    shape = list(image.shape)
    shape[axis] = target_size
    return fft(resize(image, tuple(shape)), axis, unitary=True).reshape(shape)
