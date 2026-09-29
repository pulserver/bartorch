"""Off-resonance deblurring of spiral images by a separable factorization of the transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

__all__ = [
    "ReadoutTiming",
    "SpiralTransfer",
    "deblur",
    "fit_transfer",
]

Backend = Literal["auto", "fft", "conv"]


@dataclass(frozen=True)
class ReadoutTiming:
    """Readout time of a spiral arm as a function of the squared k-space radius.

    Parameters
    ----------
    squared_radius : numpy.ndarray
        Uniform grid on ``[0, 1]`` of :math:`|k|^2 / k_{max}^2`.
    times : numpy.ndarray
        Readout time at each grid point, as a fraction of ``duration``.
    density : numpy.ndarray
        Number of trajectory samples per grid interval, the weight of the fit
        in :func:`fit_transfer`.
    duration : float
        Readout duration, in seconds.
    """

    squared_radius: np.ndarray
    times: np.ndarray
    density: np.ndarray
    duration: float

    @classmethod
    def from_trajectory(
        cls,
        trajectory: np.ndarray,
        duration: float,
        *,
        points: int = 512,
    ) -> ReadoutTiming:
        """Time map of one readout arm whose samples are uniform in time.

        Parameters
        ----------
        trajectory : array_like
            k-space positions of one arm, ``(samples, ndim)``, in any unit.
        duration : float
            Readout duration, in seconds.
        points : int, default=512
            Size of the uniform squared-radius grid.

        Returns
        -------
        ReadoutTiming

        Raises
        ------
        ValueError
            If the radius is not monotonic in time, so that time is not a
            function of :math:`|k|`.  An out-in arm is split into its two
            halves, each deblurred separately.
        """
        trajectory = np.asarray(
            trajectory.detach().cpu() if isinstance(trajectory, torch.Tensor) else trajectory,
            dtype=float,
        )
        if trajectory.ndim != 2:
            raise ValueError(f"expected (samples, ndim), got {trajectory.shape}")
        radius = np.linalg.norm(trajectory, axis=1)
        step = np.diff(radius)
        tolerance = 1e-6 * max(radius.max(), 1.0)
        if not (np.all(step >= -tolerance) or np.all(step <= tolerance)):
            raise ValueError(
                "readout radius is not monotonic, so time is not a function of "
                "|k|; split an out-in arm and deblur each half separately"
            )
        peak = radius.max()
        if peak <= 0:
            raise ValueError("trajectory has zero extent")

        squared = (radius / peak) ** 2
        sample_times = np.linspace(0.0, 1.0, len(radius))
        grid = np.linspace(0.0, 1.0, points)
        order = np.argsort(squared)
        times = np.interp(grid, squared[order], sample_times[order])
        edges = np.linspace(0.0, 1.0, points + 1)
        density = np.histogram(squared, bins=edges)[0].astype(float)
        return cls(grid, times, np.maximum(density, 1e-6), float(duration))


@dataclass(frozen=True)
class SpiralTransfer:
    r"""Separable factorization of the off-resonance transfer of a spiral readout.

    .. math:: e^{-2\pi i f t(k)} \approx \sum_m a_m(f)\, e^{\alpha_m |k|^2 / k_{max}^2}

    with :math:`t(k)` the readout time at which :math:`k` is sampled.  Each
    k-space factor is a product over the axes, so each term is a separable
    convolution of the image, and the field map enters only through the
    per-voxel weights :math:`a_m(f)`.  As :math:`|k|` is invariant under
    rotation, one factorization serves every arm of a rotated 2D spiral or 3D
    spiral-projection acquisition.

    Parameters
    ----------
    rates : numpy.ndarray
        Complex rates :math:`\alpha_m`, ``(terms,)``.
    weights : numpy.ndarray
        Weights :math:`a_m(f)` at the tabulated frequencies,
        ``(terms, frequencies)``.
    frequencies : numpy.ndarray
        Uniformly spaced off-resonance frequencies, in Hz; a frequency outside
        them is clamped to the nearest end.
    """

    rates: np.ndarray
    weights: np.ndarray
    frequencies: np.ndarray

    @property
    def terms(self) -> int:
        """Number of separable terms."""
        return len(self.rates)

    @property
    def amplification(self) -> float:
        r"""Largest :math:`\sum_m |a_m(f)|` over the tabulated frequencies.

        The transfer has unit modulus, so a value much above one means nearly
        cancelling terms, whose residual is amplified by about this factor.  It
        grows with the phase accrued over the readout and can be large while
        :meth:`error` is small.
        """
        return float(np.abs(self.weights).sum(axis=0).max())

    def error(self, timing: ReadoutTiming) -> float:
        """Largest density-weighted RMS error of the factorization over the frequencies.

        Parameters
        ----------
        timing : ReadoutTiming
            The time map the transfer was fitted to.

        Returns
        -------
        float
        """
        exact = _exact_transfer(timing, self.frequencies)
        fitted = np.exp(np.outer(timing.squared_radius, self.rates)) @ self.weights
        weight = timing.density[:, None]
        residual = np.sum(weight * np.abs(fitted - exact) ** 2, axis=0)
        return float(np.sqrt(residual / timing.density.sum()).max())


def _exact_transfer(timing: ReadoutTiming, frequencies: np.ndarray) -> np.ndarray:
    """Off-resonance transfer on the (squared radius, frequency) grid."""
    cycles = frequencies * timing.duration
    return np.exp(-2j * np.pi * np.outer(timing.times, cycles))


def fit_transfer(
    timing: ReadoutTiming,
    *,
    band: float,
    terms: int = 6,
    frequencies: int = 65,
) -> SpiralTransfer:
    """Fit a :class:`SpiralTransfer` to a readout's time map.

    The rates are shared by every frequency and found by multi-snapshot ESPRIT
    over the columns of the exact transfer, with growing terms made purely
    oscillatory; the weights are the density-weighted least-squares fit at
    each frequency.

    Parameters
    ----------
    timing : ReadoutTiming
        Time map of the readout.
    band : float
        Half-width of the off-resonance range, in Hz.
    terms : int, default=6
        Number of separable terms; :func:`deblur` costs one k-space multiply or
        separable convolution per term.
    frequencies : int, default=65
        Number of tabulated frequencies spanning ``[-band, band]``.

    Returns
    -------
    SpiralTransfer
        The factorization; :meth:`SpiralTransfer.error` and
        :attr:`SpiralTransfer.amplification` report its quality.

    Examples
    --------
    >>> timing = ReadoutTiming.from_trajectory(arm, duration=5e-3)
    >>> transfer = fit_transfer(timing, band=100.0, terms=5)
    """
    if terms < 1:
        raise ValueError("terms must be at least 1")
    grid = timing.squared_radius
    tabulated = np.linspace(-band, band, frequencies)
    exact = _exact_transfer(timing, tabulated)

    rates = _shared_rates(grid, exact, terms)
    basis = np.exp(np.outer(grid, rates))
    root = np.sqrt(timing.density)[:, None]
    weights = np.linalg.lstsq(root * basis, root * exact, rcond=None)[0]
    return SpiralTransfer(rates, weights, tabulated)


def _shared_rates(grid: np.ndarray, exact: np.ndarray, terms: int) -> np.ndarray:
    """Rates of the dominant exponentials common to every frequency column.

    Multi-snapshot ESPRIT: every column is a sum of exponentials in the squared
    radius, so the columns share a signal subspace and one shift-invariance
    solve gives rates that serve the whole band.
    """
    window = len(grid) // 2
    blocks = [
        np.lib.stride_tricks.sliding_window_view(column, window)[:window] for column in exact.T
    ]
    subspace = np.linalg.svd(np.hstack(blocks), full_matrices=False)[0][:, :terms]
    shift = np.linalg.lstsq(subspace[:-1], subspace[1:], rcond=None)[0]
    poles = np.linalg.eigvals(shift).astype(complex)
    step = grid[1] - grid[0]
    rates = np.log(np.where(np.abs(poles) > 0, poles, 1.0)) / step
    # A growing term would amplify the corners of the Cartesian grid, which lie
    # outside the sampled sphere and carry no measured signal.
    return np.where(rates.real > 0, 1j * rates.imag, rates)


def _axis_factors(
    rate: complex,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """One-dimensional k-space factors whose product is ``exp(rate * u)``.

    Kept as separate axis factors so the full ``u`` volume is never formed.
    """
    factors = []
    for axis, length in enumerate(shape):
        coordinate = torch.fft.fftfreq(length, device=device) * 2.0
        factor = torch.exp(rate * coordinate.to(dtype) ** 2)
        view = [1] * len(shape)
        view[axis] = length
        factors.append(factor.reshape(view))
    return factors


def _term_weights(transfer: SpiralTransfer, term: int, field_map: torch.Tensor) -> torch.Tensor:
    """Per-voxel weight of one term, interpolated over the field map."""
    frequencies = transfer.frequencies
    spacing = frequencies[1] - frequencies[0]
    position = (field_map - float(frequencies[0])) / float(spacing)
    position = position.clamp(0, len(frequencies) - 1)
    lower = position.floor()
    fraction = (position - lower).to(field_map.dtype)
    index = lower.long()
    upper = index.clamp(max=len(frequencies) - 2) + 1
    table = torch.as_tensor(transfer.weights[term], device=field_map.device).to(
        _complex_for(field_map.dtype)
    )
    return torch.lerp(
        table[index.clamp(max=len(frequencies) - 2)],
        table[upper],
        fraction.to(table.dtype),
    )


def _complex_for(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _real_for(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.complex128 else torch.float32


def deblur(
    image: torch.Tensor,
    field_map: torch.Tensor,
    transfer: SpiralTransfer,
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    r"""Remove off-resonance blur from a reconstructed spiral image.

    The result is :math:`\sum_m a_m(f(r))\, (h_m * x)(r)`, with :math:`h_m`
    the image-domain kernel of term :math:`m` of ``transfer`` and the weights
    linearly interpolated at each voxel's frequency :math:`f(r)`.
    :math:`|k| / k_{max}` is the k-space coordinate normalized to one at the
    Nyquist edge of each axis: the arm ``transfer`` was fitted to is taken to
    reach the edge of the image's grid.

    Parameters
    ----------
    image : torch.Tensor
        Complex image ``(..., *spatial)``, 2D or 3D, with the spatial axes
        those of ``field_map``.
    field_map : torch.Tensor
        Off-resonance frequency at each voxel, in Hz, ``(*spatial)``.
    transfer : SpiralTransfer
        Factorization from :func:`fit_transfer`.
    backend : {"auto", "fft", "conv"}, default='auto'
        ``"fft"`` applies each term as a k-space multiply; ``"conv"``, for a
        CUDA image with Triton installed, as a separable circular convolution
        truncated to the taps holding all but 1e-5 of the kernel's energy per
        axis; ``"auto"`` chooses
        ``"conv"`` for a CUDA image when Triton is importable and ``"fft"``
        otherwise.

    Returns
    -------
    torch.Tensor
        The deblurred image, of the shape and dtype of ``image``.

    Raises
    ------
    ValueError
        If ``image`` is not complex or its spatial shape is not that of
        ``field_map``.

    Notes
    -----
    Peak memory is one accumulator and one working volume, independent of the
    number of terms.
    """
    if not image.is_complex():
        raise ValueError("image must be complex")
    spatial = field_map.ndim
    axes = tuple(range(image.ndim - spatial, image.ndim))
    if image.shape[-spatial:] != field_map.shape:
        raise ValueError(
            f"image spatial shape {tuple(image.shape[-spatial:])} does not match "
            f"field map {tuple(field_map.shape)}"
        )
    chosen = _resolve_backend(backend, image)
    shape = tuple(image.shape[-spatial:])

    result = torch.zeros_like(image)
    fused = _fused_accumulate(image)
    spectrum = torch.fft.fftn(image, dim=axes) if chosen == "fft" else None
    working = torch.empty_like(image) if chosen == "fft" else None
    for term in range(transfer.terms):
        rate = complex(transfer.rates[term])
        if chosen == "fft":
            factors = _axis_factors(rate, shape, device=image.device, dtype=_real_for(image.dtype))
            torch.mul(spectrum, factors[0], out=working)
            for factor in factors[1:]:
                working *= factor
            contribution = torch.fft.ifftn(working, dim=axes)
        else:
            from bartorch.tools._correct._triton_spiral import separable_convolve

            contribution = separable_convolve(image, rate, axes)
        if fused is not None:
            fused(
                result,
                contribution,
                field_map,
                torch.as_tensor(transfer.weights[term], device=image.device, dtype=image.dtype),
                float(transfer.frequencies[0]),
                float(transfer.frequencies[1] - transfer.frequencies[0]),
            )
            continue
        result += _term_weights(transfer, term, field_map) * contribution
        del contribution
    return result


def _fused_accumulate(image: torch.Tensor):
    """The Triton weighted accumulator, when the device and dtype allow it."""
    if image.device.type != "cuda" or image.dtype != torch.complex64:
        return None
    try:
        from bartorch.tools._correct._triton_spiral import accumulate_weighted
    except ImportError:
        return None
    return accumulate_weighted


def _resolve_backend(backend: Backend, image: torch.Tensor) -> str:
    if backend not in ("auto", "fft", "conv"):
        raise ValueError(f"backend must be 'auto', 'fft' or 'conv', not {backend!r}")
    if backend != "auto":
        return backend
    if image.device.type != "cuda":
        return "fft"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "fft"
    return "conv"
