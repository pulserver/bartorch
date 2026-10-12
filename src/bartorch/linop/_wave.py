"""The wave point-spread function from a phase-encode trajectory, and its calibration."""

from __future__ import annotations

import math

import numpy as np
import torch

from bartorch.linop._mri import _per_axis

__all__ = ["wave_calibrate", "wave_psf"]


def _trajectory(trajectory) -> np.ndarray:
    k = np.asarray(
        trajectory.detach().cpu() if isinstance(trajectory, torch.Tensor) else trajectory,
        dtype=np.float64,
    )
    if k.ndim == 1:
        k = k[:, None]
    if k.ndim != 2:
        raise ValueError(f"a trajectory is (readout, encodes), not {k.shape}")
    return k


def _played(k: np.ndarray, scale, delay) -> np.ndarray:
    """``scale * k(t - delay)`` per axis, with ``t`` and ``delay`` in readout samples.

    The delay is a linear phase on the spectrum of each axis's trajectory, so
    the trajectory is taken as periodic over the readout.
    """
    n = k.shape[0]
    spectrum = np.fft.rfft(k, axis=0)
    ramp = np.exp(-2j * math.pi * np.fft.rfftfreq(n)[:, None] * np.asarray(delay)[None, :])
    return np.asarray(scale)[None, :] * np.fft.irfft(spectrum * ramp, n, axis=0)


def _locations(n: int, fov: float) -> np.ndarray:
    """Voxel positions in metres, voxel ``n // 2`` at the centre."""
    return fov / n * (np.arange(n) - n // 2)


def wave_psf(
    trajectory,
    image_shape,
    fov,
    *,
    scale: float | tuple[float, ...] = 1.0,
    delay: float | tuple[float, ...] = 0.0,
    centred: bool = True,
) -> torch.Tensor:
    r"""Wave point-spread function from the wave's phase-encode trajectory.

    .. math::

        \mathrm{psf}[r, t] = \exp\left(-2\pi i \sum_a s_a k_a(t - d_a) \, r_a\right)

    over voxel positions :math:`r` along the phase encodes and readout samples
    :math:`t`, in the layout :func:`~bartorch.linop.WaveSense` takes as
    ``psf``.  Any waveform is accepted, for instance a tapered wave.

    Parameters
    ----------
    trajectory : array_like
        Wave k-space coordinates along the phase encodes in cycles/m, one row
        per ADC sample in acquisition order: ``(readout, encodes)`` with the
        encodes ``(z, y)`` in 3D or ``(y,)`` in 2D.  ``readout`` is the
        oversampled readout length.
    image_shape : tuple of int
        Spatial image shape ``([z,] y, x)`` before readout oversampling.
    fov : float or tuple of float
        Field of view along the phase encodes in m, one value or ``(z, y)``.
        Voxel ``n // 2`` is at the centre.
    scale : float or tuple of float, default=1.0
        Gradient amplitude relative to nominal, one value or ``(z, y)``.
    delay : float or tuple of float, default=0.0
        Delay of each wave relative to the ADC in readout samples, one value
        or ``(z, y)``.  Applied as a linear phase on the trajectory's
        spectrum, so the trajectory is taken as periodic over the readout.
    centred : bool, default=True
        Readout in acquisition order, for ``WaveSense(centred=True)``.  False
        applies ``ifftshift`` along the readout, for BART's uncentred
        transform.

    Returns
    -------
    tensor, shape ([z,] y, readout)
        The point spread function, complex128.

    Raises
    ------
    ValueError
        When the trajectory's axes do not match the phase encodes, or the
        readout is shorter than ``x``.
    """
    k = _trajectory(trajectory)
    image_shape = tuple(int(n) for n in image_shape)
    encodes = image_shape[:-1]
    readout, axes = k.shape
    if axes != len(encodes):
        raise ValueError(f"a trajectory of {axes} axes for {len(encodes)} phase encodes")
    if readout < image_shape[-1]:
        raise ValueError(f"a readout of {readout} is shorter than the image's {image_shape[-1]}")
    fovs = _per_axis(fov, axes, "fov")
    k = _played(k, _per_axis(scale, axes, "scale"), _per_axis(delay, axes, "delay"))

    phase = np.zeros((*encodes, readout))
    for axis, (n, length) in enumerate(zip(encodes, fovs)):
        shape = [1] * axes + [readout]
        shape[axis] = n
        phase = phase + (_locations(n, length)[:, None] * k[None, :, axis]).reshape(shape)
    psf = np.exp(-2j * math.pi * phase)
    if not centred:
        psf = np.fft.ifftshift(psf, axes=-1)
    return torch.from_numpy(psf)


def _kernel(lines_out, lines_in, shift, n) -> np.ndarray:
    """``(out, in, readout)``: the wave's point-spread function as a k-space convolution.

    Multiplying a centred unitary DFT of ``n`` points by ``exp(-2 pi i s p / n)``
    over ``p = -n // 2 ...`` moves line ``l`` into line ``j`` with weight
    ``(1/n) sum_p exp(-2 pi i (j - l + s) p / n)``, the Dirichlet kernel below.
    """
    u = lines_out[:, None, None] - lines_in[None, :, None] + shift[None, None, :]
    a = -(n // 2)
    return np.exp(-1j * math.pi * u * (2 * a + n - 1) / n) * np.sinc(u) / np.sinc(u / n)


def _along(mask: np.ndarray, axis: int) -> np.ndarray:
    return np.any(mask, axis=tuple(b for b in range(mask.ndim) if b != axis))


def _support(lines: np.ndarray) -> slice:
    where = np.flatnonzero(lines)
    return slice(int(where[0]), int(where[-1]) + 1)


def wave_calibrate(
    reference,
    wave,
    trajectory,
    fov,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Fit the wave's gradient amplitude and delay to a wave-free calibration region.

    The model predicts the wave-encoded samples from the reference: inverse
    FFT along the phase encodes, multiplication by :func:`wave_psf`, and FFT
    back.  Per phase-encode axis, scale and delay are fitted by nonlinear
    least squares (``scipy.optimize.least_squares``) starting from 1 and 0.
    The residual is taken on the acquired wave lines whose neighbourhood,
    within the wave's largest excursion in lines, lies inside the calibration
    region; k-space outside the region is neglected.

    Parameters
    ----------
    reference : array_like
        Wave-free k-space ``(coils, [kz,] ky, readout)``, zero outside the
        calibration region, with the readout in acquisition order.
    wave : array_like
        Wave-encoded k-space on the same grid, zero where not acquired.
    trajectory : array_like
        Nominal wave trajectory, as for :func:`wave_psf`.
    fov : float or tuple of float
        Field of view along the phase encodes in m, one value or ``(z, y)``.

    Returns
    -------
    scale, delay : tuple of float
        One per phase encode, ``(z, y)`` in 3D: the gradient amplitude
        relative to nominal and the delay in readout samples, as
        :func:`wave_psf` takes them.

    Raises
    ------
    ValueError
        When the shapes disagree, or no acquired line has its neighbourhood
        inside the calibration region.
    """
    from scipy.ndimage import binary_erosion
    from scipy.optimize import least_squares

    def array(x):
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        return x.astype(np.complex128)

    ref, data = array(reference), array(wave)
    k = _trajectory(trajectory)
    if ref.shape != data.shape:
        raise ValueError(f"reference {ref.shape} and wave {data.shape} differ")
    axes = ref.ndim - 2
    if k.shape != (ref.shape[-1], axes):
        raise ValueError(f"a trajectory of {k.shape} for {axes} phase encodes of {ref.shape[-1]}")
    grid = ref.shape[1:-1]
    fovs = _per_axis(fov, axes, "fov")

    region = np.any(ref != 0, axis=(0, -1))
    margin = [int(math.ceil(np.abs(k[:, a]).max() * fovs[a])) for a in range(axes)]
    inner = binary_erosion(region, np.ones([2 * m + 1 for m in margin], bool), border_value=0)
    fitted = inner & np.any(data != 0, axis=(0, -1))
    if not fitted.any():
        raise ValueError(f"no acquired line lies {margin} lines inside the calibration region")

    # Only the calibration region's lines feed the model and only its inner lines are compared.
    lines_in = [_support(_along(region, a)) for a in range(axes)]
    lines_out = [_support(_along(fitted, a)) for a in range(axes)]
    source = ref[(slice(None), *lines_in)]
    keep = fitted[tuple(lines_out)]
    target = data[(slice(None), *lines_out)][:, keep]
    norm = np.linalg.norm(target)

    def residual(p):
        played = _played(k, p[:axes], p[axes:])
        model = source
        for a in range(axes):
            K = _kernel(
                np.arange(grid[a])[lines_out[a]],
                np.arange(grid[a])[lines_in[a]],
                played[:, a] * fovs[a],
                grid[a],
            )
            moved = np.einsum("jit,...it->...jt", K, np.moveaxis(model, 1 + a, -2))
            model = np.moveaxis(moved, -2, 1 + a)
        r = (model[:, keep] - target).ravel() / norm
        return np.concatenate([r.real, r.imag])

    fit = least_squares(residual, np.r_[np.ones(axes), np.zeros(axes)])
    return tuple(float(v) for v in fit.x[:axes]), tuple(float(v) for v in fit.x[axes:])
