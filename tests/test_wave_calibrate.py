"""The wave point-spread function from a trajectory, and its gradient calibration.

Each is pinned against the encoding written out here: a phase of
``exp(-2 pi i k(t) . r)`` on the voxel positions, and the centred unitary
transform from numpy.
"""

import math

import numpy as np
import pytest

from bartorch.linop import wave_calibrate, wave_psf

READOUT = 48
CYCLES = 3
FOV = (0.16, 0.2)  # (z, y), metres
GRID = (20, 24)  # (z, y)


def _tapered_wave(t, amplitude=(15.0, 12.5)):
    """``(kz, ky)`` in cycles/m of a balanced, tapered wave: a cosine on z, a sine on y."""
    taper = np.sin(math.pi * t / READOUT) ** 2
    phase = 2 * math.pi * CYCLES * t / READOUT
    return np.stack(
        [amplitude[0] * taper * np.cos(phase), amplitude[1] * taper * np.sin(phase)], axis=-1
    )


def _positions(n, fov):
    return fov / n * (np.arange(n) - n // 2)


def _fftc(x, axes):
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes
    )


def _explicit_psf(k):
    z = _positions(GRID[0], FOV[0])[:, None, None]
    y = _positions(GRID[1], FOV[1])[None, :, None]
    return np.exp(-2j * math.pi * (k[:, 0] * z + k[:, 1] * y))


def test_wave_psf_is_the_phase_of_the_trajectory_at_the_voxel_positions():
    k = _tapered_wave(np.arange(READOUT))
    psf = wave_psf(k, (*GRID, 32), FOV).numpy()
    assert psf.shape == (*GRID, READOUT)
    np.testing.assert_allclose(psf, _explicit_psf(k), atol=1e-12)


def test_wave_psf_at_a_whole_sample_delay_is_the_trajectory_rolled():
    k = _tapered_wave(np.arange(READOUT))
    delayed = wave_psf(k, (*GRID, 32), FOV, delay=(2.0, -3.0), scale=(1.1, 0.9)).numpy()
    rolled = np.stack([1.1 * np.roll(k[:, 0], 2), 0.9 * np.roll(k[:, 1], -3)], axis=-1)
    np.testing.assert_allclose(delayed, _explicit_psf(rolled), atol=1e-10)


def test_wave_psf_uncentred_is_the_readout_reordered_for_the_uncentred_transform():
    k = _tapered_wave(np.arange(READOUT))
    uncentred = wave_psf(k, (*GRID, 32), FOV, centred=False).numpy()
    np.testing.assert_allclose(uncentred, np.roll(_explicit_psf(k), -(READOUT // 2), -1))


def _phantom(rng):
    """Smooth coil images ``(coils, z, y, readout)``: Gaussian blobs under smooth coils."""
    z, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in (*GRID, READOUT)), indexing="ij")
    image = np.zeros(z.shape, complex)
    for cz, cy, cx, width, amplitude in [
        (0, 0, 0, 3.0, 1.0),
        (2, -3, 4, 2.5, 0.6j),
        (-3, 2, -5, 2.0, -0.5 + 0.3j),
    ]:
        image += amplitude * np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / width**2)
    coils = []
    for sz, sy in [(-8, -8), (-8, 8), (8, -8), (8, 8)]:
        profile = np.exp(-((z - sz) ** 2 + (y - sy) ** 2) / 400.0)
        coils.append(profile * np.exp(1j * (0.05 * sz * z + 0.04 * sy * y + rng.uniform(0, 6))))
    return np.stack(coils) * image


def _acquisition(scale, delay):
    """The reference and the wave k-space, as pypulseqpp's wave-CAIPI GRE acquires them."""
    hybrid = _fftc(_phantom(np.random.default_rng(0)), (-1,))
    t = np.arange(READOUT)
    played = np.stack([scale[a] * _tapered_wave(t - delay[a])[:, a] for a in range(2)], axis=-1)
    wave = _fftc(hybrid * _explicit_psf(played), (1, 2))
    reference = _fftc(hybrid, (1, 2))

    lz, ly = np.meshgrid(*(np.arange(n) - n // 2 for n in GRID), indexing="ij")
    calibration = (np.abs(lz) < 7) & (np.abs(ly) < 8)
    caipi = (ly % 2 == 0) & ((lz + ly // 2) % 2 == 0)
    reference = reference * calibration[None, :, :, None]
    wave = wave * (calibration | caipi)[None, :, :, None]
    return reference, wave, _tapered_wave(t)


@pytest.mark.parametrize("scale, delay", [((1.0, 1.0), (0.0, 0.0)), ((1.03, 0.97), (1.5, -0.7))])
def test_wave_calibrate_recovers_the_gradient_scale_and_delay(scale, delay):
    reference, wave, nominal = _acquisition(scale, delay)
    fitted_scale, fitted_delay = wave_calibrate(reference, wave, nominal, FOV)
    np.testing.assert_allclose(fitted_scale, scale, atol=1e-4)
    np.testing.assert_allclose(fitted_delay, delay, atol=1e-3)


def test_wave_calibrate_refuses_a_calibration_region_narrower_than_the_wave():
    reference, wave, nominal = _acquisition((1.0, 1.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="inside the calibration region"):
        wave_calibrate(reference, wave, 4 * nominal, FOV)
