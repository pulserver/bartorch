"""Apodization windows over a centred k-space grid, and readout-oversampling removal."""

import numpy as np
import pytest
import torch

import bartorch


def test_the_fermi_window_is_flat_at_the_centre_and_gone_at_the_corner():
    window = bartorch.fermi_window((256, 256))
    assert window[128, 128] > 0.999
    assert window[0, 0] < 0.01


def test_the_fermi_half_height_sits_on_each_axis_nyquist_edge():
    """Bernstein et al.'s normalization: the kernel's FWHM is at u = 1."""
    window = bartorch.fermi_window((256, 256))
    assert float(window[128, 0]) == pytest.approx(0.5)
    assert float(window[0, 128]) == pytest.approx(0.5)


def test_the_default_transition_is_ten_samples_of_the_longest_axis():
    """T = 10 / (N / 2), the product setting the paper describes."""
    for shape in ((256, 256), (64, 256), (32, 128, 128)):
        stated = bartorch.fermi_window(shape, width=10.0 / (max(shape) // 2))
        torch.testing.assert_close(bartorch.fermi_window(shape), stated)


def test_the_default_transition_leaves_a_slab_flat_at_dc():
    window = bartorch.fermi_window((32, 256, 256))
    assert window[16, 128, 128] > 0.999


def test_the_radial_geometry_apodizes_the_corners_as_the_paper_measures():
    """Bernstein et al., Eq. 13: the diagonal Nyquist point, radial over separable.

    The window at DC with the radius moved to ``1 - u`` is the one-dimensional
    kernel at ``u``.  The paper's N = 256 transition width gives 52.4% in two
    dimensions and 50.7% in three.
    """
    transition = 10.0 / 128

    def kernel(u):
        return float(bartorch.fermi_window((4, 4), radius=1.0 - u, width=transition)[2, 2])

    assert 0.5 / kernel(1 / np.sqrt(2)) ** 2 == pytest.approx(0.524, abs=5e-4)
    assert 0.5 / kernel(1 / np.sqrt(3)) ** 3 == pytest.approx(0.507, abs=5e-4)


def test_the_separable_geometry_keeps_more_of_the_corners_than_the_radial_one():
    ratios = []
    for shape in ((128, 128), (128, 128, 128)):
        corner = (0,) * len(shape)
        radial = float(bartorch.fermi_window(shape, geometry="radial")[corner])
        separable = float(bartorch.fermi_window(shape, geometry="separable")[corner])
        assert separable > radial
        ratios.append(radial / separable)
    assert ratios[1] < ratios[0]


def test_a_geometry_that_names_neither_extension_is_refused():
    with pytest.raises(ValueError, match="radial or separable"):
        bartorch.fermi_window((8, 8), geometry="cartesian")


def test_a_three_dimensional_window_is_an_ellipsoid_on_its_own_axes():
    window = bartorch.hann_window((8, 16, 64))
    assert window[4, 8, 32] == 1.0
    for corner in ((0, 8, 32), (4, 0, 32), (4, 8, 0)):
        assert window[corner] == 0.0


def test_the_fermi_radius_moves_where_the_roll_off_sits():
    narrow = bartorch.fermi_window((64, 64), radius=0.4)
    wide = bartorch.fermi_window((64, 64), radius=0.9)
    assert wide.sum() > narrow.sum()


def test_a_wider_transition_is_a_gentler_roll_off():
    sharp = bartorch.fermi_window((64, 64), radius=0.6, width=0.01).numpy()
    gentle = bartorch.fermi_window((64, 64), radius=0.6, width=0.2).numpy()
    assert 0.0 < gentle[32, 51] < 1.0
    assert np.abs(np.gradient(gentle[32])).max() < np.abs(np.gradient(sharp[32])).max()


def test_the_hann_taper_is_one_at_the_centre_and_zero_at_its_radius():
    window = bartorch.hann_window((32, 32))
    assert window[16, 16] == 1.0
    assert window[0, 0] == 0.0


def test_the_hann_taper_matches_the_raised_cosine_along_an_axis():
    """``0.5 (1 + cos(pi u))`` with ``u = (i - n // 2) / (n // 2)``."""
    u = (np.arange(32) - 16) / 16
    expected = 0.5 * (1 + np.cos(np.pi * u))
    np.testing.assert_allclose(bartorch.hann_window((32,)).numpy(), expected, atol=1e-6)


def test_the_hann_taper_costs_more_than_a_fermi_roll_off_of_the_same_radius():
    assert (
        bartorch.hann_window((64, 64), radius=0.8).sum()
        < bartorch.fermi_window((64, 64), radius=0.8).sum()
    )


def test_a_window_is_ellipsoidal_on_an_anisotropic_grid():
    window = bartorch.hann_window((16, 64))
    assert window[0, 32] == 0.0
    assert window[8, 0] == 0.0
    assert window[8, 32] == 1.0


def test_a_window_is_built_on_the_device_asked_for(device):
    window = bartorch.fermi_window((16, 16), device=device)
    assert window.device.type == device
    assert window.dtype == torch.float32


def test_a_non_positive_transition_width_is_refused():
    with pytest.raises(ValueError, match="width must be positive"):
        bartorch.fermi_window((8, 8), width=0.0)


def test_a_non_positive_taper_radius_is_refused():
    with pytest.raises(ValueError, match="radius must be positive"):
        bartorch.hann_window((8, 8), radius=0.0)


def test_apodize_broadcasts_one_window_across_every_channel(device):
    kspace = torch.ones(4, 32, 32, dtype=torch.complex64, device=device)
    apodized = bartorch.apodize(kspace, kind="hann")
    assert apodized.device.type == device
    assert apodized.dtype == torch.complex64
    apodized = apodized.cpu()
    assert apodized.shape == (4, 32, 32)
    torch.testing.assert_close(apodized[0], apodized[3])
    assert apodized[0, 0, 0] == 0.0


def test_apodize_works_over_named_axes(device):
    kspace = torch.ones(32, 4, 32, dtype=torch.complex64, device=device)
    apodized = bartorch.apodize(kspace, kind="hann", axes=(0, 2)).cpu()
    assert apodized.shape == (32, 4, 32)
    torch.testing.assert_close(apodized[:, 0], apodized[:, 3])


def test_apodizing_suppresses_truncation_ringing():
    """Side lobes of the point spread function, on a grid four times finer.

    On the measurement's own grid a hard truncation's side lobes fall on the
    Dirichlet nulls and read as zero.
    """
    kspace = torch.ones(64, 64, dtype=torch.complex64)
    fine = (256, 256)

    def psf(measurement):
        padded = bartorch.resize(measurement, fine)
        return bartorch.fft(padded, (0, 1), inverse=True, unitary=True).abs().numpy()

    def side_lobe(pattern):
        mask = np.ones(fine, dtype=bool)
        mask[120:136, 120:136] = False
        return pattern[mask].max() / pattern.max()

    sharp = side_lobe(psf(kspace))
    tapered = side_lobe(psf(bartorch.apodize(kspace, kind="hann")))
    assert sharp > 0.1
    assert tapered < 0.25 * sharp


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="fermi or hann"):
        bartorch.apodize(torch.ones(8, 8, dtype=torch.complex64), kind="gaussian")


def test_the_bart_window_command_is_not_exported():
    assert "window" not in bartorch.__all__
    assert not hasattr(bartorch, "window")


# --- readout oversampling ---------------------------------------------------


def _centred_fft(x, inverse=False):
    """Centred unitary DFT along the last axis, written with numpy."""
    op = np.fft.ifft if inverse else np.fft.fft
    return np.fft.fftshift(op(np.fft.ifftshift(x, axes=-1), axis=-1, norm="ortho"), axes=-1)


def test_removing_oversampling_halves_the_readout(device):
    rng = np.random.default_rng(0)
    readout = _centred_fft(rng.normal(size=(4, 128)) + 1j * rng.normal(size=(4, 128)))
    kspace = torch.as_tensor(readout, dtype=torch.complex64, device=device)
    cropped = bartorch.remove_readout_oversampling(kspace, 64)
    assert cropped.shape == (4, 64)
    assert cropped.device.type == device


def test_removing_oversampling_keeps_the_object_it_cropped_around(device):
    """An object inside the middle half of the field of view is kept exactly."""
    image = np.zeros((1, 128), dtype=complex)
    image[0, 32:96] = np.linspace(0.2, 1.0, 64)
    kspace = torch.as_tensor(_centred_fft(image), dtype=torch.complex64, device=device)
    cropped = bartorch.remove_readout_oversampling(kspace, 64).cpu().numpy()
    recovered = _centred_fft(cropped, inverse=True)
    np.testing.assert_allclose(recovered[0], image[0, 32:96], atol=1e-5)


def test_a_partial_echo_keeps_its_object_where_it_is(device):
    """The crop is in the image domain, so an echo off the readout's centre moves nothing.

    192 of 256 samples, the echo at index 64: the centred transform then puts a
    phase ramp on the image, and the magnitude still lands on the same pixels.
    """
    samples, echo, pixels = 192, 64, (-20, 30)
    k = np.arange(samples) - echo
    readout = sum(np.exp(-2j * np.pi * k * p / samples) for p in pixels)[None]
    kspace = torch.as_tensor(readout, dtype=torch.complex64, device=device)
    cropped = bartorch.remove_readout_oversampling(kspace, samples // 2).cpu().numpy()
    magnitude = np.abs(_centred_fft(cropped, inverse=True)[0])
    centre = samples // 4
    assert sorted(np.argsort(magnitude)[-2:] - centre) == list(pixels)
    np.testing.assert_allclose(
        np.delete(magnitude, [centre + p for p in pixels]), 0, atol=1e-4 * magnitude.max()
    )


def test_a_target_wider_than_the_samples_there_are_is_refused():
    with pytest.raises(ValueError, match=r"must be in \[1, 64\]"):
        bartorch.remove_readout_oversampling(torch.ones(2, 64, dtype=torch.complex64), 128)


def test_a_target_that_matches_is_returned_untouched():
    data = torch.ones(2, 64, dtype=torch.complex64)
    assert bartorch.remove_readout_oversampling(data, 64) is data
