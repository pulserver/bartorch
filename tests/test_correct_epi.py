"""EPI ramp resampling and odd/even phase correction."""

import numpy as np
import pytest
import torch

import bartorch.tools as bt


def _fftc(x, axes=(-2, -1)):
    """Centred unitary DFT, written with numpy."""
    axes = (axes,) if isinstance(axes, int) else axes
    x = np.fft.ifftshift(x, axes=axes)
    return np.fft.fftshift(np.fft.fftn(x, axes=axes, norm="ortho"), axes=axes)


def _ifftc(x, axes=(-2, -1)):
    axes = (axes,) if isinstance(axes, int) else axes
    x = np.fft.ifftshift(x, axes=axes)
    return np.fft.fftshift(np.fft.ifftn(x, axes=axes, norm="ortho"), axes=axes)


def _numpy(x):
    return x.detach().cpu().numpy()


@pytest.fixture
def ramp_sampled():
    """A readout swept sinusoidally, and the uniform grid it belongs on."""
    samples, support = 64, 16
    sweep = np.sin(np.linspace(-np.pi / 2, np.pi / 2, samples)) / 2.0
    uniform = np.linspace(-0.5, 0.5, samples)
    return sweep, uniform, support


def _band_limited(positions, support, rng):
    """The non-uniform transform, at ``positions``, of an object ``support`` wide."""
    grid = np.arange(support) - support // 2
    obj = rng.normal(size=support) + 1j * rng.normal(size=support)
    return np.exp(-2j * np.pi * np.outer(positions, grid)) @ obj, obj


def test_ramp_resampling_is_exact_for_a_band_limited_readout(ramp_sampled):
    sweep, uniform, support = ramp_sampled
    rng = np.random.default_rng(0)
    taken, obj = _band_limited(sweep, support, rng)
    grid = np.arange(support) - support // 2
    wanted = np.exp(-2j * np.pi * np.outer(uniform, grid)) @ obj

    operator = _numpy(
        bt.epi_ramp_operator(torch.as_tensor(sweep), torch.as_tensor(uniform), support)
    )
    resampled = operator @ taken
    error = np.linalg.norm(resampled - wanted) / np.linalg.norm(wanted)
    assert error < 1e-3


def test_ramp_resampling_beats_linear_interpolation(ramp_sampled):
    sweep, uniform, support = ramp_sampled
    rng = np.random.default_rng(1)
    taken, obj = _band_limited(sweep, support, rng)
    grid = np.arange(support) - support // 2
    wanted = np.exp(-2j * np.pi * np.outer(uniform, grid)) @ obj

    def error(estimate):
        return np.linalg.norm(estimate - wanted) / np.linalg.norm(wanted)

    operator = _numpy(
        bt.epi_ramp_operator(torch.as_tensor(sweep), torch.as_tensor(uniform), support)
    )
    linear = np.interp(uniform, sweep, taken.real) + 1j * np.interp(uniform, sweep, taken.imag)
    assert error(operator @ taken) < 0.1 * error(linear)


def test_the_operator_is_built_once_and_applies_to_a_whole_train(ramp_sampled):
    sweep, uniform, support = ramp_sampled
    operator = bt.epi_ramp_operator(torch.as_tensor(sweep), torch.as_tensor(uniform), support)
    assert operator.shape == (uniform.size, sweep.size)
    train = torch.ones(8, sweep.size, dtype=torch.complex64)
    assert (train @ operator.T).shape == (8, uniform.size)


def test_positions_that_do_not_describe_a_readout_are_refused():
    with pytest.raises(ValueError, match="describe a readout"):
        bt.epi_ramp_operator(torch.tensor([0.0]), torch.tensor([0.0, 0.1]), 4)


def test_a_non_positive_support_is_refused():
    positions = torch.linspace(-0.5, 0.5, 8)
    with pytest.raises(ValueError, match="support must be positive"):
        bt.epi_ramp_operator(positions, positions, 0)


def _navigator_line(*, coils=4, samples=64, width=1.0):
    """A blip-nulled navigator line, broad in hybrid space."""
    profile = np.exp(-(np.linspace(-width, width, samples) ** 2)).astype(complex)
    line = _fftc(profile, axes=-1)
    return np.repeat(line[None], coils, axis=0).astype(np.complex64)


def _delayed(line, coefficients):
    """A phase polynomial on a line, in hybrid space, as a delay puts one."""
    hybrid = _ifftc(line, axes=-1)
    coordinate = np.linspace(-1.0, 1.0, hybrid.shape[-1])
    ramp = np.polynomial.polynomial.polyval(coordinate, coefficients)
    return _fftc(hybrid * np.exp(1j * ramp), axes=-1)


def _tensors(lines, device="cpu"):
    return [
        torch.as_tensor(np.ascontiguousarray(line), dtype=torch.complex64, device=device)
        for line in lines
    ]


def test_the_navigator_fit_is_the_correction_not_the_impressed_phase():
    """The fit is what ``correct_lines`` adds to a reversed line: minus its phase."""
    line = _navigator_line()
    impressed = [0.3, 0.8]
    fit = _numpy(bt.estimate_epi_phase(_tensors([line, _delayed(line, impressed), line])))
    assert fit.shape == (2,)
    np.testing.assert_allclose(fit, [-value for value in impressed], atol=0.05)


@pytest.mark.parametrize("coils", [1, 4])
def test_the_fit_undoes_the_phase_it_measured(coils):
    line = _navigator_line(coils=coils)
    reversed_line = _delayed(line, [0.3, 0.8])
    fit = bt.estimate_epi_phase(_tensors([line, reversed_line, line]))
    corrected = bt.correct_lines([(_tensors([reversed_line[..., ::-1]])[0], True)], fit)[0]
    np.testing.assert_allclose(_numpy(corrected), line, atol=1e-2)


def test_a_higher_order_fit_returns_the_coefficients_it_was_asked_for():
    line = _navigator_line(coils=2)
    lines = _tensors([line, _delayed(line, [0.1, 0.5]), line])
    assert bt.estimate_epi_phase(lines, polynomial_order=3).shape == (4,)


def test_fewer_than_three_navigator_lines_is_refused():
    line = torch.ones(2, 16, dtype=torch.complex64)
    with pytest.raises(ValueError, match="three lines"):
        bt.estimate_epi_phase([line, line])


def test_a_negative_polynomial_order_is_refused():
    line = torch.ones(2, 16, dtype=torch.complex64)
    with pytest.raises(ValueError, match="non-negative"):
        bt.estimate_epi_phase([line] * 3, polynomial_order=-1)


def test_correcting_the_train_removes_the_ghost():
    """An uncorrected odd/even phase puts a ghost half a field of view away."""
    truth = np.zeros((32, 32))
    truth[10:22, 12:20] = 1.0
    kspace = _fftc(truth.astype(complex))
    coefficients = [0.4, 0.9]

    def backwards(line):
        return _delayed(line, coefficients)[..., ::-1]

    train = [
        (line[None], False) if index % 2 == 0 else (backwards(line[None]), True)
        for index, line in enumerate(kspace)
    ]
    train = [(_tensors([data])[0], flag) for data, flag in train]
    middle = kspace[16][None]
    fit = bt.estimate_epi_phase(_tensors([middle, backwards(middle)[..., ::-1], middle]))

    def placed(phase):
        return np.stack([_numpy(bt.correct_lines([line], phase)[0])[0] for line in train])

    def ghost(image):
        return np.linalg.norm(np.abs(image)[:8]) / np.linalg.norm(np.abs(image))

    assert ghost(_ifftc(placed(fit))) < 0.5 * ghost(_ifftc(placed(None)))


def test_a_forward_line_passes_through_untouched():
    line = torch.as_tensor((np.arange(16) + 1j).astype(np.complex64)[None])
    torch.testing.assert_close(bt.correct_lines([(line, False)])[0], line)


def test_a_reversed_line_is_flipped_even_without_a_phase(device):
    line = torch.as_tensor((np.arange(16) + 1j).astype(np.complex64)[None], device=device)
    corrected = bt.correct_lines([(line, True)])[0]
    assert corrected.device == line.device
    torch.testing.assert_close(corrected, torch.flip(line, [-1]))


def _ghosted_train(n=64, slope=0.6, offset=0.25):
    """An EPI train carrying an odd/even phase, and the navigator that sees it."""
    y, x = np.mgrid[-1 : 1 : n * 1j, -1 : 1 : n * 1j]
    image = (((x / 0.5) ** 2 + (y / 0.7) ** 2) <= 1).astype(complex)
    hybrid = _ifftc(_fftc(image), axes=-1)
    readout = np.linspace(-1.0, 1.0, n)

    train = []
    for line in range(n):
        backwards = line % 2 == 1
        sign = -1.0 if backwards else 1.0
        row = hybrid[line] * np.exp(1j * sign * (slope * readout + offset))
        encoded = _fftc(row[None], axes=-1)
        stored = np.ascontiguousarray(encoded[:, ::-1]) if backwards else encoded
        train.append((stored, backwards))

    centre = hybrid[n // 2]
    forward = _fftc((centre * np.exp(1j * (slope * readout + offset)))[None], axes=-1)
    backward = _fftc((centre * np.exp(-1j * (slope * readout + offset)))[None], axes=-1)
    return image, train, [forward, backward, forward]


def test_the_ramp_operator_is_on_the_device_of_its_positions(device):
    sampled = np.sin(np.linspace(-np.pi / 2, np.pi / 2, 24)) * 0.5
    uniform = np.linspace(-0.5, 0.5, 24)
    operator = bt.epi_ramp_operator(
        torch.as_tensor(sampled, device=device), torch.as_tensor(uniform, device=device), 8
    )
    assert operator.device.type == device
    reference = bt.epi_ramp_operator(torch.as_tensor(sampled), torch.as_tensor(uniform), 8)
    np.testing.assert_allclose(_numpy(operator), _numpy(reference), atol=1e-5)


def test_the_phase_fit_is_the_same_rotation_on_every_device(device):
    """The constant is fixed only modulo 2 pi, so it is compared as a rotation."""
    _, _, navigator = _ghosted_train()
    reference = _numpy(bt.estimate_epi_phase(_tensors(navigator)))
    fit = bt.estimate_epi_phase(_tensors(navigator, device))
    assert fit.device.type == device
    fit = _numpy(fit)
    np.testing.assert_allclose(fit[1:], reference[1:], atol=1e-4)
    np.testing.assert_allclose(np.exp(1j * fit[0]), np.exp(1j * reference[0]), atol=1e-4)


def test_correcting_a_train_leaves_no_signal_outside_the_object(device):
    image, train, navigator = _ghosted_train()
    moved = [(_tensors([data], device)[0], backwards) for data, backwards in train]
    fit = bt.estimate_epi_phase(_tensors(navigator, device))

    corrected = bt.correct_lines(moved, fit)
    assert corrected[0].device.type == device
    estimate = _ifftc(np.concatenate([_numpy(line) for line in corrected], axis=0))
    background = np.abs(image) < 1e-6
    assert np.abs(estimate)[background].max() / np.abs(estimate).max() < 1e-3
