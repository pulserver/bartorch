"""Off-resonance deblurring of spiral readouts."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch

from bartorch.tools import ReadoutTiming, deblur, fit_transfer

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def variable_density_arm(samples: int = 1200, power: float = 0.6) -> np.ndarray:
    """A centre-out arm whose radius grows as a fractional power of time."""
    time = np.linspace(0.0, 1.0, samples)
    radius = time**power
    angle = 40 * np.pi * time
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


@pytest.fixture(scope="module")
def timing() -> ReadoutTiming:
    return ReadoutTiming.from_trajectory(variable_density_arm(), duration=6e-3)


def phantom(size: int) -> np.ndarray:
    """Ellipses on a square grid."""
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    image = np.zeros((size, size))
    for row, column, height, width, value in (
        (0.0, 0.0, 0.70, 0.55, 1.0),
        (0.0, 0.0, 0.65, 0.50, -0.8),
        (-0.2, 0.0, 0.20, 0.30, 0.4),
        (0.3, -0.25, 0.15, 0.10, 0.6),
    ):
        inside = ((rows - row) / height) ** 2 + ((columns - column) / width) ** 2
        image[inside <= 1] += value
    return image


def blur(image: np.ndarray, timing: ReadoutTiming, field: np.ndarray):
    """Blur exactly, by summing over the distinct values of a quantised field map."""
    size = image.shape[0]
    axis = np.fft.fftfreq(size) * 2
    squared = axis[:, None] ** 2 + axis[None, :] ** 2
    disc = squared <= 1.0
    times = np.interp(np.clip(squared, 0, 1), timing.squared_radius, timing.times)
    ideal = np.fft.ifft2(np.fft.fft2(image) * disc)
    blurred = np.zeros_like(ideal)
    for value in np.unique(field):
        selected = np.fft.fft2(image * (field == value)) * disc
        phase = np.exp(2j * np.pi * value * timing.duration * times)
        blurred += np.fft.ifft2(selected * phase)
    return ideal, blurred


def relative(estimate, truth) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def test_out_in_arm_is_rejected() -> None:
    outward = variable_density_arm(600)
    out_in = np.concatenate([outward, outward[::-1]])
    with pytest.raises(ValueError, match="not monotonic"):
        ReadoutTiming.from_trajectory(out_in, duration=6e-3)


def test_inward_arm_is_accepted() -> None:
    inward = variable_density_arm(600)[::-1]
    derived = ReadoutTiming.from_trajectory(inward, duration=6e-3)
    assert derived.times[0] > derived.times[-1]


def test_more_terms_reduce_fit_error(timing: ReadoutTiming) -> None:
    errors = [fit_transfer(timing, band=120.0, terms=n).error(timing) for n in (2, 4, 6)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_constant_offset_is_corrected_to_the_fit_error(timing: ReadoutTiming) -> None:
    size = 128
    image = phantom(size)
    offset = 90.0
    ideal, blurred = blur(image, timing, np.full((size, size), offset))
    transfer = fit_transfer(timing, band=120.0, terms=6)

    corrected = deblur(
        torch.from_numpy(blurred),
        torch.full((size, size), offset, dtype=torch.float64),
        transfer,
    ).numpy()

    assert relative(blurred, ideal) > 0.2
    # A constant offset is a plain convolution, so nothing but the factorization
    # error stands between the correction and the truth.
    assert relative(corrected, ideal) < 5 * transfer.error(timing) + 1e-3


def test_varying_field_is_mostly_corrected(timing: ReadoutTiming) -> None:
    size = 128
    image = phantom(size)
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    smooth = 70 * (0.6 * columns + 0.4 * rows)
    levels = np.linspace(smooth.min(), smooth.max(), 24)
    field = levels[np.abs(smooth[..., None] - levels).argmin(-1)]

    ideal, blurred = blur(image, timing, field)
    transfer = fit_transfer(timing, band=80.0, terms=6)
    corrected = deblur(torch.from_numpy(blurred), torch.from_numpy(field), transfer).numpy()

    assert relative(corrected, ideal) < 0.5 * relative(blurred, ideal)


def test_deblur_rejects_a_real_image(timing: ReadoutTiming) -> None:
    transfer = fit_transfer(timing, band=80.0, terms=4)
    with pytest.raises(ValueError, match="complex"):
        deblur(torch.zeros(8, 8), torch.zeros(8, 8), transfer)


def test_deblur_rejects_a_mismatched_field_map(timing: ReadoutTiming) -> None:
    transfer = fit_transfer(timing, band=80.0, terms=4)
    with pytest.raises(ValueError, match="does not match"):
        deblur(torch.zeros(8, 8, dtype=torch.complex64), torch.zeros(4, 4), transfer)


@requires_cuda
def test_cuda_matches_the_reference(timing: ReadoutTiming) -> None:
    size = 64
    torch.manual_seed(0)
    image = torch.randn(size, size, size, dtype=torch.complex64)
    field = torch.rand(size, size, size) * 200 - 100
    transfer = fit_transfer(timing, band=120.0, terms=6)

    reference = deblur(image, field, transfer, backend="fft")
    fused = deblur(image.cuda(), field.cuda(), transfer, backend="fft").cpu()
    assert (fused - reference).abs().max() < 1e-4 * reference.abs().max()


@requires_cuda
def test_convolution_backend_matches_the_transform(timing: ReadoutTiming) -> None:
    size = 64
    torch.manual_seed(0)
    image = torch.randn(size, size, size, dtype=torch.complex64, device="cuda")
    field = torch.rand(size, size, size, device="cuda") * 200 - 100
    transfer = fit_transfer(timing, band=120.0, terms=6)

    exact = deblur(image, field, transfer, backend="fft")
    truncated = deblur(image, field, transfer, backend="conv")
    assert (truncated - exact).abs().max() < 0.02 * exact.abs().max()


@requires_cuda
def test_peak_memory_does_not_grow_with_terms(timing: ReadoutTiming) -> None:
    size = 96
    image = torch.randn(size, size, size, dtype=torch.complex64, device="cuda")
    field = torch.zeros(size, size, size, device="cuda")

    peaks = []
    for terms in (4, 10):
        transfer = fit_transfer(timing, band=120.0, terms=terms)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        deblur(image, field, transfer, backend="fft")
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    assert peaks[1] <= 1.05 * peaks[0]


def test_amplification_climbs_with_the_accrued_phase(timing: ReadoutTiming) -> None:
    """Cancelling weights, not a poor fit, are what break a long readout."""
    short = ReadoutTiming.from_trajectory(variable_density_arm(), duration=8e-3)
    long = ReadoutTiming.from_trajectory(variable_density_arm(), duration=30e-3)
    lightly = fit_transfer(short, band=260.0, terms=40)
    heavily = fit_transfer(long, band=260.0, terms=40)
    assert heavily.amplification > 3 * lightly.amplification


def test_the_term_weights_interpolate_the_table_and_clamp_at_its_ends() -> None:
    from bartorch.tools._correct._spiral import SpiralTransfer, _term_weights

    frequencies = np.linspace(-100.0, 100.0, 5)
    table = np.array([1.0, 2.0 - 1.0j, 3.0, 4.0 + 2.0j, 5.0 + 1.0j])
    transfer = SpiralTransfer(np.zeros(1, dtype=complex), table[None], frequencies)
    field = np.array([-150.0, -100.0, -30.0, 0.0, 70.0, 99.0, 100.0, 150.0])

    weights = _term_weights(transfer, 0, torch.as_tensor(field)).numpy()

    expected = np.interp(field, frequencies, table.real) + 1j * np.interp(
        field, frequencies, table.imag
    )
    np.testing.assert_allclose(weights, expected, atol=1e-12)


def test_on_the_host_the_automatic_backend_is_the_transform(timing: ReadoutTiming) -> None:
    transfer = fit_transfer(timing, band=80.0, terms=4)
    torch.manual_seed(0)
    image = torch.randn(16, 16, dtype=torch.complex64)
    field = torch.rand(16, 16) * 160 - 80
    assert torch.equal(
        deblur(image, field, transfer), deblur(image, field, transfer, backend="fft")
    )


def test_an_unknown_backend_is_refused(timing: ReadoutTiming) -> None:
    transfer = fit_transfer(timing, band=80.0, terms=2)
    with pytest.raises(ValueError, match="backend must be"):
        deblur(torch.zeros(8, 8, dtype=torch.complex64), torch.zeros(8, 8), transfer, backend="x")


def test_a_trajectory_that_is_not_samples_by_axes_is_refused() -> None:
    with pytest.raises(ValueError, match=r"expected \(samples, ndim\)"):
        ReadoutTiming.from_trajectory(np.zeros(16), duration=1e-3)


def test_a_trajectory_that_stays_at_the_centre_is_refused() -> None:
    with pytest.raises(ValueError, match="zero extent"):
        ReadoutTiming.from_trajectory(np.zeros((16, 2)), duration=1e-3)


def test_a_factorization_without_terms_is_refused(timing: ReadoutTiming) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        fit_transfer(timing, band=80.0, terms=0)


# --- the Triton kernels ------------------------------------------------------------
#
# On a card these compile; without one they run under Triton's interpreter,
# which is read when a kernel is defined, so ``TRITON_INTERPRET=1`` has to be in
# the environment before the suite starts.

_interpreted = os.environ.get("TRITON_INTERPRET") == "1"
requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None or not (torch.cuda.is_available() or _interpreted),
    reason="needs Triton, and a CUDA device or TRITON_INTERPRET=1",
)
_kernel_device = "cuda" if torch.cuda.is_available() and not _interpreted else "cpu"


@requires_triton
def test_the_separable_convolution_is_the_k_space_factor_to_its_truncation() -> None:
    from bartorch.tools._correct._triton_spiral import separable_convolve

    torch.manual_seed(0)
    image = torch.randn(3, 40, 64, dtype=torch.complex64, device=_kernel_device)
    rate = complex(-4.0, 6.0)

    convolved = separable_convolve(image, rate, (1, 2))

    ky = torch.fft.fftfreq(40, device=_kernel_device)[:, None] * 2
    kx = torch.fft.fftfreq(64, device=_kernel_device)[None, :] * 2
    factor = torch.exp(torch.as_tensor(rate) * (ky**2 + kx**2)).to(torch.complex64)
    exact = torch.fft.ifft2(torch.fft.fft2(image) * factor)
    # Each axis drops at most a hundred-thousandth of its kernel's energy.
    assert float((convolved - exact).norm() / exact.norm()) < 1e-2


@requires_triton
def test_a_one_dimensional_image_is_convolved_along_its_only_axis() -> None:
    from bartorch.tools._correct._triton_spiral import separable_convolve

    torch.manual_seed(0)
    line = torch.randn(64, dtype=torch.complex64, device=_kernel_device)
    rate = complex(-2.0, 10.0)
    k = torch.fft.fftfreq(64, device=_kernel_device) * 2
    exact = torch.fft.ifft(torch.fft.fft(line) * torch.exp(torch.as_tensor(rate) * k**2))
    convolved = separable_convolve(line, rate, (0,))
    assert float((convolved - exact).norm() / exact.norm()) < 1e-2


@requires_triton
def test_a_kernel_that_does_not_fit_inside_its_axis_is_refused() -> None:
    from bartorch.tools._correct._triton_spiral import separable_convolve

    image = torch.ones(8, 8, dtype=torch.complex64, device=_kernel_device)
    with pytest.raises(ValueError, match="use backend='fft'"):
        separable_convolve(image, complex(-200.0, 0.0), (0, 1))
    with pytest.raises(ValueError, match="complex"):
        separable_convolve(image.real, complex(-1.0, 0.0), (0, 1))


@requires_triton
def test_the_fused_accumulation_adds_the_interpolated_weight_times_the_term() -> None:
    from bartorch.tools._correct._triton_spiral import accumulate_weighted

    torch.manual_seed(0)
    frequencies = np.linspace(-100.0, 100.0, 9)
    table = torch.randn(9, dtype=torch.complex64, device=_kernel_device)
    field = torch.rand(6, 7, device=_kernel_device) * 260 - 130
    working = torch.randn(2, 6, 7, dtype=torch.complex64, device=_kernel_device)
    result = torch.randn(2, 6, 7, dtype=torch.complex64, device=_kernel_device)
    before = result.clone()

    accumulate_weighted(
        result, working, field, table, frequencies[0], frequencies[1] - frequencies[0]
    )

    grid, values = frequencies, table.cpu().numpy()
    at = field.cpu().numpy()
    weight = np.interp(at, grid, values.real) + 1j * np.interp(at, grid, values.imag)
    expected = before.cpu().numpy() + weight * working.cpu().numpy()
    np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


@requires_triton
def test_the_convolution_backend_deblurs_as_the_transform_does(timing: ReadoutTiming) -> None:
    transfer = fit_transfer(timing, band=120.0, terms=6)
    torch.manual_seed(0)
    image = torch.randn(64, 64, dtype=torch.complex64, device=_kernel_device)
    field = torch.rand(64, 64, device=_kernel_device) * 200 - 100

    exact = deblur(image, field, transfer, backend="fft")
    truncated = deblur(image, field, transfer, backend="conv")
    assert float((truncated - exact).norm() / exact.norm()) < 2e-2
