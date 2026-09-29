"""The pieces a learned reconstruction is assembled from, each against what it claims to compute.

The network is held to the identity it starts as and to the per-frame
network its factorised convolution reduces to; the complex layout to the
parts written out; the patchwise application to the same network applied
whole; the schedules to the iteration written out in torch; the partition
of the samples to its counts; the spread and the calibration to torch's
variance and the normal distribution's quantile.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from scipy.stats import norm
from torch import nn

from bartorch import learning, linop, optim, priors
from bartorch.learning.nets import _Conv

# --- the network --------------------------------------------------------------------


@pytest.mark.parametrize(("spatial", "shape"), [(2, (2, 4, 13, 10)), (3, (1, 6, 9, 7, 5))])
def test_an_untrained_residual_unet_is_the_identity_at_any_extent(spatial, shape):
    net = learning.UNet(shape[1], spatial=spatial, widths=(8, 16, 16))
    x = torch.randn(shape)
    assert torch.equal(net(x), x)


def test_a_unet_without_a_residual_maps_between_channel_counts():
    net = learning.UNet(2, 5, spatial=2, widths=(8, 16), residual=False)
    assert (3, 5, 12, 11) == tuple(net(torch.randn(3, 2, 12, 11)).shape)
    with pytest.raises(ValueError, match="returns 2 channels"):
        learning.UNet(2, 5, spatial=2)


def _perturbed(net):
    """``net`` with its last convolution moved off zero, so it is no longer the identity."""
    torch.manual_seed(1)
    for p in net.tail.parameters():
        nn.init.normal_(p, std=0.1)
    return net


def test_the_factorised_convolution_is_the_3d_convolution_by_the_kernels_product():
    """A spatial kernel S then a temporal one T is the (t, y, x) kernel sum_m T[o, m] S[m, i]."""
    torch.manual_seed(0)
    conv = _Conv(3, 4, spatial=2, frames=True, periodic=False)
    x = torch.randn(2, 3, 6, 9, 8)

    made = conv(x)

    S, bs = conv.space.weight, conv.space.bias
    T, bt = conv.time.weight, conv.time.bias
    kernel = torch.einsum("omt,miyx->oityx", T, S)
    # The spatial bias is spread by the temporal kernel over the frames that exist.
    reference = F.conv3d(x, kernel, bias=None, padding=1)
    edge = torch.einsum("omt,m->ot", T, bs)
    spread = bt.reshape(1, -1, 1, 1, 1) + torch.stack(
        [sum(edge[:, k] * float(0 <= t + k - 1 < 6) for k in range(3)) for t in range(6)], -1
    ).reshape(1, 4, 6, 1, 1)
    assert torch.allclose(made, reference + spread, atol=1e-5)


def test_a_single_frame_passes_the_frame_network_as_the_spatial_network():
    """With every temporal kernel (0, 1, 0) and one frame, the frame network is the 2D one."""
    torch.manual_seed(0)
    frames = _perturbed(learning.UNet(2, spatial=2, widths=(8, 16), frames=True))
    alone = learning.UNet(2, spatial=2, widths=(8, 16))
    for module in frames.modules():
        if isinstance(module, nn.Conv1d):
            with torch.no_grad():
                module.weight.zero_()
                module.bias.zero_()
                module.weight[:, :, 1] = torch.eye(module.weight.shape[0])
    alone.load_state_dict(
        {k: v for k, v in frames.state_dict().items() if ".time." not in k}, strict=True
    )
    x = torch.randn(2, 2, 1, 12, 10)
    assert torch.allclose(frames(x)[:, :, 0], alone(x[:, :, 0]), atol=1e-5)


def test_a_periodic_frame_axis_wraps_around():
    """An impulse in the first frame reaches the last only when the axis is periodic."""
    x = torch.zeros(1, 2, 6, 5, 5)
    x[:, :, 0, 2, 2] = 1.0
    for periodic in (True, False):
        conv = _Conv(2, 2, spatial=2, frames=True, periodic=periodic)
        with torch.no_grad():
            conv.space.bias.zero_()
            conv.time.bias.zero_()
        assert (float(conv(x)[:, :, -1].abs().amax()) > 0) is periodic


def test_the_conditioning_changes_the_output_and_is_held_to_what_was_declared():
    net = _perturbed(learning.UNet(2, spatial=2, widths=(8, 16), steps=True, noise=True, classes=3))
    x = torch.randn(2, 2, 8, 8)
    base = net(x, sigma=0.1, step=1, label=0)
    assert not torch.equal(base, net(x, sigma=0.1, step=2, label=0))
    assert not torch.equal(base, net(x, sigma=0.2, step=1, label=0))
    assert not torch.equal(base, net(x, sigma=0.1, step=1, label=2))
    per_item = net(x, sigma=torch.tensor([0.1, 0.2]), step=1, label=0)
    assert torch.equal(per_item[0], base[0]) and not torch.equal(per_item[1], base[1])

    with pytest.raises(ValueError, match="conditioned on step"):
        net(x, sigma=0.1, label=0)
    with pytest.raises(ValueError, match="not conditioned"):
        learning.UNet(2, spatial=2, widths=(8,))(x, step=1)


# --- complex images as channels ------------------------------------------------------


class _Keep(nn.Module):
    """The identity, remembering what it was given and with which keywords."""

    def forward(self, v, sigma=None, **keywords):
        self.planes, self.sigma, self.keywords = v.clone(), sigma, keywords
        return v


def test_the_channels_are_the_real_parts_then_the_imaginary_parts():
    x = torch.randn(2, 3, 4, 8, 8, dtype=torch.complex64)
    keep = _Keep()
    learning.ComplexNet(keep, spatial=2, channels=2, normalize=None)(x, 0.5, step=3)
    planes = keep.planes.reshape(2, 2, 3, 4, 8, 8)
    assert torch.equal(planes[:, 0], x.real) and torch.equal(planes[:, 1], x.imag)
    assert 0.5 == keep.sigma and {"step": 3} == keep.keywords


def test_a_frame_axis_is_kept_in_front_of_the_spatial_ones():
    x = torch.randn(1, 5, 7, 8, 8, dtype=torch.complex64)
    keep = _Keep()
    learning.ComplexNet(keep, spatial=2, channels=1, frames=True)(x)
    assert (1, 10, 7, 8, 8) == tuple(keep.planes.shape)


@pytest.mark.parametrize("normalize", [None, "peak", "whiten"])
def test_the_identity_network_returns_the_image_under_every_normalization(normalize):
    x = 40.0 * torch.randn(2, 3, 6, 5, 4, dtype=torch.complex64) + 3.0
    made = learning.ComplexNet(_Keep(), channels=1, normalize=normalize)(x)
    assert torch.allclose(made, x, rtol=1e-4, atol=1e-3)


def test_whitened_channels_are_uncorrelated_with_unit_variance():
    torch.manual_seed(0)
    mixing = torch.randn(4, 4)
    real = (mixing @ torch.randn(4, 16 * 16)).reshape(1, 2, 2, 16, 16)
    x = torch.complex(real[:, 0], real[:, 1]) + (1 + 2j)
    keep = _Keep()
    learning.ComplexNet(keep, spatial=2, channels=1, normalize="whiten")(x)
    flat = keep.planes.reshape(4, -1).double()
    assert torch.allclose(flat.mean(-1), torch.zeros(4, dtype=torch.float64), atol=1e-5)
    assert torch.allclose(
        torch.cov(flat, correction=0), torch.eye(4, dtype=torch.float64), atol=1e-4
    )


# --- patch by patch ------------------------------------------------------------------


class _Pointwise(nn.Module):
    """A voxel-wise network scaled by ``sigma``: its patchwise and whole applications agree."""

    def __init__(self):
        super().__init__()
        self.mix = nn.Conv3d(4, 4, 1)

    def forward(self, v, sigma=1.0):
        s = sigma.reshape(-1, 1, 1, 1, 1) if isinstance(sigma, torch.Tensor) else sigma
        return torch.tanh(self.mix(v)) * s


@pytest.mark.parametrize("shift", [False, True])
def test_patchwise_application_of_a_voxel_wise_network_is_the_whole_application(device, shift):
    torch.manual_seed(0)
    net = _Pointwise()
    whole = _Pointwise()
    whole.load_state_dict(net.state_dict())
    patchwise = learning.Patchwise(
        net, patch=(4, 5, 3), device=device, dtype=None, batch=3, shift=shift
    )
    x = torch.randn(2, 4, 9, 11, 7, requires_grad=True)
    sigma = torch.tensor([1.0, 2.0])

    made = patchwise(x, sigma)

    assert made.device == x.device
    assert torch.allclose(made, whole(x, sigma), atol=1e-6)
    made.square().sum().backward()
    reference = x.detach().requires_grad_()
    whole(reference, sigma).square().sum().backward()
    assert torch.allclose(x.grad, reference.grad, atol=1e-5)
    assert torch.allclose(net.mix.weight.grad.cpu(), whole.mix.weight.grad, rtol=1e-4, atol=1e-5)


def test_the_network_is_put_back_on_its_device_if_moved():
    net = _Pointwise()
    patchwise = learning.Patchwise(net, patch=(4, 4, 4), device="cpu")
    patchwise.to(torch.float64)
    weight = net.mix.weight
    patchwise(torch.randn(1, 4, 4, 4, 4, dtype=torch.float64), 1.0)
    assert weight is net.mix.weight


def test_mixed_precision_is_bfloat16_or_float16_on_a_card_and_none_on_the_host():
    assert learning.Patchwise(nn.Identity(), (4, 4), device="cpu").dtype is None
    if torch.cuda.is_available():
        expected = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        assert learning.Patchwise(nn.Identity(), (4, 4), device="cuda").dtype is expected


# --- schedules across iterations ------------------------------------------------------


class _Scale(nn.Module):
    """``x (1 - sigma)``, recording the sigma and the step it was called with."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, sigma, step=None):
        self.calls.append((round(float(sigma), 6), step))
        return x * (1 - sigma)


def test_the_denoiser_is_given_each_iterations_sigma_and_index():
    A = linop.FFT((8, 8), axes=(-1, -2))
    y = A(torch.randn(8, 8, dtype=torch.complex64))
    net = _Scale()
    prior = priors.ImplicitPrior(net, sigma=[0.3, 0.2, 0.1], step=True)
    learning.Unrolled(optim.ISTBlock(prior, step=0.5), iterations=4)(y, A)
    # Each iteration's image is IST's threshold once more, at that iteration's index.
    expected = [(0.3, 0), (0.2, 1), (0.1, 2), (0.1, 3)]
    assert [call for call in expected for _ in range(2)] == net.calls


def test_a_schedule_has_no_fixed_point():
    prior = priors.ImplicitPrior(_Scale(), sigma=[0.3, 0.1])
    with pytest.raises(ValueError, match="no fixed point"):
        optim.FixedPoint(optim.ISTBlock(prior))
    with pytest.raises(ValueError, match="no fixed point"):
        optim.FixedPoint(optim.ADMMBlock(priors.ImplicitPrior(_Scale(), sigma=0.1), rho=[1, 2]))


def test_admm_with_a_rho_schedule_is_the_scaled_iteration_with_its_duals_rescaled():
    """Against ADMM written out, with A^H A the identity so each x-update is exact."""
    torch.manual_seed(0)
    A = linop.FFT((8, 8), axes=(-1, -2))
    y = A(torch.randn(8, 8, dtype=torch.complex64))
    rhos = [0.5, 1.0, 4.0, 4.0]
    prior = priors.ImplicitPrior(lambda v, sigma: v * (1 - sigma), sigma=0.25)
    block = optim.ADMMBlock(prior, rho=rhos, alpha=1.0, fast=True, cg_maxiter=5)

    made = learning.Unrolled(block, iterations=4)(y, A)

    adjoint = A.adjoint(y)
    z = torch.zeros_like(adjoint)
    u = torch.zeros_like(adjoint)
    previous = rhos[0]
    for rho in rhos:
        u = u * (previous / rho)
        x = (adjoint + rho * (z - u)) / (1 + rho)
        z = (x + u) * 0.75
        u = x + u - z
        previous = rho
    assert torch.allclose(made, x, rtol=1e-4, atol=1e-5)


def test_a_constant_rho_schedule_is_the_constant_rho_to_the_bit():
    torch.manual_seed(0)
    A = linop.FFT((8, 8), axes=(-1, -2))
    y = A(torch.randn(8, 8, dtype=torch.complex64))
    prior = priors.ImplicitPrior(lambda v, sigma: v * (1 - sigma), sigma=0.25)
    a = learning.Unrolled(optim.ADMMBlock(prior, rho=0.7), iterations=3)(y, A)
    b = learning.Unrolled(optim.ADMMBlock(prior, rho=[0.7, 0.7]), iterations=3)(y, A)
    assert torch.equal(a, b)


# --- splitting the samples ------------------------------------------------------------


def test_the_split_partitions_the_acquired_samples_and_spares_the_centre():
    torch.manual_seed(0)
    pattern = (torch.rand(32, 24) < 0.4).float()
    pattern[14:18, 10:14] = 1
    generator = torch.Generator().manual_seed(3)
    keep, held = learning.split(pattern, 0.4, keep=(4, 4), generator=generator)
    assert torch.equal(keep + held, pattern)
    assert 0 == float((keep * held).sum())
    assert 0 == float(held[14:18, 10:14].sum())
    outside = float(pattern.sum()) - 16
    assert round(0.4 * outside) == int(held.sum())


def test_a_gaussian_split_holds_out_more_near_the_centre_than_a_uniform_one():
    pattern = torch.ones(64, 64)
    centre = (slice(24, 40), slice(24, 40))
    near = {}
    for density in ("gaussian", "uniform"):
        generator = torch.Generator().manual_seed(0)
        _, held = learning.split(pattern, 0.3, density=density, width=0.2, generator=generator)
        near[density] = float(held[centre].sum())
    assert near["gaussian"] > 2 * near["uniform"]


# --- uncertainty ----------------------------------------------------------------------


def test_the_running_moments_are_torchs_mean_and_variance():
    torch.manual_seed(0)
    draws = [torch.randn(5, 6, dtype=torch.complex64) * 3 + 1 for _ in range(7)]
    it = iter(draws)
    mean, variance = learning.moments(lambda: next(it), samples=7)
    stacked = torch.stack(draws)
    assert torch.allclose(mean, stacked.mean(0), atol=1e-5)
    assert torch.allclose(variance, stacked.var(0), rtol=1e-4)


def test_calibrating_unit_normal_errors_gives_the_normal_quantile():
    torch.manual_seed(0)
    error = torch.randn(200_000)
    factor = learning.calibrate(error, torch.ones_like(error), coverage=0.9)
    assert math.isclose(factor, norm.ppf(0.95), rel_tol=1e-2)
    assert math.isinf(learning.calibrate(error[:5], torch.ones(5), coverage=0.9))
