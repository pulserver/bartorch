"""The training stages, held to the losses they are defined by and to the order they run in.

The self-supervised loss is held to SSDU's written out over torch's own FFT;
the greedy weights to the geometric progression they are; the augmentation to
one complex multiplication shared by the images of a subject.
"""

import math

import pytest
import torch
from torch import nn

lightning = pytest.importorskip("lightning")
torchio = pytest.importorskip("torchio")

from bartorch import learning, linop, optim, priors  # noqa: E402
from bartorch.learning import training  # noqa: E402

SHAPE = (1, 16, 16)


class _Gain(nn.Module):
    """``x`` times a learned complex gain, which a denoiser stage can fit in a few steps."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor([0.2, 0.0]))

    def forward(self, x, sigma=None, step=None):
        return x * torch.complex(self.gain[0], self.gain[1])


def _trainer(**settings):
    return lightning.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **settings,
    )


def test_the_greedy_weights_grow_geometrically_to_ratio_and_sum_to_one():
    block = optim.ISTBlock(priors.ImplicitPrior(_Gain()))
    stage = training.Reconstruction(
        learning.Unrolled(block, iterations=6, detach=True), "greedy", ratio=10.0
    )
    weights = stage._weights()
    assert math.isclose(math.fsum(weights), 1.0)
    assert math.isclose(weights[-1] / weights[0], 10.0)
    ratios = [b / a for a, b in zip(weights, weights[1:])]
    assert all(math.isclose(r, ratios[0]) for r in ratios)


def test_greedy_training_needs_a_detached_stack_and_a_block_whose_image_it_can_reach():
    prior = priors.ImplicitPrior(_Gain())
    with pytest.raises(ValueError, match="detach=True"):
        training.Reconstruction(learning.Unrolled(optim.ISTBlock(prior), 3), "greedy")

    A = linop.FFT(SHAPE, axes=(-1, -2))
    x = torch.randn(*SHAPE, dtype=torch.complex64)
    admm = learning.Unrolled(optim.ADMMBlock(prior, cg_maxiter=2), 3, detach=True)
    stage = training.Reconstruction(admm, "greedy")
    with pytest.raises(ValueError, match="its own denoiser"):
        # manual_backward is Lightning's; outside a trainer the plain backward stands in.
        stage.manual_backward = lambda loss: loss.backward()
        stage._backward({"y": A(x), "A": A, "target": x})


def test_the_batch_stays_where_the_dataset_put_it():
    stage = training.Reconstruction(_Gain(), "denoiser")
    batch = [{"input": torch.ones(2)}]
    assert stage.transfer_batch_to_device(batch, torch.device("meta"), 0) is batch


def test_the_denoiser_stage_fits_the_network_to_its_pairs():
    torch.manual_seed(0)
    images = [torch.randn(*SHAPE, dtype=torch.complex64) for _ in range(4)]
    pairs = [{"input": x, "target": (0.5 - 0.5j) * x} for x in images]
    loader = torch.utils.data.DataLoader(pairs, batch_size=2, collate_fn=list)
    net = _Gain()
    _trainer(max_epochs=60).fit(training.Reconstruction(net, "denoiser", lr=0.05), loader, loader)
    assert torch.allclose(net.gain.detach(), torch.tensor([0.5, -0.5]), atol=2e-2)


def test_the_self_supervised_loss_is_ssdus_on_the_held_out_samples():
    """Held to the reconstruct-from-part, compare-on-the-rest loss written out over torch.fft."""
    torch.manual_seed(0)
    pattern = (torch.rand(*SHAPE) < 0.6).float()
    A = linop.Diagonal(pattern.to(torch.complex64), SHAPE) @ linop.FFT(SHAPE, axes=(-1, -2))
    y = A(torch.randn(*SHAPE, dtype=torch.complex64))

    class _Zero(nn.Module):
        def forward(self, y, A, x0=None):
            return A.adjoint(y)

    stage = training.Reconstruction(_Zero(), "end-to-end")
    loss = stage._loss({"y": y, "A": A, "pattern": pattern}, validating=True)

    keep, held = learning.split(pattern, 0.4, generator=torch.Generator().manual_seed(0))
    image = torch.fft.ifft2(keep * y, norm="ortho")
    residual = held * torch.fft.fft2(image, norm="ortho") - held * y
    reference = held * y
    expected = (
        residual.abs().square().sum().sqrt() / reference.abs().square().sum().sqrt()
        + residual.abs().sum() / reference.abs().sum()
    )
    assert torch.allclose(loss, expected, rtol=1e-4)


def test_the_unrolled_stages_train_in_a_trainer():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    items = []
    for _ in range(2):
        x = torch.randn(*SHAPE, dtype=torch.complex64)
        items.append({"y": A(x), "A": A, "target": x})
    loader = torch.utils.data.DataLoader(items, batch_size=2, collate_fn=list)
    prior = priors.ImplicitPrior(_Gain())
    for stage, model in (
        ("greedy", learning.Unrolled(optim.ISTBlock(prior, step=0.5), 3, detach=True)),
        ("end-to-end", learning.Unrolled(optim.ISTBlock(prior, step=0.5), 3, checkpoint=True)),
    ):
        before = prior.denoiser.gain.detach().clone()
        trainer = _trainer(max_epochs=2)
        trainer.fit(training.Reconstruction(model, stage, accumulate=2), loader, loader)
        assert "val_loss" in trainer.callback_metrics
        assert not torch.equal(before, prior.denoiser.gain.detach())


def test_a_random_gain_multiplies_every_image_of_the_subject_by_the_same_complex_number():
    torch.manual_seed(0)
    a = torch.randn(3, 4, 5, 6, dtype=torch.complex64)
    b = torch.randn(3, 4, 5, 6, dtype=torch.complex64)
    subject = torchio.Subject(
        a=torchio.ScalarImage(tensor=learning.as_real(a).reshape(6, 4, 5, 6)),
        b=torchio.ScalarImage(tensor=learning.as_real(b).reshape(6, 4, 5, 6)),
    )
    made = training.RandomGain(log_scale=0.3)(subject)
    ma = learning.as_complex(made["a"].data.reshape(2, 3, 4, 5, 6))
    mb = learning.as_complex(made["b"].data.reshape(2, 3, 4, 5, 6))
    gain = (ma / a).flatten()
    assert torch.allclose(gain, gain[0].expand_as(gain), atol=1e-4)
    assert torch.allclose(mb, gain[0] * b, atol=1e-4)
