import math

import pytest
import torch

from bartorch import linop, optim, tools


def _segmented(device, planes=1, shots=3, coils=4, rows=24, columns=20, seed=0):
    generator = torch.Generator().manual_seed(seed)
    y = torch.linspace(-1, 1, rows)[:, None]
    x = torch.linspace(-1, 1, columns)[None]
    image = ((y**2 / 0.6 + x**2 / 0.5) < 1).to(torch.complex64) * torch.exp(1j * 0.3 * x)
    image = image.expand(planes, rows, columns).contiguous()
    maps = torch.stack(
        [
            torch.exp(-((y - math.cos(c)) ** 2 + (x - math.sin(c)) ** 2) / 1.5)
            * torch.exp(1j * torch.tensor(0.5 * c))
            for c in range(coils)
        ]
    ).to(torch.complex64)
    maps = (maps / maps.abs().square().sum(0).sqrt())[:, None].expand(-1, planes, -1, -1)
    masks = torch.zeros(shots, planes, rows, dtype=torch.bool)
    for shot in range(shots):
        masks[shot, :, shot::shots] = True
    basis = tools.shot_phase_basis((planes, rows, columns), 1)
    weights = 0.4 * torch.randn(shots, basis.shape[0], generator=generator)
    weights[0] = 0
    phases = torch.einsum("st,tzyx->szyx", weights, basis)
    encoding = linop.CartesianSense(maps.contiguous(), (planes, rows, columns), device=device)
    if planes == 1:
        encoding = linop.CartesianSense(maps[:, 0].contiguous(), (rows, columns), device=device)
    kspace = torch.zeros(coils, planes, rows, columns, dtype=torch.complex64)
    for shot in range(shots):
        read = encoding(
            (torch.exp(1j * phases[shot]) * image).reshape(encoding.ishape).to(torch.complex64)
        ).reshape(kspace.shape)
        kspace = kspace + read.cpu() * masks[shot][..., None]
    return kspace, maps.contiguous(), masks, image, phases, basis, encoding


@pytest.mark.parametrize("planes", [1, 2])
def test_shot_phases_fitted_against_the_true_image_are_the_true_phases(device, planes):
    kspace, maps, masks, image, truth, basis, _ = _segmented(device, planes=planes)
    start = torch.zeros_like(truth)
    fitted = tools.estimate_shot_phases(kspace, maps, masks, image, start, basis, steps=4)
    inside = image.abs() > 0
    assert float((fitted - truth)[:, inside].abs().max()) < 1e-3


def test_the_residual_vanishes_under_the_true_shot_phases(device):
    kspace, maps, masks, image, truth, *_ = _segmented(device)
    assert tools.shot_residual(kspace, maps, masks, image, truth) < 1e-5
    assert tools.shot_residual(kspace, maps, masks, image, torch.zeros_like(truth)) > 0.05


def test_the_shot_encoding_reproduces_the_segmented_k_space(device):
    kspace, maps, masks, image, truth, _, encoding = _segmented(device)
    segmented = tools.shot_encoding(encoding, masks, torch.exp(1j * truth))
    read = segmented(image.reshape(segmented.ishape).to(torch.complex64)).cpu()
    assert torch.allclose(read.reshape(kspace.shape), kspace, atol=1e-5)


def test_cg_through_the_shot_encoding_recovers_the_image(device):
    kspace, maps, masks, image, truth, _, encoding = _segmented(device)
    segmented = tools.shot_encoding(encoding, masks, torch.exp(1j * truth))
    solved = optim.CG(maxiter=60)(kspace.reshape(segmented.oshape), segmented).cpu()
    error = (solved.reshape(image.shape) - image).norm() / image.norm()
    assert float(error) < 1e-3


def test_a_negative_basis_order_is_refused():
    with pytest.raises(ValueError, match="order"):
        tools.shot_phase_basis((1, 8, 8), -1)
