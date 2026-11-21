import math

import pytest
import torch

import bartorch
from bartorch import linop, tools

GRID = (24, 24)


def _object(grid, shift=0.0):
    y, x = torch.meshgrid(*(torch.arange(n) - n // 2 for n in grid), indexing="ij")
    disc = ((y / 8.0) ** 2 + ((x - shift) / 6.0) ** 2 < 1).to(torch.complex64)
    return disc + 0.5 * (((y - 3) / 2.0) ** 2 + ((x + shift) / 3.0) ** 2 < 1)


def _maps(grid, coils, phase=0.0):
    y, x = torch.meshgrid(*(torch.arange(n) / n for n in grid), indexing="ij")
    angles = 2 * math.pi * torch.arange(coils) / coils
    rows = [
        torch.exp(
            -(((y - 0.5 - 0.4 * math.sin(a)) ** 2) + (x - 0.5 - 0.4 * math.cos(a)) ** 2) / 0.2
        )
        * torch.exp(1j * (a + phase) * (x + y))
        for a in angles
    ]
    return torch.stack(rows).to(torch.complex64)


def _kspace(image, maps):
    coils = maps * image
    return bartorch.fft(coils, (-1, -2), unitary=True).reshape(coils.shape)


def test_smooth_coil_kspace_is_consistent_with_the_spirit_kernel_of_its_centre(
    device,
):
    kspace = _kspace(_object(GRID), _maps(GRID, 8)).to(device)
    centre = kspace[:, 6:18, 6:18]

    inconsistency = linop.spirit_consistency(tools.spirit_kernel(centre, 5), GRID)

    residual = torch.linalg.vector_norm(inconsistency(kspace)) / torch.linalg.vector_norm(kspace)
    assert residual < 0.05


def test_the_spirit_inconsistency_adjoint_satisfies_the_inner_product_identity(device):
    generator = torch.Generator().manual_seed(0)
    kernel = torch.randn(4, 4, 3, 3, dtype=torch.complex64, generator=generator).to(device)
    operator = linop.spirit_consistency(kernel, (8, 10))
    x = torch.randn(4, 8, 10, dtype=torch.complex64, generator=generator).to(device)
    y = torch.randn(4, 8, 10, dtype=torch.complex64, generator=generator).to(device)

    forward = torch.vdot(operator(x).flatten(), y.flatten())
    backward = torch.vdot(x.flatten(), operator.H(y).flatten())

    assert forward == pytest.approx(backward, rel=1e-4)


def test_the_spirit_kernel_applied_in_the_image_domain_is_its_cyclic_convolution_in_kspace():
    generator = torch.Generator().manual_seed(1)
    kernel = torch.randn(3, 3, 3, 3, dtype=torch.complex64, generator=generator)
    kspace = torch.randn(3, 8, 8, dtype=torch.complex64, generator=generator)

    applied = linop.spirit_consistency(kernel, (8, 8))(kspace) + kspace

    direct = torch.zeros_like(kspace)
    for dy in range(3):
        for dx in range(3):
            shifted = torch.roll(kspace, shifts=(1 - dy, 1 - dx), dims=(1, 2))
            direct += torch.einsum("cd,dyx->cyx", kernel[:, :, dy, dx], shifted)
    assert torch.allclose(applied, direct, atol=1e-4)


def test_split_slice_grappa_separates_two_blipped_caipi_bands_it_was_calibrated_on():
    coils, lines = 8, GRID[0]
    images = [_object(GRID, 0.0), _object(GRID, 4.0)]
    maps = [_maps(GRID, coils, 0.0), _maps(GRID, coils, 1.5)]
    bands = torch.stack([_kspace(i, m) for i, m in zip(images, maps)])
    caipi = torch.stack([torch.ones(lines), (-1.0) ** torch.arange(lines)]).to(torch.complex64)
    collapsed = (bands * caipi[:, None, :, None]).sum(dim=0)

    kernels = tools.slice_grappa_kernel(bands, caipi)
    separated = tools.slice_grappa(collapsed, kernels)

    expected = bands * caipi[:, None, :, None]
    inner = (slice(None), slice(None), slice(2, -2), slice(2, -2))
    error = torch.linalg.vector_norm(separated[inner] - expected[inner]) / torch.linalg.vector_norm(
        expected[inner]
    )
    assert error < 0.1
