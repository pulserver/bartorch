import math

import pytest
import torch

from bartorch import linop, optim, tools

VOXEL = (2.5e-3, 2e-3, 2e-3)


def _scan(shape=(12, 21, 21), coils=4, shots=3):
    nz, ny, nx = shape
    z = torch.linspace(-1, 1, nz)[:, None, None]
    y = torch.linspace(-1, 1, ny)[None, :, None]
    x = torch.linspace(-1, 1, nx)[None, None, :]
    body = torch.exp(-((x / 0.55) ** 2 + ((y - 0.1) / 0.45) ** 2 + (z / 0.6) ** 2) * 3)
    spot = torch.exp(-(((x - 0.25) / 0.15) ** 2 + ((y + 0.2) / 0.2) ** 2 + (z / 0.3) ** 2) * 3)
    image = (body + 0.6 * spot).to(torch.complex64) * torch.exp(1j * 0.4 * x)
    maps = torch.stack(
        [
            torch.exp(-((y - math.cos(c)) ** 2 + (x - math.sin(c)) ** 2 + z**2) / 2.0)
            * torch.exp(1j * torch.tensor(0.7 * c))
            for c in range(coils)
        ]
    ).to(torch.complex64)
    maps = maps / maps.abs().square().sum(0).sqrt()
    masks = torch.zeros(shots, nz, ny, dtype=torch.bool)
    for shot in range(shots):
        masks[shot, :, shot::shots] = True
    return image.contiguous(), maps.contiguous(), masks


def _turn(axis, angle):
    vector = torch.zeros(3, dtype=torch.float64)
    vector[axis] = angle
    x, y, z = vector
    skew = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float64)
    return torch.linalg.matrix_exp(skew)


def _cartesian(maps, masks, image, device):
    pattern = masks.any(0)[..., None].expand(image.shape).to(torch.complex64)
    encoding = linop.CartesianSense(maps, tuple(image.shape), pattern=pattern, device=device)
    return encoding(image.to(device)).cpu()


def _relative(a, b):
    return float((a - b).norm() / b.norm())


def test_a_rigid_encoding_at_rest_is_cartesian_sense_over_the_placed_lines(device):
    image, maps, masks = _scan()
    encoding = tools.rigid_encoding(maps, masks, VOXEL, device=device)
    assert (
        _relative(encoding(image.to(device)).cpu(), _cartesian(maps, masks, image, device)) < 5e-3
    )


def test_a_shift_of_whole_voxels_reads_the_image_rolled_by_them(device):
    image, maps, masks = _scan()
    shifts = torch.tensor([[2 * VOXEL[2], -VOXEL[1], VOXEL[0]]] * 3, dtype=torch.float64)
    encoding = tools.rigid_encoding(maps, masks, VOXEL, shifts=shifts, device=device)
    rolled = torch.roll(image, shifts=(1, -1, 2), dims=(0, 1, 2))
    moved = torch.roll(maps, shifts=(1, -1, 2), dims=(1, 2, 3))
    expected = _cartesian(moved, masks, rolled, device)
    assert _relative(encoding(image.to(device)).cpu(), expected) < 5e-3


def test_a_quarter_turn_about_z_reads_the_image_turned_counterclockwise_in_x_y(device):
    image, maps, masks = _scan()
    turns = _turn(2, math.pi / 2).expand(3, 3, 3)
    encoding = tools.rigid_encoding(maps, masks, (VOXEL[0], 2e-3, 2e-3), turns, device=device)
    turned = torch.rot90(image, 1, dims=(2, 1))
    expected = _cartesian(torch.rot90(maps, 1, dims=(3, 2)), masks, turned, device)
    read = encoding(image.to(device)).cpu()
    assert _relative(read, expected) < 5e-3


def test_the_rigid_encoding_adjoint_matches_its_forward(device):
    image, maps, masks = _scan()
    turns = torch.stack([_turn(0, 0.0), _turn(0, 0.05), _turn(1, -0.04)])
    shifts = torch.tensor([[0, 0, 0], [1e-3, -2e-3, 0.5e-3], [0, 1e-3, -1e-3]], dtype=torch.float64)
    encoding = tools.rigid_encoding(maps, masks, VOXEL, turns, shifts, device=device)
    generator = torch.Generator().manual_seed(3)
    kspace = torch.randn(encoding.oshape, dtype=torch.complex64, generator=generator)
    kspace = kspace * masks.any(0)[..., None]
    forward = complex((encoding(image.to(device)).cpu().conj() * kspace).sum())
    adjoint = complex((image.conj() * encoding.H(kspace.to(device)).cpu()).sum())
    assert abs(forward - adjoint) < 1e-3 * abs(forward)


def test_shot_poses_fitted_from_the_image_solved_at_rest_are_the_true_poses(device):
    image, maps, masks = _scan()
    turns = torch.stack([_turn(0, 0.0), _turn(0, 0.03) @ _turn(1, -0.02), _turn(2, 0.025)])
    shifts = torch.tensor(
        [[0, 0, 0], [0.6e-3, -1.0e-3, 0.8e-3], [-0.8e-3, 0.5e-3, -0.6e-3]], dtype=torch.float64
    )
    kspace = tools.rigid_encoding(maps, masks, VOXEL, turns, shifts, device=device)(
        image.to(device)
    ).cpu()
    at_rest = tools.rigid_encoding(maps, masks, VOXEL, device=device)
    start = optim.CG(lambda_=1e-3, maxiter=10)(kspace.to(device), at_rest).cpu()
    found, moved, solved, residual = tools.estimate_rigid_motion(
        kspace, maps, masks, start, VOXEL, steps=4, device=device
    )
    angle = torch.arccos(((torch.einsum("sij,sij->s", found, turns) - 1) / 2).clamp(-1, 1))
    assert float(angle.max()) < 2e-3
    assert float((moved - shifts).abs().max()) < 0.05 * min(VOXEL)
    assert _relative(solved, image) < 0.2 * _relative(start, image)
    assert residual < 1e-3


def test_a_line_placed_by_two_shots_is_refused():
    image, maps, masks = _scan()
    masks[1, 0, 0] = True
    with pytest.raises(ValueError, match="more than one shot"):
        tools.rigid_encoding(maps, masks, VOXEL)
