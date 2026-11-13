"""A known displacement field applied to an image (no PyHySCO needed)."""

import pytest
import torch

from bartorch.tools import apply_susceptibility
from bartorch.tools._correct._susceptibility import _sampled


def _image(rows: int = 32, cols: int = 12) -> torch.Tensor:
    return torch.rand(rows, cols, dtype=torch.float64, generator=torch.Generator().manual_seed(0))


def _uniform(image: torch.Tensor, voxels: float, voxel: float) -> torch.Tensor:
    return torch.full((image.shape[0] + 1, image.shape[1]), voxels * voxel, dtype=torch.float64)


def test_a_uniform_field_samples_blip_up_at_x_plus_b_and_blip_down_at_x_minus_b():
    image = _image()
    field = _uniform(image, 3.0, 2.0)

    up = apply_susceptibility(image, field, voxel_size=(2.0, 1.0))
    down = apply_susceptibility(image, field, voxel_size=(2.0, 1.0), blip="down")

    torch.testing.assert_close(up, torch.roll(image, -3, dims=0))
    torch.testing.assert_close(down, torch.roll(image, 3, dims=0))


def test_a_fractional_uniform_field_is_the_pre_alignment_shift():
    image = _image()
    field = _uniform(image, -7.3, 1.5)

    torch.testing.assert_close(
        apply_susceptibility(image, field, voxel_size=(1.5, 1.0)), _sampled(image, -7.3, 0)
    )


def test_the_intensity_is_modulated_by_the_field_derivative_and_mass_is_kept():
    rows = 128
    x = torch.arange(rows, dtype=torch.float64)
    image = torch.exp(-(((x - 64) / 10) ** 2))[:, None].repeat(1, 4)
    edges = torch.arange(rows + 1, dtype=torch.float64) - 0.5
    field = (3.0 * torch.sin(2 * torch.pi * edges / rows))[:, None].repeat(1, 4)

    up = apply_susceptibility(image, field, voxel_size=(1.0, 1.0))
    down = apply_susceptibility(image, field, voxel_size=(1.0, 1.0), blip="down")

    # At the bump's centre the field is zero and its derivative -0.15 per voxel.
    slope = float(field[65, 0] - field[64, 0])
    assert slope == pytest.approx(-0.147, abs=1e-3)
    assert float(up[64, 0]) == pytest.approx(1 + slope)
    assert float(down[64, 0]) == pytest.approx(1 - slope)
    torch.testing.assert_close(up.sum(0), image.sum(0), rtol=1e-3, atol=0)
    torch.testing.assert_close(down.sum(0), image.sum(0), rtol=1e-3, atol=0)


def test_blip_down_with_a_field_is_blip_up_with_its_negative():
    image = _image()
    field = torch.randn(33, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(1))

    torch.testing.assert_close(
        apply_susceptibility(image, field, voxel_size=(1.2, 1.0), blip="down"),
        apply_susceptibility(image, -field, voxel_size=(1.2, 1.0)),
    )


def test_a_volume_is_corrected_slice_by_slice_along_its_phase_encoding_axis():
    volume = torch.rand(5, 16, 9, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    field = torch.randn(5, 17, 9, dtype=torch.float64, generator=torch.Generator().manual_seed(3))

    corrected = apply_susceptibility(
        volume, field, voxel_size=(4.0, 1.1, 1.0), phase_encoding_axis=1
    )

    for k in range(5):
        torch.testing.assert_close(
            corrected[k], apply_susceptibility(volume[k], field[k], voxel_size=(1.1, 1.0))
        )


def test_a_complex_image_is_corrected_as_its_real_and_imaginary_parts():
    image = torch.complex(_image(), _image().flip(0))
    field = torch.randn(33, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(4))

    corrected = apply_susceptibility(image, field, voxel_size=(1.0, 1.0))

    torch.testing.assert_close(
        corrected.real, apply_susceptibility(image.real, field, voxel_size=(1.0, 1.0))
    )
    torch.testing.assert_close(
        corrected.imag, apply_susceptibility(image.imag, field, voxel_size=(1.0, 1.0))
    )


def test_a_field_not_on_the_cell_edges_of_the_image_is_refused():
    image = _image()
    with pytest.raises(ValueError, match="field"):
        apply_susceptibility(image, torch.zeros(32, 12, dtype=torch.float64), voxel_size=(1.0, 1.0))
    with pytest.raises(ValueError, match="blip"):
        apply_susceptibility(
            image, torch.zeros(33, 12, dtype=torch.float64), voxel_size=(1.0, 1.0), blip="left"
        )
