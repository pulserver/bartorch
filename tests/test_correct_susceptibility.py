"""Susceptibility correction from a reversed phase-encoding pair, by PyHySCO."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from bartorch.tools import correct_susceptibility

needs_pyhysco = pytest.mark.skipif(
    importlib.util.find_spec("EPI_MRI") is None,
    reason="PyHySCO is GPL-3.0-only and an optional extra",
)


def displaced_pair(shape=(24, 32, 32), amplitude=3.0):
    """A phantom and the two images a field would displace it into."""
    axes = [torch.linspace(-1, 1, n, dtype=torch.float64) for n in shape]
    rows, columns, planes = torch.meshgrid(*axes, indexing="ij")
    phantom = (((rows / 0.7) ** 2 + (columns / 0.8) ** 2 + (planes / 0.8) ** 2) < 1).double() * (
        1 + 0.5 * torch.sin(5 * rows) * torch.cos(4 * columns)
    )
    shift = amplitude * torch.exp(-(columns**2 + planes**2) / 0.3)
    index = torch.arange(shape[0], dtype=torch.float64)[:, None, None]

    def warp(image, offset):
        source = (index + offset).clamp(0, shape[0] - 1)
        lower = source.floor().long()
        upper = (lower + 1).clamp(max=shape[0] - 1)
        fraction = source - lower
        return (
            image.gather(0, lower.expand_as(image)) * (1 - fraction)
            + image.gather(0, upper.expand_as(image)) * fraction
        )

    return warp(phantom, shift), warp(phantom, -shift), shift


def test_pair_must_share_a_shape() -> None:
    with pytest.raises(ValueError, match="one shape"):
        correct_susceptibility(
            torch.zeros(4, 4, 4), torch.zeros(4, 4, 5), voxel_size=(1.0, 1.0, 1.0)
        )


def test_voxel_size_must_cover_every_axis() -> None:
    with pytest.raises(ValueError, match="voxel_size has"):
        correct_susceptibility(torch.zeros(4, 4, 4), torch.zeros(4, 4, 4), voxel_size=(1.0, 1.0))


def test_phase_encoding_axis_must_be_one_pyhysco_orders() -> None:
    with pytest.raises(ValueError, match="no PyHySCO axis order"):
        correct_susceptibility(
            torch.zeros(4, 4, 4),
            torch.zeros(4, 4, 4),
            voxel_size=(1.0, 1.0, 1.0),
            phase_encoding_axis=2,
        )


@needs_pyhysco
def test_correction_returns_tensors_in_the_callers_orientation() -> None:
    up, down, _ = displaced_pair()
    result = correct_susceptibility(up, down, voxel_size=(1.0, 1.0, 1.0), max_iter=6)
    assert result.blip_up.shape == up.shape
    assert result.blip_down.shape == up.shape
    # The field lives on cell edges, so it is one longer along phase encoding.
    assert result.field_map.shape == (up.shape[0] + 1, *up.shape[1:])


@needs_pyhysco
def test_correction_recovers_the_imposed_displacement() -> None:
    up, down, shift = displaced_pair()
    result = correct_susceptibility(up, down, voxel_size=(1.0, 1.0, 1.0), max_iter=8)
    assert result.field_map.abs().max().item() == pytest.approx(float(shift.max()), rel=0.4)
    before = (up - down).abs().mean()
    after = (result.blip_up - result.blip_down).abs().mean()
    assert after < before


@needs_pyhysco
def test_field_is_in_the_units_of_the_voxel_size() -> None:
    up, down, shift = displaced_pair()
    result = correct_susceptibility(up, down, voxel_size=(2.0, 1.0, 1.0), max_iter=8)
    assert result.field_map.abs().max().item() == pytest.approx(2.0 * float(shift.max()), rel=0.4)


@needs_pyhysco
def test_corrected_pair_is_in_the_intensity_units_of_the_input() -> None:
    """Both corrected images match the undisplaced phantom, not PyHySCO's [0, 256] range."""
    up, down, _ = displaced_pair()
    phantom, _, _ = displaced_pair(amplitude=0.0)
    up, down, phantom = 7.0 * up + 3.0, 7.0 * down + 3.0, 7.0 * phantom + 3.0
    result = correct_susceptibility(up, down, voxel_size=(1.0, 1.0, 1.0), max_iter=8)
    for corrected, distorted in ((result.blip_up, up), (result.blip_down, down)):
        assert corrected.mean().item() == pytest.approx(phantom.mean().item(), rel=0.02)
        assert (corrected - phantom).abs().mean() < 0.5 * (distorted - phantom).abs().mean()


@needs_pyhysco
def test_correction_writes_nothing_to_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    up, down, _ = displaced_pair(shape=(16, 24, 24))
    correct_susceptibility(up, down, voxel_size=(1.0, 1.0, 1.0), max_iter=4)
    assert list(tmp_path.rglob("*")) == []


@needs_pyhysco
def test_phase_encoding_along_the_second_axis_is_corrected_in_place() -> None:
    up, down, _ = displaced_pair(shape=(16, 24, 24))
    up, down = up.transpose(0, 1).contiguous(), down.transpose(0, 1).contiguous()
    result = correct_susceptibility(
        up, down, voxel_size=(1.0, 1.0, 1.0), phase_encoding_axis=1, max_iter=4
    )
    assert result.blip_up.shape == up.shape
    assert result.field_map.shape == (up.shape[0], up.shape[1] + 1, up.shape[2])


@needs_pyhysco
def test_unknown_optimizer_is_rejected() -> None:
    up, down, _ = displaced_pair(shape=(16, 24, 24))
    with pytest.raises(ValueError, match="unknown optimizer"):
        correct_susceptibility(up, down, voxel_size=(1.0, 1.0, 1.0), optimizer="newton-raphson")


def displaced_slice(shape=(64, 64), amplitude=2.0, voxel=2.0):
    """A smooth head-like slice and the pair a smooth field displaces it into, in mm."""
    rows, columns = torch.meshgrid(
        *[torch.linspace(-1, 1, n, dtype=torch.float64) for n in shape], indexing="ij"
    )
    radius = (rows / 0.75) ** 2 + (columns / 0.6) ** 2
    phantom = torch.sigmoid((1 - radius) * 12) * (
        1 + 0.4 * torch.cos(6 * rows) * torch.cos(5 * columns)
    )
    shift = amplitude * torch.exp(-((rows + 0.4) ** 2 + columns**2) / 0.15)
    index = torch.arange(shape[0], dtype=torch.float64)[:, None]

    def warp(offset):
        source = (index + offset).clamp(0, shape[0] - 1)
        lower = source.floor().long().clamp(max=shape[0] - 2)
        fraction = source - lower
        sampled = (
            phantom.gather(0, lower) * (1 - fraction) + phantom.gather(0, lower + 1) * fraction
        )
        return sampled * (1 + torch.gradient(offset, dim=0)[0])

    # The phantom sampled at x - b, its mass kept, is displaced by +b: the
    # blip-up image.
    return warp(-shift), warp(shift), shift * voxel, radius < 0.8


def _centred(field_map):
    return 0.5 * (field_map[1:] + field_map[:-1])


@needs_pyhysco
def test_a_slice_pair_is_corrected_to_the_full_displacement() -> None:
    up, down, shift, inside = displaced_slice()
    result = correct_susceptibility(up, down, voxel_size=(2.0, 2.0), prealign=False)
    field = _centred(result.field_map)
    assert result.field_map.shape == (up.shape[0] + 1, up.shape[1])
    slope = (field[inside] * shift[inside]).sum() / (shift[inside] ** 2).sum()
    assert slope.item() == pytest.approx(1.0, abs=0.15)


@needs_pyhysco
def test_a_few_bright_voxels_do_not_stiffen_the_field() -> None:
    up, down, _, _ = displaced_slice()
    plain = correct_susceptibility(up, down, voxel_size=(2.0, 2.0), prealign=False)
    up[2, 2] = down[2, 2] = 20 * up.max()
    bright = correct_susceptibility(up, down, voxel_size=(2.0, 2.0), prealign=False)
    assert (bright.field_map - plain.field_map).abs().max() < 0.1 * plain.field_map.abs().max()
