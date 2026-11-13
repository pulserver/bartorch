"""The uniform displacement removed before PyHySCO (no PyHySCO needed)."""

import torch

from bartorch.tools._correct._susceptibility import _profile_lag, _sampled


def _disk(centre_row: float) -> torch.Tensor:
    rows = torch.arange(96, dtype=torch.float64)[:, None, None]
    cols = torch.arange(64, dtype=torch.float64)[None, :, None]
    return (
        (((rows - centre_row) ** 2 + (cols - 32) ** 2) < 15**2).to(torch.float64).expand(96, 64, 3)
    )


def test_the_lag_of_a_pair_moved_apart_is_found_to_a_fraction_of_a_voxel():
    up, down = _disk(30.0), _disk(63.5)
    assert abs(_profile_lag(up, down, 0) - 33.5) < 0.6
    assert abs(_profile_lag(down, up, 0) + 33.5) < 0.6


def test_sampling_half_the_lag_brings_the_pair_together():
    up, down = _disk(30.0), _disk(64.0)
    offset = -0.5 * _profile_lag(up, down, 0)
    moved_up, moved_down = _sampled(up, offset, 0), _sampled(down, -offset, 0)
    assert torch.allclose(moved_up, moved_down, atol=1e-6)
    assert torch.allclose(moved_up, _disk(47.0), atol=1e-6)
