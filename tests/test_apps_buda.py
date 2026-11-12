"""BUDA: one image from EPI shots of opposite phase-encode polarity."""

from __future__ import annotations

import pytest
import torch

from bartorch import apps, linop, optim
from tests._epi_pair import acquire, coil_maps, disc, field, times

N, COILS, LINE_TIME = 48, 6, 1e-3


def nrmse(image, truth, support):
    image, truth = image.abs()[support], truth.abs()[support]
    scale = float((image * truth).sum() / (image * image).sum())
    return float((scale * image - truth).norm() / truth.norm())


@pytest.fixture(scope="module")
def pair():
    image, maps, off_resonance = disc(N), coil_maps(N, COILS), field(N, 60.0)
    support = image.abs() > 0
    off_resonance = off_resonance * support
    lines = range(0, N, 2)
    up, down = times(N, LINE_TIME, +1), times(N, LINE_TIME, -1)
    kspace = torch.stack(
        [
            acquire(image, maps, off_resonance, up, lines),
            acquire(image, maps, off_resonance, down, lines),
        ]
    )
    readout_time = torch.stack([up, down])[:, :, None].expand(2, N, N)
    return image, maps, off_resonance, support, kspace, readout_time


def test_the_joint_image_is_undistorted_where_each_shot_alone_is_not(pair):
    image, maps, off_resonance, support, kspace, readout_time = pair
    sense = linop.CartesianSense(
        maps, (N, N), pattern=(kspace[0].abs().sum(0) > 0).to(torch.complex64)
    )
    alone = optim.CG(1e-3, maxiter=30)(kspace[0].reshape(sense.oshape), sense).reshape(N, N)

    joint = apps.buda(kspace, maps, off_resonance, readout_time, l2=1e-3, mask=support)

    assert nrmse(alone, image, support) > 0.1
    assert nrmse(joint, image, support) < 0.03


def test_each_shot_has_its_own_phase(pair):
    image, maps, off_resonance, support, kspace, readout_time = pair
    drift = torch.linspace(-0.5, 0.5, N)[None, :].expand(N, N)
    shifted = kspace.clone()
    shifted[1] = acquire(
        image * torch.exp(1j * drift), maps, off_resonance, readout_time[1, :, 0], range(0, N, 2)
    )

    phase = apps.shot_phase(shifted, maps, off_resonance, readout_time, mask=support)
    difference = torch.angle(torch.exp(1j * (phase[1] - phase[0])))

    assert phase.shape == (2, N, N)
    assert float((difference - drift)[support].abs().mean()) < 0.1


def test_a_phase_named_otherwise_is_refused(pair):
    _, maps, off_resonance, _, kspace, readout_time = pair
    with pytest.raises(ValueError, match="estimate"):
        apps.buda(kspace, maps, off_resonance, readout_time, phase="guess")
