"""Density compensation weights, and what they are worth to a navigator plane."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

import bartorch
from bartorch.tools import RigidRegistration, reconstruct_navigator

GRID = 64
SPOKES = 64


def spokes(count: int = SPOKES, samples: int = GRID) -> np.ndarray:
    """Golden-angle spokes ``(count * samples, 2)`` in grid units of ``GRID``."""
    angles = np.arange(count) * np.pi * (3 - np.sqrt(5))
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False) * GRID
    return np.stack(
        [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
        axis=-1,
    ).reshape(-1, 2)


def phantom(size: int = GRID) -> np.ndarray:
    rows, columns = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    image = ((rows / 0.7) ** 2 + (columns / 0.55) ** 2 < 1).astype(float)
    image[size // 3 : size // 2, size // 3 : 2 * size // 3] += 0.6
    return image


def acquire(image: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """The explicit Fourier sum of ``image`` at ``traj``: ``kx`` pairs with columns."""
    rows, columns = (np.arange(n) - n // 2 for n in image.shape)
    along_columns = np.exp(-2j * np.pi * np.outer(traj[:, 0], columns) / image.shape[1])
    along_rows = np.exp(-2j * np.pi * np.outer(traj[:, 1], rows) / image.shape[0])
    return ((image @ along_columns.T).T * along_rows).sum(axis=1)


def _ramp() -> torch.Tensor:
    return torch.as_tensor(np.tile(np.abs(np.linspace(-0.5, 0.5, GRID, endpoint=False)), SPOKES))


def _plane(image: np.ndarray, traj: np.ndarray, density: torch.Tensor) -> np.ndarray:
    samples = torch.as_tensor(acquire(image, traj)[None, None])
    return reconstruct_navigator(
        samples, torch.as_tensor(traj[None]), (GRID, GRID), density=density[None]
    )[0].numpy()


def test_the_weights_follow_the_trajectory() -> None:
    """One weight per sample, non-negative, and each frame weighted on its own."""
    traj = torch.as_tensor(spokes())
    flat = bartorch.estimate_density(traj, (GRID, GRID), iterations=5)
    stacked = bartorch.estimate_density(traj[None], (GRID, GRID), iterations=5)
    assert flat.shape == (traj.shape[0],)
    assert stacked.shape == (1, traj.shape[0])
    assert bool((flat >= 0).all())
    torch.testing.assert_close(flat, stacked[0])


def test_the_weight_rises_with_radius() -> None:
    """The centre is visited by every spoke, so it counts for least."""
    weights = bartorch.estimate_density(torch.as_tensor(spokes()), (GRID, GRID)).numpy()
    radius = np.abs(np.linspace(-0.5, 0.5, GRID, endpoint=False))
    profile = weights.reshape(SPOKES, GRID).mean(axis=0)
    inner, outer = profile[radius < 0.1].mean(), profile[radius > 0.3].mean()
    assert outer > 4 * inner


def test_the_weights_are_on_the_device_of_the_trajectory(device) -> None:
    traj = torch.as_tensor(spokes(), device=device)
    weights = bartorch.estimate_density(traj, (GRID, GRID), iterations=3)
    assert weights.device.type == device
    reference = bartorch.estimate_density(traj.cpu(), (GRID, GRID), iterations=3)
    torch.testing.assert_close(weights.cpu(), reference, rtol=1e-2, atol=1e-4)


def test_it_beats_the_ramp_on_the_image() -> None:
    """The ramp over-weights the outer k-space the spokes have left sparse."""
    traj = spokes()
    truth = phantom()
    truth = truth / truth.max()
    pipe = bartorch.estimate_density(torch.as_tensor(traj), (GRID, GRID))

    def residual(density):
        plane = _plane(truth, traj, density)
        return np.sqrt(((plane / plane.max() - truth) ** 2).mean())

    assert residual(pipe) < residual(_ramp())


@pytest.mark.skipif(
    importlib.util.find_spec("SimpleITK") is None, reason="registration needs the motion extra"
)
def test_it_registers_a_pose_to_a_hundredth_of_a_pixel() -> None:
    """The shift is along the columns, which is SimpleITK's ``tx``."""
    traj = spokes()
    truth = phantom()
    shifted = np.roll(truth, 5, axis=1)
    pipe = bartorch.estimate_density(torch.as_tensor(traj), (GRID, GRID))
    registration = RigidRegistration()

    def measured(density):
        planes = [_plane(image, traj, density) for image in (truth, shifted)]
        estimate = registration(*planes, spacing=(1.0, 1.0)).parameters
        return abs(np.asarray(estimate)[1] - 5.0)

    assert measured(pipe) < 0.01
    assert measured(pipe) <= measured(_ramp()) + 0.01


def test_a_trajectory_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="grid units"):
        bartorch.estimate_density(torch.as_tensor(spokes() * 3), (GRID, GRID), iterations=2)


def test_a_kz_on_a_two_dimensional_grid_is_refused() -> None:
    traj = torch.ones(SPOKES * GRID, 3)
    with pytest.raises(ValueError, match="takes no kz"):
        bartorch.estimate_density(traj, (GRID, GRID), iterations=2)


def test_a_zero_kz_on_a_plane_weighs_as_the_planar_trajectory() -> None:
    planar = torch.as_tensor(spokes(16, 32), dtype=torch.float32)
    padded = torch.cat([planar, torch.zeros(planar.shape[0], 1)], dim=-1)
    torch.testing.assert_close(
        bartorch.estimate_density(padded, (GRID, GRID), iterations=3),
        bartorch.estimate_density(planar, (GRID, GRID), iterations=3),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("traj", "shape", "match"),
    [
        (torch.zeros(8, 2), (GRID,), "two- or three-dimensional"),
        (torch.zeros(8), (GRID, GRID), "samples, 2 or 3"),
        (torch.zeros(8, 4), (GRID, GRID), "samples, 2 or 3"),
        (torch.zeros(8, 2), (GRID, GRID, GRID), "3D grid needs 3"),
    ],
)
def test_a_trajectory_that_does_not_fit_the_grid_is_refused(traj, shape, match) -> None:
    with pytest.raises(ValueError, match=match):
        bartorch.estimate_density(traj, shape)
