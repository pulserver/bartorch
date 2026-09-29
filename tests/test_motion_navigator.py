"""Reconstructing a navigator's planes, and handing them to the registration."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from bartorch.tools import RigidRegistration, reconstruct_navigator

needs_simpleitk = pytest.mark.skipif(
    importlib.util.find_spec("SimpleITK") is None,
    reason="rigid registration needs the motion extra",
)


def radial(size: int, samples: int = 64, spokes: int = 48) -> np.ndarray:
    """A radial plane trajectory ``(1, spokes * samples, 2)`` in grid units of ``size``."""
    angles = np.linspace(0, np.pi, spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False) * size
    return np.stack(
        [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
        axis=-1,
    ).reshape(1, -1, 2)


def sample(image: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """The explicit Fourier sum of ``image`` at ``traj``: ``kx`` pairs with columns."""
    rows, columns = (np.arange(n) - n // 2 for n in image.shape)
    kx, ky = traj[0, :, 0], traj[0, :, 1]
    along_columns = np.exp(-2j * np.pi * np.outer(kx, columns) / image.shape[1])
    along_rows = np.exp(-2j * np.pi * np.outer(ky, rows) / image.shape[0])
    return ((image @ along_columns.T).T * along_rows).sum(axis=1)[None]


def phantom(size: int = 32) -> np.ndarray:
    image = np.zeros((size, size))
    image[10:22, 12:20] = 1.0
    return image


def _ramp(traj: np.ndarray) -> np.ndarray:
    return np.tile(np.abs(np.linspace(-0.5, 0.5, 64, endpoint=False)), traj.shape[1] // 64)[None]


def test_a_plane_comes_back(device) -> None:
    """Sampled off a phantom by an explicit sum and gridded back, the plane is the phantom."""
    truth = phantom()
    traj = radial(32)
    planes = reconstruct_navigator(
        torch.as_tensor(sample(truth, traj), device=device),
        torch.as_tensor(traj),
        truth.shape,
        density=torch.as_tensor(_ramp(traj)),
    )
    assert planes.device.type == device
    recovered = planes[0].cpu().numpy()
    recovered = recovered / recovered.max()
    assert np.corrcoef(recovered.ravel(), truth.ravel())[0, 1] > 0.9


@needs_simpleitk
def test_the_planes_can_be_registered() -> None:
    """Sample, grid, register, and get the shift back."""
    truth = phantom()
    traj = radial(32)
    weights = torch.as_tensor(_ramp(traj))

    def reconstruct(image):
        samples = torch.as_tensor(sample(image, traj))
        planes = reconstruct_navigator(samples, torch.as_tensor(traj), truth.shape, density=weights)
        return planes[0]

    estimate = RigidRegistration()(
        reconstruct(truth), reconstruct(np.roll(truth, 2, axis=0)), spacing=(1.0, 1.0)
    )
    assert np.abs(np.asarray(estimate.parameters)[1:]).max() > 1.0


def test_several_coils_are_combined() -> None:
    traj = torch.as_tensor(radial(16))
    samples = torch.ones(1, 4, traj.shape[1], dtype=torch.complex64)
    assert reconstruct_navigator(samples, traj, (16, 16)).shape == (1, 16, 16)


def test_a_mismatched_trajectory_is_refused() -> None:
    with pytest.raises(ValueError, match="does not match samples"):
        reconstruct_navigator(
            torch.ones(1, 10, dtype=torch.complex64), torch.as_tensor(radial(16)), (16, 16)
        )


def test_a_trajectory_out_of_range_is_refused() -> None:
    traj = torch.as_tensor(radial(16) * 10)
    samples = torch.ones(1, traj.shape[1], dtype=torch.complex64)
    with pytest.raises(ValueError, match="grid units"):
        reconstruct_navigator(samples, traj, (16, 16))


def test_samples_that_are_not_planes_of_coils_are_refused() -> None:
    with pytest.raises(ValueError, match="expected \\(planes, samples\\)"):
        reconstruct_navigator(
            torch.ones(1, 1, 1, 10, dtype=torch.complex64), torch.as_tensor(radial(16)), (16, 16)
        )
