"""Navigator plane reconstruction by density-compensated gridding."""

from __future__ import annotations

import torch

__all__ = ["reconstruct_navigator"]


def reconstruct_navigator(
    kspace: torch.Tensor,
    traj: torch.Tensor,
    shape: tuple[int, int],
    *,
    density: torch.Tensor | None = None,
) -> torch.Tensor:
    """Magnitude images of a navigator's planes by density-compensated adjoint NUFFT.

    Each plane is reconstructed as :math:`|A^H (w \\odot y)|` over its own
    trajectory, with :math:`A` the non-uniform transform of
    :class:`bartorch.linop.NUFFT`, and several coils are combined by root sum
    of squares.

    Parameters
    ----------
    kspace : torch.Tensor
        Samples ``(planes, samples)`` or ``(planes, coils, samples)``.
    traj : torch.Tensor
        Trajectory ``(planes, samples, 2)`` in grid units of ``shape``,
        ``kx, ky`` with ``kx`` along the last image axis, within
        :math:`\\pm n/2`.
    shape : tuple of int
        Matrix ``(y, x)`` of each plane.
    density : torch.Tensor, default=None
        Density compensation weights broadcastable to ``(planes, samples)``,
        such as :func:`bartorch.estimate_density` returns.  ``None`` weights
        every sample by one.

    Returns
    -------
    torch.Tensor
        Real ``float32`` magnitudes ``(planes, *shape)``, on the device of
        ``kspace``, for :meth:`NavigatorMotionTracker.track`.

    Raises
    ------
    ValueError
        If the trajectory does not match the samples, or leaves the grid.
    """
    from bartorch.fourier import _check_trajectory
    from bartorch.linop import NUFFT

    kspace = torch.as_tensor(kspace)
    if kspace.ndim == 2:
        kspace = kspace[:, None, :]
    if kspace.ndim != 3:
        raise ValueError(
            f"expected (planes, samples) or (planes, coils, samples), got {tuple(kspace.shape)}"
        )
    planes, coils, samples = kspace.shape
    shape = tuple(int(n) for n in shape)
    traj = torch.as_tensor(traj, device=kspace.device).to(torch.float32)
    if tuple(traj.shape) != (planes, samples, 2):
        raise ValueError(
            f"trajectory {tuple(traj.shape)} does not match samples {(planes, samples, 2)}"
        )
    traj = _check_trajectory(traj, shape)

    values = kspace.to(torch.complex64).permute(1, 0, 2).reshape(coils, planes, 1, samples)
    if density is not None:
        weights = torch.as_tensor(density, device=kspace.device).to(torch.float32)
        values = values * torch.broadcast_to(weights, (planes, samples)).reshape(planes, 1, samples)
    operator = NUFFT(traj.reshape(planes, 1, samples, 2), (coils, planes, *shape), toeplitz=False)
    images = operator.H(values.contiguous())
    return images.abs().square().sum(dim=0).sqrt()
