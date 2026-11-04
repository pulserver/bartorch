"""iGROG: samples moved by closed-form GRAPPA kernels land on the grid's own k-space."""

import math

import torch

from bartorch import tools


def _object(n=32, coils=6, seed=0):
    """A smooth 2-D object under smooth coil maps, and its exact k-space at any position."""
    g = torch.Generator().manual_seed(seed)
    r = torch.arange(n, dtype=torch.float64) - n // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    obj = torch.exp(-((xx / 7) ** 2 + (yy / 9) ** 2)) * (1 + 0.3 * torch.cos(xx / 3))
    centres = torch.rand(coils, 2, generator=g, dtype=torch.float64) * n - n / 2
    maps = torch.stack(
        [torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12.0**2)) for cx, cy in centres]
    )
    phase = torch.exp(1j * torch.rand(coils, 1, 1, generator=g, dtype=torch.float64) * 2 * math.pi)
    images = (maps * phase * obj).to(torch.complex128)  # (C, y, x)

    def kspace(k):  # k (..., 2) = (kx, ky) in grid units -> (C, ...)
        e = torch.exp(-2j * math.pi * (k[..., 0, None, None] * xx + k[..., 1, None, None] * yy) / n)
        return torch.einsum("cyx,...yx->c...", images, e)

    return images, kspace


def test_gridded_samples_match_the_grid_k_space():
    n = 32
    _, kspace = _object(n)
    # Radial-ish readouts through the centre at assorted angles, half a cell apart.
    angles = torch.linspace(0, math.pi, 24, dtype=torch.float64)[:-1] + 0.1
    s = torch.arange(-10, 10.5, 0.5, dtype=torch.float64)
    traj = torch.stack(
        [s[None] * torch.cos(angles)[:, None], s[None] * torch.sin(angles)[:, None]], -1
    )
    y = kspace(traj).to(torch.complex64)  # (C, shots, samples)
    c = torch.arange(-8, 8, dtype=torch.float64)
    ky, kx = torch.meshgrid(c, c, indexing="ij")
    calib = kspace(torch.stack([kx, ky], -1)).to(torch.complex64)  # (C, 16, 16), centre at 8

    g = tools.igrog(y, traj.float(), (n, n), calib, lam=1e-6)
    assert (
        g.values.shape == (6, g.index.numel())
        and g.index.numel() == g.shot.numel() == g.sample.numel()
    )
    # The exact k-space at each target's grid point.
    iy, ix = g.index // n - n // 2, g.index % n - n // 2
    truth = kspace(torch.stack([ix, iy], -1).double())
    err = (g.values.to(torch.complex128) - truth).norm() / truth.norm()
    moved = (y[:, g.shot, g.sample].to(torch.complex128) - truth).norm() / truth.norm()
    assert err < 0.02, err
    assert err < moved / 5  # far closer than leaving each sample where it was
    w = g.weights()
    assert torch.isclose(w.square().mean(), torch.tensor(1.0)) and (w > 0).all()


def test_groups_pick_their_own_calibration():
    n = 32
    _, kspace = _object(n)
    s = torch.arange(-10, 10.5, 0.5, dtype=torch.float64)
    angles = torch.tensor([0.3, 1.1, 2.0, 2.7], dtype=torch.float64)
    traj = torch.stack(
        [s[None] * torch.cos(angles)[:, None], s[None] * torch.sin(angles)[:, None]], -1
    )
    y = kspace(traj).to(torch.complex64)
    c = torch.arange(-8, 8, dtype=torch.float64)
    ky, kx = torch.meshgrid(c, c, indexing="ij")
    calib = kspace(torch.stack([kx, ky], -1)).to(torch.complex64)
    one = tools.igrog(y, traj.float(), (n, n), calib)
    two = tools.igrog(y, traj.float(), (n, n), [calib, calib * 2], group=torch.tensor([0, 1, 0, 1]))
    # Scaling a calibration leaves its Wiener kernel unchanged (lam is relative to its power).
    assert torch.allclose(
        one.values, two.values, rtol=1e-3, atol=1e-4 * float(one.values.abs().max())
    )


def test_single_precision_matches_double():
    """A card solves the kernels in single precision; the gridded samples barely move."""
    n = 32
    _, kspace = _object(n)
    angles = torch.linspace(0, math.pi, 24, dtype=torch.float64)[:-1] + 0.1
    s = torch.arange(-10, 10.5, 0.5, dtype=torch.float64)
    traj = torch.stack(
        [s[None] * torch.cos(angles)[:, None], s[None] * torch.sin(angles)[:, None]], -1
    )
    y = kspace(traj).to(torch.complex64)
    c = torch.arange(-8, 8, dtype=torch.float64)
    ky, kx = torch.meshgrid(c, c, indexing="ij")
    calib = kspace(torch.stack([kx, ky], -1)).to(torch.complex64)
    double = tools.igrog(y, traj.float(), (n, n), calib, double=True)
    single = tools.igrog(y, traj.float(), (n, n), calib, double=False)
    assert torch.equal(double.index, single.index)
    rel = (single.values - double.values).norm() / double.values.norm()
    assert rel < 1e-3, rel
    assert torch.allclose(single.noise, double.noise, rtol=1e-2)
