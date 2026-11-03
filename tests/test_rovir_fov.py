"""ROVir from non-Cartesian samples: a source past the field of view is rejected."""

import numpy as np
import torch

from bartorch import tools


def _samples(seed: int = 0):
    rng = np.random.default_rng(seed)
    channels, n = 8, 32
    # Twice as dense as the field of view needs, as a spiral's centre is.
    g = torch.arange(-8, 8.5, 0.5)
    k = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 1, 3)
    blob = torch.exp(-2 * (2 * torch.pi / n) ** 2 * (k * k).sum(-1))  # two voxels wide

    def source(r):
        return blob * torch.exp(-2j * torch.pi * (k @ torch.tensor(r)) / n)

    c_in = rng.standard_normal(channels) + 1j * rng.standard_normal(channels)
    c_in = torch.from_numpy(c_in).to(torch.complex64)
    c_out = torch.zeros(channels, dtype=torch.complex64)
    c_out[:2] = 3.0  # the coils near the neck
    y = c_in[:, None, None] * source([0.0, 0.0, 0.0])
    y = y + c_out[:, None, None] * source([0.0, 0.0, -0.8 * n])
    # The interference region's noise keeps its correlation invertible.
    gen = torch.Generator().manual_seed(seed)
    y = y + 0.01 * torch.randn(y.shape, dtype=torch.complex64, generator=gen)
    return y.to(torch.complex64), k, c_in, c_out


def test_a_source_past_the_field_of_view_is_rejected_and_one_inside_kept():
    y, k, c_in, c_out = _samples()
    matrix, (kept, ratio) = tools.rovir_fov(y, k, coils=2, keep=4, return_energy=True)
    w = matrix.reshape(8, 8)[:2]
    # Noise stays white.
    assert torch.allclose(w @ w.conj().mT, torch.eye(2, dtype=w.dtype), atol=1e-4)
    share = lambda c: float((w.conj() @ c).abs().pow(2).sum() / c.abs().pow(2).sum())  # noqa: E731
    assert share(c_in) > 0.5
    assert share(c_out) < 0.05
    assert ratio < 0.05 and 0 < kept <= 1


def test_ccapply_takes_the_matrix_as_its_virtual_coils():
    y, k, *_ = _samples()
    matrix = tools.rovir_fov(y, k, coils=2, keep=4)
    w = matrix.reshape(8, 8)[:2]
    virtual = tools.ccapply(y[..., None].contiguous(), matrix, p=2)[..., 0]
    assert torch.allclose(virtual, torch.einsum("jc,c...->j...", w.conj(), y), atol=1e-4)
