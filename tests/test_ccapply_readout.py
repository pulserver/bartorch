"""A readout streamed on its own, projected onto SVD virtual channels."""

import pytest
import torch

from bartorch import tools


def _scan(channels=12, lines=24, samples=40):
    generator = torch.Generator().manual_seed(0)
    mixing = torch.randn(channels, 3, dtype=torch.complex64, generator=generator)
    sources = torch.randn(3, lines, samples, dtype=torch.complex64, generator=generator)
    noise = 0.05 * torch.randn(channels, lines, samples, dtype=torch.complex64, generator=generator)
    return (torch.einsum("cs,s...->c...", mixing, sources) + noise).contiguous()


def test_each_readout_projects_as_ccapply_projects_the_whole_scan(device):
    kspace = _scan()
    matrix = tools.cc(kspace[:, None], M=True, S=True)
    whole = tools.ccapply(kspace[:, None], matrix, p=4, S=True)[:, 0]

    lines = [
        tools.ccapply_readout(kspace[:, line].to(device), matrix, p=4)
        for line in range(kspace.shape[1])
    ]

    assert lines[0].device.type == torch.device(device).type
    assert torch.allclose(torch.stack(lines, 1).cpu(), whole.cpu(), atol=1e-4)


def test_a_readout_of_other_channels_is_refused():
    matrix = torch.eye(8, dtype=torch.complex64)

    with pytest.raises(ValueError, match="channels"):
        tools.ccapply_readout(torch.zeros(6, 10, dtype=torch.complex64), matrix, p=4)
