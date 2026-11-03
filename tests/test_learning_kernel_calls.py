"""How the learning module's CUDA passes are called, checked on the host.

The passes themselves need a card (``test_learning_attention.py`` and
``test_learning_groupnorm.py`` check them there). Here a stand-in library does
the same arithmetic in torch on host memory reached through the pointers it is
handed, so everything on the Python side of the ABI -- shapes, pointers,
workspace, the autograd function around the attention passes -- runs anywhere.
"""

import ctypes

import pytest
import torch

from bartorch.learning import _groupnorm, _neighborhood


def _address(pointer) -> int:
    return pointer if isinstance(pointer, int) else ctypes.cast(pointer, ctypes.c_void_p).value


def _view(pointer, *shape) -> torch.Tensor:
    """The float32 host memory at ``pointer`` as a tensor of ``shape``, without a copy."""
    count = 1
    for size in shape:
        count *= size
    array = (ctypes.c_float * count).from_address(_address(pointer))
    return torch.frombuffer(array, dtype=torch.float32).reshape(shape)


class _HostAttention:
    """``bartorch_na3d_*`` on host float32 memory."""

    def bartorch_na3d_built(self):
        return 1

    def _index(self, nz, ny, nx, window):
        return _neighborhood._neighbours((nz, ny, nx), window, "cpu")

    def bartorch_na3d_gather_dot(
        self, a, b, s, groups, nz, ny, nx, channels, window, dtype, stream
    ):
        voxels, index = nz * ny * nx, self._index(nz, ny, nx, window)
        a, b = _view(a, groups, voxels, channels), _view(b, groups, voxels, channels)
        _view(s, groups, voxels, window**3)[:] = torch.einsum("gvc,gvmc->gvm", a, b[:, index])
        return 0

    def bartorch_na3d_gather_sum(
        self, s, b, o, groups, nz, ny, nx, channels, window, dtype, stream
    ):
        voxels, index = nz * ny * nx, self._index(nz, ny, nx, window)
        s, b = _view(s, groups, voxels, window**3), _view(b, groups, voxels, channels)
        _view(o, groups, voxels, channels)[:] = torch.einsum("gvm,gvmc->gvc", s, b[:, index])
        return 0

    def bartorch_na3d_scatter_sum(
        self, s, a, o, groups, nz, ny, nx, channels, window, dtype, stream
    ):
        voxels, index = nz * ny * nx, self._index(nz, ny, nx, window)
        s, a = _view(s, groups, voxels, window**3), _view(a, groups, voxels, channels)
        out = torch.zeros(groups, voxels, channels)
        contributions = s[..., None] * a[:, :, None, :]  # (groups, voxels, window^3, channels)
        out.index_add_(1, index.reshape(-1), contributions.reshape(groups, -1, channels))
        _view(o, groups, voxels, channels)[:] = out
        return 0


class _HostGroupNorm:
    """``bartorch_gn_*`` on host float32 memory; ``fail`` makes the pass report a CUDA error."""

    def __init__(self, fail=0):
        self.fail = fail

    def bartorch_gn_built(self):
        return 1

    def bartorch_gn_workspace(self, items, channels, voxels, groups):
        return 2 * items * groups

    def bartorch_gn_silu(
        self, x, y, n, c, voxels, groups, w, b, s, t, eps, silu, last, dtype, ws, stream
    ):
        if self.fail:
            return self.fail
        assert dtype == 0 and stream is None

        def optional(pointer, *shape):
            return None if pointer is None else _view(pointer, *shape)

        def volume(pointer):  # (n, c, voxels), whichever way it lies in memory
            if last:
                return _view(pointer, n, voxels, c).transpose(1, 2)
            return _view(pointer, n, c, voxels)

        made = _groupnorm._reference(
            volume(x),
            groups,
            optional(w, c),
            optional(b, c),
            optional(s, n, c),
            optional(t, n, c),
            eps,
            bool(silu),
        )
        volume(y)[:] = made
        return 0


def test_attention_passes_and_their_gradients_through_the_abi(monkeypatch):
    monkeypatch.setattr(_neighborhood, "_library", _HostAttention)
    torch.manual_seed(0)
    n, heads, extent, channels, window = 2, 2, (4, 5, 3), 8, 3
    shape = (n * heads, extent[0] * extent[1] * extent[2], channels)
    q, k, v = (torch.randn(shape, requires_grad=True) for _ in range(3))
    bias = (0.1 * torch.randn(heads, window**3)).requires_grad_()
    made = _neighborhood._Attend.apply(q, k, v, bias, extent, window, heads)
    grad = torch.randn(shape)
    made.backward(grad)

    rq, rk, rv, rbias = (t.detach().clone().requires_grad_() for t in (q, k, v, bias))
    expected = _neighborhood._attend_reference(rq, rk, rv, extent, window, rbias, heads)
    expected.backward(grad)
    torch.testing.assert_close(made, expected, atol=1e-5, rtol=1e-5)
    for got, want in (
        (q.grad, rq.grad),
        (k.grad, rk.grad),
        (v.grad, rv.grad),
        (bias.grad, rbias.grad),
    ):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_attention_without_a_bias_and_a_failing_pass(monkeypatch):
    monkeypatch.setattr(_neighborhood, "_library", _HostAttention)
    q = torch.randn(1, 27, 8)
    made = _neighborhood._Attend.apply(q, q, q, None, (3, 3, 3), 3, 1)
    want = _neighborhood._attend_reference(q, q, q, (3, 3, 3), 3, None, 1)
    torch.testing.assert_close(made, want, atol=1e-5, rtol=1e-5)

    class Failing(_HostAttention):
        def bartorch_na3d_gather_dot(self, *args):
            return 700

    monkeypatch.setattr(_neighborhood, "_library", Failing)
    with pytest.raises(RuntimeError, match="CUDA error 700"):
        _neighborhood._Attend.apply(q, q, q, None, (3, 3, 3), 3, 1)


def test_kernels_are_unavailable_without_a_card(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_neighborhood, "_library", _HostAttention)
    monkeypatch.setattr(_groupnorm, "_library", _HostGroupNorm)
    assert not _neighborhood.kernels_available()
    assert not _groupnorm.kernels_available()


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("film", [False, True])
def test_group_norm_pass_through_the_abi(monkeypatch, affine, film):
    monkeypatch.setattr(_groupnorm, "_library", _HostGroupNorm)
    torch.manual_seed(1)
    x = 2 + torch.randn(2, 8, 3, 4, 5)
    weight, bias = (torch.randn(8), torch.randn(8)) if affine else (None, None)
    scale, shift = (torch.randn(2, 8), torch.randn(2, 8)) if film else (None, None)
    made = _groupnorm._launch(x, 4, weight, bias, scale, shift, 1e-5, True)
    want = _groupnorm._reference(x, 4, weight, bias, scale, shift, 1e-5, True)
    torch.testing.assert_close(made, want)


def test_group_norm_pass_reports_its_error(monkeypatch):
    monkeypatch.setattr(_groupnorm, "_library", lambda: _HostGroupNorm(fail=700))
    with pytest.raises(RuntimeError, match="CUDA error 700"):
        _groupnorm._launch(torch.randn(1, 4, 8), 2, None, None, None, None, 1e-5, False)


def test_group_norm_pass_keeps_a_channels_last_layout(monkeypatch):
    monkeypatch.setattr(_groupnorm, "_library", _HostGroupNorm)
    torch.manual_seed(2)
    x = (1 + torch.randn(2, 8, 3, 4, 5)).to(memory_format=torch.channels_last_3d)
    assert _groupnorm._channels_last(x)
    scale, shift = torch.randn(2, 8), torch.randn(2, 8)
    made = _groupnorm._launch(x, 4, None, None, scale, shift, 1e-5, True)
    assert made.is_contiguous(memory_format=torch.channels_last_3d)
    want = _groupnorm._reference(x, 4, None, None, scale, shift, 1e-5, True)
    torch.testing.assert_close(made, want)
    assert not _groupnorm._channels_last(torch.zeros(1, 4, 1, 1, 1))  # contiguous either way: not
