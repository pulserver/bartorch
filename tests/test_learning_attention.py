"""Neighbourhood attention: the plain-torch path against attention written out voxel by voxel,
and the library's CUDA passes, with their gradients, against the plain-torch path.
"""

import itertools

import pytest
import torch

from bartorch import learning
from bartorch.learning._neighborhood import _attend_reference, kernels_available


def _brute(q, k, v, window, bias):
    """Softmax over each voxel's shifted window, one voxel at a time."""
    n, heads, nz, ny, nx, c = q.shape
    out = torch.zeros_like(q)

    def start(i, size):
        return min(max(i - window // 2, 0), size - window)

    for b, h, z, y, x in itertools.product(range(n), range(heads), range(nz), range(ny), range(nx)):
        sz, sy, sx = start(z, nz), start(y, ny), start(x, nx)
        keys = k[b, h, sz : sz + window, sy : sy + window, sx : sx + window].reshape(-1, c)
        values = v[b, h, sz : sz + window, sy : sy + window, sx : sx + window].reshape(-1, c)
        logits = keys @ q[b, h, z, y, x] / c**0.5
        if bias is not None:
            logits = logits + bias[h]
        out[b, h, z, y, x] = logits.softmax(0) @ values
    return out


@pytest.mark.parametrize("window", [1, 3, 5])
def test_reference_is_attention_over_the_shifted_window(window):
    torch.manual_seed(0)
    shape = (2, 2, 5, 6, 7, 4)
    q, k, v = (torch.randn(shape, dtype=torch.float64) for _ in range(3))
    bias = torch.randn(2, window**3, dtype=torch.float64)
    made = learning.neighborhood_attention3d(q, k, v, window, bias)
    torch.testing.assert_close(made, _brute(q, k, v, window, bias))


def test_full_window_is_full_attention():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 1, 3, 3, 3, 8, dtype=torch.float64) for _ in range(3))
    made = learning.neighborhood_attention3d(q, k, v, 3)
    flat = [t.reshape(27, 8) for t in (q, k, v)]
    full = torch.softmax(flat[0] @ flat[1].T / 8**0.5, -1) @ flat[2]
    torch.testing.assert_close(made.reshape(27, 8), full)


def test_rejects_what_it_cannot_do():
    q = torch.zeros(1, 1, 4, 4, 4, 8)
    with pytest.raises(ValueError, match="odd"):
        learning.neighborhood_attention3d(q, q, q, 4)
    with pytest.raises(ValueError, match="at least the window"):
        learning.neighborhood_attention3d(q, q, q, 5)
    with pytest.raises(ValueError, match="bias"):
        learning.neighborhood_attention3d(q, q, q, 3, torch.zeros(2, 27))


def test_module_keeps_the_shape_and_trains():
    torch.manual_seed(2)
    block = learning.NeighborhoodAttention3d(16, window=3)
    x = torch.randn(2, 16, 5, 4, 6, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert x.grad is not None and block.bias.grad is not None and block.qkv.weight.grad is not None


cuda = pytest.mark.skipif(
    not kernels_available(), reason="needs a card and the library's CUDA kernels"
)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("channels,window", [(8, 3), (16, 5), (32, 3), (64, 5)])
def test_cuda_passes_match_the_reference_with_gradients(dtype, channels, window):
    torch.manual_seed(3)
    shape = (2, 2, 7, 9, 6, channels)
    tensors = [torch.randn(shape, device="cuda", dtype=torch.float64) for _ in range(3)]
    bias64 = 0.1 * torch.randn(2, window**3, device="cuda", dtype=torch.float64)
    grad = torch.randn(shape, device="cuda", dtype=torch.float64)

    q, k, v = (t.to(dtype).requires_grad_() for t in tensors)
    bias = bias64.to(dtype).requires_grad_()
    made = learning.neighborhood_attention3d(q, k, v, window, bias)
    made.backward(grad.to(dtype))

    rq, rk, rv = (t.clone().requires_grad_() for t in tensors)
    rbias = bias64.clone().requires_grad_()
    flat = [t.reshape(4, -1, channels) for t in (rq, rk, rv)]
    expected = _attend_reference(*flat, shape[2:5], window, rbias, 2).reshape(shape)
    expected.backward(grad)

    tolerance = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 1e-1}[dtype]
    pairs = [
        (made, expected),
        (q.grad, rq.grad),
        (k.grad, rk.grad),
        (v.grad, rv.grad),
        (bias.grad, rbias.grad),
    ]
    for got, want in pairs:
        torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)


@cuda
def test_cuda_uses_memory_linear_in_the_volume():
    q = torch.randn(1, 2, 64, 64, 64, 16, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    learning.neighborhood_attention3d(q, q, q, 5)
    torch.cuda.synchronize()
    # float32 weights (2 x 64^3 x 125 x 4 bytes = 262 MB) and the output; no gathered keys.
    assert torch.cuda.max_memory_allocated() - before < 600e6
