"""The fused group norm: the plain-torch path against the steps written out, and the
library's CUDA pass against the plain-torch path.
"""

import pytest
import torch
import torch.nn.functional as F

from bartorch import learning
from bartorch.learning._groupnorm import kernels_available


def _steps(x, groups, weight, bias, scale, shift, silu):
    n, c = x.shape[:2]
    g = x.double().reshape(n, groups, -1)
    g = (g - g.mean(-1, keepdim=True)) / (g.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    h = g.reshape(x.shape)
    shape = (n, c, *([1] * (x.ndim - 2)))
    if weight is not None:
        h = h * weight.double().reshape(1, c, *([1] * (x.ndim - 2))) + bias.double().reshape(
            1, c, *([1] * (x.ndim - 2))
        )
    if scale is not None:
        h = h * (1 + scale.double().reshape(shape)) + shift.double().reshape(shape)
    return F.silu(h) if silu else h


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("film", [False, True])
@pytest.mark.parametrize("silu", [False, True])
def test_reference_is_the_steps(affine, film, silu):
    torch.manual_seed(0)
    x = 3 + 2 * torch.randn(2, 8, 5, 6, 7)
    weight, bias = (torch.randn(8), torch.randn(8)) if affine else (None, None)
    scale, shift = (torch.randn(2, 8), torch.randn(2, 8)) if film else (None, None)
    made = learning.group_norm_silu(x, 4, weight, bias, scale, shift, silu=silu)
    torch.testing.assert_close(
        made.double(), _steps(x, 4, weight, bias, scale, shift, silu), atol=1e-5, rtol=1e-5
    )


def test_channels_must_split_into_groups():
    with pytest.raises(ValueError, match="groups"):
        learning.group_norm_silu(torch.zeros(1, 6, 4, 4, 4), 4)


def test_gradients_go_through_plain_torch():
    x = torch.randn(1, 4, 3, 3, 3, requires_grad=True)
    learning.group_norm_silu(x, 2).sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not kernels_available(), reason="no card, or no CUDA pass in this library")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 64, 9, 10, 11), (1, 128, 32, 32, 32), (3, 32, 7)])
def test_pass_matches_plain_torch(dtype, shape):
    torch.manual_seed(0)
    x = (1 + 3 * torch.randn(shape, device="cuda")).to(dtype)
    n, c = shape[:2]
    weight, bias = torch.randn(c, device="cuda"), torch.randn(c, device="cuda")
    scale, shift = torch.randn(n, c, device="cuda"), torch.randn(n, c, device="cuda")
    with torch.no_grad():
        made = learning.group_norm_silu(x, 32 if c >= 32 else 8, weight, bias, scale, shift)
    want = _steps(x.float(), 32 if c >= 32 else 8, weight, bias, scale, shift, True)
    tolerance = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}[dtype]
    assert float((made.double() - want).abs().max() / want.abs().max()) < tolerance
