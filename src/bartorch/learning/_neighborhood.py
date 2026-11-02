"""Neighbourhood attention over 3D volumes: each voxel attends to a window around it."""

from __future__ import annotations

import ctypes

import torch
from torch import nn

__all__ = ["NeighborhoodAttention3d", "neighborhood_attention3d"]

_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_CHANNELS = (8, 16, 32, 64)
_MAX_WINDOW = 9


def _starts(n: int, window: int, device) -> torch.Tensor:
    """First voxel of each voxel's window along an axis of ``n``, shifted inwards at the edges."""
    i = torch.arange(n, device=device)
    return (i - window // 2).clamp(0, n - window)


def _neighbours(extent, window: int, device) -> torch.Tensor:
    """``(voxels, window^3)`` index of each neighbour of each voxel, z-major."""
    nz, ny, nx = extent
    offsets = torch.arange(window, device=device)
    z = (_starts(nz, window, device)[:, None] + offsets)[:, None, None, :, None, None]
    y = (_starts(ny, window, device)[:, None] + offsets)[None, :, None, None, :, None]
    x = (_starts(nx, window, device)[:, None] + offsets)[None, None, :, None, None, :]
    return ((z * ny + y) * nx + x).reshape(nz * ny * nx, window**3)


def _attend_reference(q, k, v, extent, window, bias, heads):
    """Neighbourhood attention in plain torch on ``(groups, voxels, channels)``."""
    index = _neighbours(extent, window, q.device)
    scale = q.shape[-1] ** -0.5
    kn = k[:, index]  # (groups, voxels, window^3, channels)
    logits = torch.einsum("gvc,gvmc->gvm", q.float() * scale, kn.float())
    if bias is not None:
        logits = logits + bias.float().repeat(q.shape[0] // heads, 1)[:, None, :]
    attention = logits.softmax(-1)
    return torch.einsum("gvm,gvmc->gvc", attention, v[:, index].float()).to(q.dtype)


def _library():
    from bartorch._lib import library

    return library()


def kernels_available() -> bool:
    """Whether this library carries the CUDA kernels and a card is present."""
    try:
        return bool(_library().bartorch_na3d_built()) and torch.cuda.is_available()
    except (ImportError, OSError, AttributeError):
        return False


def _floats(t: torch.Tensor):
    return ctypes.cast(ctypes.c_void_p(t.data_ptr()), ctypes.POINTER(ctypes.c_float))


def _call(name: str, x, y, out, extent, window: int):
    """One CUDA pass; ``x`` and ``out`` are float32 weights or elements, as the pass takes them."""
    groups, _, channels = y.shape
    nz, ny, nx = extent
    if name == "gather_dot":  # elements, elements -> weights
        first, third = ctypes.c_void_p(x.data_ptr()), _floats(out)
    else:  # weights, elements -> elements
        first, third = _floats(x), ctypes.c_void_p(out.data_ptr())
    with torch.cuda.device(y.device):
        stream = torch.cuda.current_stream(y.device).cuda_stream
        error = getattr(_library(), f"bartorch_na3d_{name}")(
            first,
            ctypes.c_void_p(y.data_ptr()),
            third,
            groups,
            nz,
            ny,
            nx,
            channels,
            window,
            _DTYPES[y.dtype],
            ctypes.c_void_p(stream),
        )
    if error:
        raise RuntimeError(f"bartorch_na3d_{name} failed with CUDA error {error}")
    return out


def _gather_dot(a, b, extent, window):
    out = torch.empty(*a.shape[:2], window**3, device=a.device, dtype=torch.float32)
    return _call("gather_dot", a, b, out, extent, window)


def _gather_sum(s, b, extent, window):
    return _call("gather_sum", s, b, torch.empty_like(b), extent, window)


def _scatter_sum(s, a, extent, window):
    return _call("scatter_sum", s, a, torch.empty_like(a), extent, window)


class _Attend(torch.autograd.Function):
    """Neighbourhood attention through the CUDA passes, with its gradient through them too."""

    @staticmethod
    def forward(ctx, q, k, v, bias, extent, window, heads):
        scale = q.shape[-1] ** -0.5
        qs = (q * scale).contiguous()
        k, v = k.to(q.dtype).contiguous(), v.to(q.dtype).contiguous()
        logits = _gather_dot(qs, k, extent, window)
        if bias is not None:
            logits = logits.view(-1, heads, *logits.shape[1:]).add_(bias.float()[None, :, None, :])
            logits = logits.view(-1, *logits.shape[2:])
        attention = logits.softmax(-1)
        out = _gather_sum(attention, v, extent, window)
        ctx.save_for_backward(qs, k, v, attention)
        ctx.extent, ctx.window, ctx.heads, ctx.scale = extent, window, heads, scale
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad):
        qs, k, v, attention = ctx.saved_tensors
        extent, window = ctx.extent, ctx.window
        grad = grad.to(qs.dtype).contiguous()
        d_attention = _gather_dot(grad, v, extent, window)
        d_logits = attention * (d_attention - (d_attention * attention).sum(-1, keepdim=True))
        dq = _gather_sum(d_logits, k, extent, window) * ctx.scale
        dk = _scatter_sum(d_logits, qs, extent, window)
        dv = _scatter_sum(attention, grad, extent, window)
        d_bias = None
        if ctx.has_bias:
            d_bias = d_logits.view(-1, ctx.heads, *d_logits.shape[1:]).sum((0, 2))
        return dq, dk, dv, d_bias, None, None, None


def neighborhood_attention3d(q, k, v, window: int = 5, bias=None) -> torch.Tensor:
    """Scaled dot-product attention of each voxel over the ``window^3`` voxels around it.

    The window is shifted inwards at an edge rather than cut, so each voxel
    sees the same number of neighbours.  On a card with the library's CUDA
    kernels, and for 8, 16, 32 or 64 channels per head, the passes and their
    gradients run there, with sums in float32 and memory linear in the
    volume; anywhere else a plain-torch version serves, which gathers every
    neighbour and needs ``window^3`` times the memory of ``k``.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``(n, heads, z, y, x, channels)``: queries, keys and values.
    window : int, default=5
        Odd edge of the window, at most 9 and at most each extent.
    bias : torch.Tensor, default=None
        ``(heads, window^3)`` added to the logits: a bias for each relative position.

    Returns
    -------
    torch.Tensor
        ``(n, heads, z, y, x, channels)``, of ``q``'s dtype.
    """
    window = int(window)
    if window < 1 or window % 2 == 0 or window > _MAX_WINDOW:
        raise ValueError(f"the window is an odd edge up to {_MAX_WINDOW}, not {window}")
    if q.ndim != 6 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k and v are (n, heads, z, y, x, channels), all of one shape")
    n, heads, nz, ny, nx, channels = q.shape
    extent = (nz, ny, nx)
    if min(extent) < window:
        raise ValueError(f"every extent is at least the window ({window}), and {extent} is not")
    if bias is not None and tuple(bias.shape) != (heads, window**3):
        raise ValueError(
            f"bias is (heads, window^3) = ({heads}, {window**3}), not {tuple(bias.shape)}"
        )

    flat = [t.reshape(n * heads, nz * ny * nx, channels) for t in (q, k, v)]
    fast = (
        q.is_cuda
        and channels in _CHANNELS
        and q.dtype in _DTYPES
        and k.device == q.device
        and v.device == q.device
        and kernels_available()
    )
    if fast:
        out = _Attend.apply(*flat, bias, extent, window, heads)
    else:
        out = _attend_reference(*flat, extent, window, bias, heads)
    return out.reshape(q.shape)


class NeighborhoodAttention3d(nn.Module):
    """Multi-head neighbourhood attention over ``(n, channels, z, y, x)`` feature maps.

    Queries, keys and values are 1-wide convolutions of the input, each head
    attends over the ``window^3`` voxels around each voxel with a learned bias
    for each relative position, and a 1-wide convolution mixes the heads.  The
    cost is linear in the volume, ``window^3`` times that of a 1-wide
    convolution, against the square of the volume for full attention.

    Parameters
    ----------
    channels : int
        Input and output channels.
    heads : int, default=None
        Heads; ``channels`` split into heads of 16 when omitted, or one head
        for fewer than 16 channels.
    window : int, default=5
        Odd edge of each voxel's window, at most 9.

    Examples
    --------
    >>> block = learning.NeighborhoodAttention3d(32, window=5)
    >>> features = features + block(features)
    """

    def __init__(self, channels: int, heads: int | None = None, *, window: int = 5):
        super().__init__()
        channels = int(channels)
        heads = max(channels // 16, 1) if heads is None else int(heads)
        if channels % heads:
            raise ValueError(f"{channels} channels do not split into {heads} heads")
        self.heads, self.window = heads, int(window)
        self.qkv = nn.Conv3d(channels, 3 * channels, 1)
        self.project = nn.Conv3d(channels, channels, 1)
        self.bias = nn.Parameter(torch.zeros(heads, self.window**3))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, nz, ny, nx = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(n, 3, self.heads, c // self.heads, nz, ny, nx)
            .permute(1, 0, 2, 4, 5, 6, 3)
        )
        made = neighborhood_attention3d(
            q.contiguous(), k.contiguous(), v.contiguous(), self.window, self.bias.to(q.dtype)
        )
        return self.project(made.permute(0, 1, 5, 2, 3, 4).reshape(n, c, nz, ny, nx))
