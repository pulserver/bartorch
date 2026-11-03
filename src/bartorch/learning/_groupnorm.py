"""Group normalisation fused with a per-item scale and shift and SiLU, in one pass on a card."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["group_norm_silu"]

_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def _library():
    from bartorch._lib import library

    return library()


def kernels_available() -> bool:
    """Whether this library carries the CUDA pass and a card is present."""
    try:
        return bool(_library().bartorch_gn_built()) and torch.cuda.is_available()
    except (ImportError, OSError, AttributeError):
        return False


def _reference(x, groups, weight, bias, scale, shift, eps, silu):
    """The same arithmetic in plain torch, statistics in float32."""
    h = F.group_norm(x.float(), groups, weight, bias, eps)
    shape = (*x.shape[:2], *([1] * (x.ndim - 2)))
    if scale is not None:
        h = h * (1 + scale.float().reshape(shape))
    if shift is not None:
        h = h + shift.float().reshape(shape)
    if silu:
        h = F.silu(h)
    return h.to(x.dtype)


def _pointer(t):
    return None if t is None else t.data_ptr()


def _as_float(t, device, shape):
    if t is None:
        return None
    return t.detach().to(device=device, dtype=torch.float32).reshape(shape).contiguous()


def group_norm_silu(
    x: torch.Tensor,
    groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
    silu: bool = True,
) -> torch.Tensor:
    """``silu((group_norm(x) * weight + bias) * (1 + scale) + shift)``: a diffusion UNet's block.

    On a card with this library's CUDA pass, and when no gradient is asked
    for, the statistics take one read of ``x`` and the rest one more read and
    one write, in ``x``'s own element type (float32, float16 or bfloat16) with
    float32 statistics -- where eager torch reads and writes the activations
    four or five times, with casts around the norm.  Anywhere else plain torch
    computes the same, also with float32 statistics.

    Parameters
    ----------
    x : torch.Tensor
        ``(n, channels, *spatial)``.
    groups : int
        Groups the channels are normalised in; it divides ``channels``.
    weight, bias : torch.Tensor, default=None
        ``(channels,)`` affine of the norm.
    scale, shift : torch.Tensor, default=None
        ``(n, channels)`` per-item scale (applied as ``1 + scale``) and shift,
        e.g. from a noise-level embedding.
    eps : float, default=1e-5
        Added to the variance.
    silu : bool, default=True
        Apply SiLU last.

    Returns
    -------
    torch.Tensor
        Of ``x``'s shape and dtype.
    """
    n, channels = x.shape[:2]
    if channels % groups:
        raise ValueError(f"{channels} channels do not split into {groups} groups")
    grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (x, weight, bias, scale, shift)
    )
    fast = x.is_cuda and x.dtype in _DTYPES and not grad and x.numel() > 0 and kernels_available()
    if not fast:
        return _reference(x, groups, weight, bias, scale, shift, eps, silu)

    return _launch(x, groups, weight, bias, scale, shift, eps, silu)


def _launch(x, groups, weight, bias, scale, shift, eps, silu):
    """The library's pass on ``x`` wherever its memory is; the caller has checked it can run."""
    from bartorch.learning._neighborhood import _on

    lib = _library()
    n, channels = x.shape[:2]
    voxels = x[0, 0].numel()
    x = x.contiguous()
    device = x.device
    w = _as_float(weight, device, (channels,))
    b = _as_float(bias, device, (channels,))
    s = _as_float(scale, device, (n, channels))
    t = _as_float(shift, device, (n, channels))
    size = lib.bartorch_gn_workspace(n, channels, voxels, groups)
    workspace = torch.empty(size, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    with _on(device) as stream:
        error = lib.bartorch_gn_silu(
            x.data_ptr(),
            out.data_ptr(),
            n,
            channels,
            voxels,
            groups,
            _pointer(w),
            _pointer(b),
            _pointer(s),
            _pointer(t),
            float(eps),
            int(bool(silu)),
            _DTYPES[x.dtype],
            workspace.data_ptr(),
            stream,
        )
    if error:
        raise RuntimeError(f"bartorch_gn_silu failed with CUDA error {error}")
    return out
