"""A network applied patch by patch on a device, to images held elsewhere."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["Patchwise"]


def _autocast_dtype(device: torch.device, dtype):
    """The precision a device computes the network in: bf16 where supported, else fp16."""
    if "auto" != dtype:
        return dtype
    if "cuda" != device.type:
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Patchwise(nn.Module):
    """A network applied to an image one patch at a time, on its own device.

    The image stays where it is -- normally the host -- and is cut into
    non-overlapping patches over its trailing ``len(patch)`` axes, which are
    copied to the device a few at a time, passed through the network under
    automatic mixed precision, and copied back into place.  What the device
    holds at any moment is the network and ``batch`` patches, so a volume or
    a time series of volumes larger than the device's memory goes through a
    network trained on patches of it.  The result is on the input's device,
    in single precision, and differentiable with respect to both the input
    and the weights.

    Before cutting, the image is padded with zeros by a random offset in front
    of each axis, drawn anew at every call from PyTorch's generator, and by
    whatever completes the last patch behind it.  The seams therefore fall in
    a different place at each iteration of a reconstruction and are not
    reinforced; with ``shift=False`` they are always at multiples of the patch.
    Drawing from PyTorch's generator is what lets
    :class:`~bartorch.learning.Unrolled` recompute a checkpointed step
    exactly.

    The network is moved to ``device`` before every call if it is not there,
    so a training loop that moves the whole model elsewhere does not move it;
    its parameters stay the same objects, and an optimizer holding them is
    unaffected.

    Parameters
    ----------
    net : nn.Module
        Called on ``(n, channels, [frames,] *patch)`` as ``net(x)``,
        ``net(x, sigma)`` or with the keywords this module is called with.
    patch : sequence of int
        Patch extent along each trailing axis, 2 or 3 of them.
    device : torch.device or str, default=None
        Where the network runs; the first CUDA device when there is one, and
        the host otherwise.
    dtype : torch.dtype or None or "auto", default="auto"
        Precision of automatic mixed precision on the device.  ``"auto"`` is
        bfloat16 on a card that supports it and float16 on one that does not,
        such as a T4, and no mixed precision on the host.
    batch : int, default=1
        Patches per call of the network.
    shift : bool, default=True
        Whether the patch grid is offset at random at every call.

    Examples
    --------
    >>> net = learning.Patchwise(learning.UNet(10), patch=(64, 64, 64), batch=4)
    >>> prior = priors.ImplicitPrior(learning.ComplexNet(net, channels=1))
    """

    def __init__(
        self,
        net: nn.Module,
        patch,
        *,
        device=None,
        dtype="auto",
        batch: int = 1,
        shift: bool = True,
    ):
        super().__init__()
        patch = tuple(int(p) for p in patch)
        if len(patch) not in (2, 3) or min(patch) < 1:
            raise ValueError(f"a patch spans 2 or 3 axes, each of at least one voxel, not {patch}")
        if batch < 1:
            raise ValueError(f"a call takes at least one patch, not {batch}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net
        self.patch = patch
        self.device = torch.device(device)
        self.dtype = _autocast_dtype(self.device, dtype)
        self.batch = int(batch)
        self.shift = bool(shift)

    def forward(self, x: torch.Tensor, sigma=None, **keywords) -> torch.Tensor:
        """Apply the network over every patch of ``x``, ``(n, channels, ..., *patch-axes)``.

        ``sigma`` and any keyword given one value per item of ``x`` are
        repeated for each of that item's patches.
        """
        dims = len(self.patch)
        if x.ndim < 2 + dims:
            raise ValueError(
                f"the network takes (n, channels, ..., {dims} patched axes), and "
                f"{tuple(x.shape)} has {x.ndim} axes"
            )
        self._place()
        n = x.shape[0]
        extent = x.shape[-dims:]
        lead = x.ndim - dims

        offset = [int(torch.randint(p, ())) if self.shift and p > 1 else 0 for p in self.patch]
        after = [(-(e + o)) % p for e, o, p in zip(extent, offset, self.patch)]
        pad = []
        for o, a in zip(reversed(offset), reversed(after)):
            pad += [o, a]
        padded = F.pad(x, pad) if any(pad) else x
        grid = [range(0, s, p) for s, p in zip(padded.shape[lead:], self.patch)]
        corners = list(itertools.product(*grid))

        out = torch.zeros(padded.shape, dtype=torch.float32, device=x.device)
        moving = "cpu" == x.device.type and "cuda" == self.device.type
        for start in range(0, len(corners), self.batch):
            chunk = corners[start : start + self.batch]
            pieces = torch.cat([padded[self._window(c, lead)] for c in chunk])
            if moving and not pieces.requires_grad:
                pieces = pieces.pin_memory()
            pieces = pieces.to(self.device, non_blocking=moving)
            made = self._run(pieces, len(chunk), n, sigma, keywords)
            made = made.to(x.device, torch.float32).reshape(len(chunk), n, *made.shape[1:])
            for i, c in enumerate(chunk):
                out[self._window(c, lead)] = made[i]
        return out[(slice(None),) * lead + tuple(slice(o, o + e) for o, e in zip(offset, extent))]

    def _window(self, corner, lead: int):
        return (slice(None),) * lead + tuple(slice(c, c + p) for c, p in zip(corner, self.patch))

    def _run(self, pieces, count: int, n: int, sigma, keywords) -> torch.Tensor:
        """One call of the network over ``count`` patches of each of ``n`` items."""

        def repeat(value):
            if isinstance(value, torch.Tensor) and value.ndim and n == value.shape[0] and n > 1:
                return value.to(self.device).repeat(count)
            return value.to(self.device) if isinstance(value, torch.Tensor) else value

        extra = {k: repeat(v) for k, v in keywords.items()}
        with torch.autocast(
            self.device.type, dtype=self.dtype or torch.float32, enabled=self.dtype is not None
        ):
            if sigma is None:
                return self.net(pieces, **extra)
            return self.net(pieces, repeat(sigma), **extra)

    def _place(self) -> None:
        first = next(itertools.chain(self.net.parameters(), self.net.buffers()), None)
        if first is not None and first.device != self.device:
            self.net.to(self.device)

    def extra_repr(self) -> str:
        return (
            f"patch={self.patch}, device={self.device}, dtype={self.dtype}, "
            f"batch={self.batch}, shift={self.shift}"
        )
