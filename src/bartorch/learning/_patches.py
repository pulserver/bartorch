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
    """Network applied to an image patch by patch on its own device.

    The image stays on its device, normally the host, and is cut into
    non-overlapping patches over its trailing ``len(patch)`` axes.  Groups of
    ``batch`` patches are copied to ``device``, passed through the network under
    automatic mixed precision and copied back into place, so a volume or a time
    series of volumes larger than the device's memory passes through a network
    trained on patches.  The device holds the network and ``batch`` patches, or
    under ``overlap`` two groups and the outputs not yet placed.  The result is
    float32 on the input's device and differentiable with respect to the input
    and the weights.

    With ``shift`` the patch grid is offset by a random amount at every call,
    drawn from PyTorch's generator, so seams fall in a different place at each
    iteration of a reconstruction, and :class:`~bartorch.learning.Unrolled`
    recomputes a checkpointed step exactly.  The image is zero-padded by the
    offset in front and to a whole patch behind.

    The network is moved to ``device`` at every call if it is elsewhere, so a
    training loop that moves the whole model does not move it; its parameters
    remain the same objects.

    Parameters
    ----------
    net : nn.Module
        Called on ``(n, channels, [frames,] *patch)`` as ``net(x)`` or
        ``net(x, sigma)``, with the keywords this module is called with.
    patch : sequence of int
        Patch extent along each trailing axis, two or three of them.
    device : torch.device or str, default=None
        Where the network runs; the first CUDA device when there is one, the
        host otherwise.
    dtype : torch.dtype or None or "auto", default="auto"
        Precision of automatic mixed precision; ``None`` disables it.
        ``"auto"`` is bfloat16 on a card that supports it, float16 on one that
        does not (such as a T4), and none on the host.
    batch : int, default=1
        Patches per call of the network.
    shift : bool, default=True
        Offset the patch grid at random at every call.
    overlap : bool, default=True
        Without gradient, with the image on the host and the network on a card,
        copy one group of patches on a second stream while the network computes
        on another.

    Raises
    ------
    ValueError
        ``patch`` not of two or three positive extents, or ``batch`` below one.

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
        overlap: bool = True,
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
        self.overlap = bool(overlap)

    def forward(self, x: torch.Tensor, sigma=None, **keywords) -> torch.Tensor:
        """Apply the network to every patch of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            ``(n, channels, ..., *patched)``, with the patched axes last.
        sigma : float or torch.Tensor, default=None
            Passed to the network; one value per item of ``x`` is repeated for each
            of that item's patches.
        **keywords
            Passed to the network, repeated per patch like ``sigma``.

        Returns
        -------
        torch.Tensor
            Float32, of ``x``'s shape, on ``x``'s device.
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
        chunks = [corners[i : i + self.batch] for i in range(0, len(corners), self.batch)]
        if moving and self.overlap and not torch.is_grad_enabled():
            self._overlapped(padded, chunks, lead, n, sigma, keywords, out)
            return out[
                (slice(None),) * lead + tuple(slice(o, o + e) for o, e in zip(offset, extent))
            ]
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

    def _overlapped(self, padded, chunks, lead: int, n: int, sigma, keywords, out) -> None:
        """The patch loop with copies overlapping the network, for inference from the host.

        Each chunk is copied up on a side stream while the previous one is in
        the network, and copied down into pinned memory on the compute stream
        without waiting for it; the host places a chunk once its copy has
        landed.  The card holds two chunks of input and the outputs not yet
        placed.
        """
        compute = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(self.device)

        def upload(chunk):
            host = torch.cat([padded[self._window(c, lead)] for c in chunk]).pin_memory()
            with torch.cuda.stream(side):
                pieces = host.to(self.device, non_blocking=True)
                arrived = torch.cuda.Event()
                arrived.record(side)
            return chunk, pieces, arrived

        def place(chunk, made):
            made = made.reshape(len(chunk), n, *made.shape[1:])
            for i, c in enumerate(chunk):
                out[self._window(c, lead)] = made[i]

        pending = []
        following = upload(chunks[0])
        for index in range(len(chunks)):
            chunk, pieces, arrived = following
            compute.wait_event(arrived)
            pieces.record_stream(compute)
            if index + 1 < len(chunks):
                following = upload(chunks[index + 1])
            made = self._run(pieces, len(chunk), n, sigma, keywords).float()
            landed = torch.empty(made.shape, dtype=torch.float32, pin_memory=True)
            landed.copy_(made, non_blocking=True)
            done = torch.cuda.Event()
            done.record(compute)
            pending.append((chunk, landed, done))
            while pending and pending[0][2].query():
                chunk, landed, _ = pending.pop(0)
                place(chunk, landed)
        for chunk, landed, done in pending:
            done.synchronize()
            place(chunk, landed)

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
            f"batch={self.batch}, shift={self.shift}, overlap={self.overlap}"
        )
