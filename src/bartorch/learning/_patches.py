"""A network applied patch by patch on a device, to images held elsewhere."""

from __future__ import annotations

import contextlib
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
    recomputes a checkpointed step exactly.  With ``shift=True`` the image is
    zero-padded by the offset in front and to a whole patch behind, which
    costs a layer of patches along every axis: a 256^3 volume in 64^3 patches
    becomes 125 of them.  With ``shift="circular"`` the image is rolled by the
    offset instead, so the patches that cross its far edge wrap round to its
    near one and a volume of whole patches stays at its own count, 64; the
    image is padded behind only to a whole patch, as without a shift.  The
    wrapped patches join opposite faces of the field of view, where an image
    with a background is empty.

    With ``skip_zero`` a patch that is zero in every item and channel, such as
    background or padding, does not reach the network and is zero in the
    result, which spares the network's time on the empty part of a volume.
    It is the network's output for such a patch only when that is zero, as
    for a residual denoiser of a nonnegative image.

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
    shift : bool or "circular", default=True
        Offset the patch grid at random at every call, padding the image
        (``True``) or rolling it (``"circular"``).
    overlap : bool, default=True
        Without gradient, with the image on the host and the network on a card,
        copy one group of patches on a second stream while the network computes
        on another.
    skip_zero : bool, default=False
        Leave the patches that are zero throughout out of the network, zero.
    autotune : bool, default=False
        On a card, let cuDNN time its convolution algorithms for each new
        shape (``torch.backends.cudnn.benchmark``) for the length of a call,
        and restore the setting after it.  The first call pays the timing;
        the patches of every later one share their shape.

    Raises
    ------
    ValueError
        ``patch`` not of two or three positive extents, ``batch`` below one,
        or ``shift`` neither a bool nor ``"circular"``.

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
        shift: bool | str = True,
        overlap: bool = True,
        skip_zero: bool = False,
        autotune: bool = False,
    ):
        super().__init__()
        patch = tuple(int(p) for p in patch)
        if len(patch) not in (2, 3) or min(patch) < 1:
            raise ValueError(f"a patch spans 2 or 3 axes, each of at least one voxel, not {patch}")
        if batch < 1:
            raise ValueError(f"a call takes at least one patch, not {batch}")
        if not isinstance(shift, bool) and "circular" != shift:
            raise ValueError(f'shift is True, False or "circular", not {shift!r}')
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net
        self.patch = patch
        self.device = torch.device(device)
        self.dtype = _autocast_dtype(self.device, dtype)
        self.batch = int(batch)
        self.shift = shift
        self.overlap = bool(overlap)
        self.skip_zero = bool(skip_zero)
        self.autotune = bool(autotune)

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
        axes = tuple(range(lead, x.ndim))

        offset = [int(torch.randint(p, ())) if self.shift and p > 1 else 0 for p in self.patch]
        rolled = "circular" == self.shift and any(offset)
        if rolled:
            x_in, front = torch.roll(x, offset, axes), [0] * dims
        else:
            x_in, front = x, offset
        after = [(-(e + o)) % p for e, o, p in zip(extent, front, self.patch)]
        pad = []
        for o, a in zip(reversed(front), reversed(after)):
            pad += [o, a]
        padded = F.pad(x_in, pad) if any(pad) else x_in
        grid = [range(0, s, p) for s, p in zip(padded.shape[lead:], self.patch)]
        corners = list(itertools.product(*grid))
        if self.skip_zero:
            corners = [c for c in corners if bool(padded[self._window(c, lead)].detach().any())]

        out = torch.zeros(padded.shape, dtype=torch.float32, device=x.device)
        crop = (slice(None),) * lead + tuple(slice(o, o + e) for o, e in zip(front, extent))
        moving = "cpu" == x.device.type and "cuda" == self.device.type
        chunks = [corners[i : i + self.batch] for i in range(0, len(corners), self.batch)]
        with self._tuned():
            if moving and self.overlap and not torch.is_grad_enabled():
                if chunks:
                    self._overlapped(padded, chunks, lead, n, sigma, keywords, out)
            else:
                for chunk in chunks:
                    pieces = torch.cat([padded[self._window(c, lead)] for c in chunk])
                    if moving and not pieces.requires_grad:
                        pieces = pieces.pin_memory()
                    pieces = pieces.to(self.device, non_blocking=moving)
                    made = self._run(pieces, len(chunk), n, sigma, keywords)
                    made = made.to(x.device, torch.float32).reshape(len(chunk), n, *made.shape[1:])
                    for i, c in enumerate(chunk):
                        out[self._window(c, lead)] = made[i]
        out = out[crop]
        return torch.roll(out, [-o for o in offset], axes) if rolled else out

    @contextlib.contextmanager
    def _tuned(self):
        """cuDNN's algorithm search on for the call, under ``autotune`` on a card."""
        if not (self.autotune and "cuda" == self.device.type):
            yield
            return
        before = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        try:
            yield
        finally:
            torch.backends.cudnn.benchmark = before

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
            f"batch={self.batch}, shift={self.shift!r}, overlap={self.overlap}, "
            f"skip_zero={self.skip_zero}, autotune={self.autotune}"
        )
