"""A residual UNet for image restoration, conditioned on the iteration and the noise level."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["UNet"]


def _sinusoid(value: torch.Tensor, features: int) -> torch.Tensor:
    """Sinusoidal embedding of one scalar per item, ``(n,)`` to ``(n, features)``."""
    half = features // 2
    frequency = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=value.device) / half
    )
    angle = value.float()[:, None] * frequency[None]
    return torch.cat([angle.sin(), angle.cos()], -1)


def _per_item(value, n: int, device, dtype) -> torch.Tensor:
    """A scalar, or one value per item, as an ``(n,)`` tensor."""
    value = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if 1 == value.numel():
        return value.expand(n)
    if value.numel() != n:
        raise ValueError(f"a condition is one value or one per item; {value.numel()} for {n}")
    return value


class _Conditioning(nn.Module):
    """The embedding of the iteration index, the noise level and the class, summed."""

    def __init__(self, features: int, steps: bool, noise: bool, classes: int | None):
        super().__init__()
        self.features = features
        self.steps = steps
        self.noise = noise
        self.classes = None if classes is None else nn.Embedding(int(classes), features)
        self.mlp = nn.Sequential(
            nn.Linear(features, 4 * features), nn.SiLU(), nn.Linear(4 * features, features)
        )

    def forward(self, n: int, device, sigma, step, label) -> torch.Tensor:
        made = torch.zeros(n, self.features, device=device)
        for wanted, value, name in ((self.steps, step, "step"), (self.noise, sigma, "sigma")):
            if wanted and value is None:
                raise ValueError(f"this network is conditioned on {name}, so it takes one")
            if not wanted and value is not None:
                raise ValueError(f"this network is not conditioned on {name}")
        if self.steps:
            made = made + _sinusoid(_per_item(step, n, device, torch.float32), self.features)
        if self.noise:
            level = _per_item(sigma, n, device, torch.float32).clamp_min(1e-6).log()
            # Spread the log noise level over the frequencies an index would occupy.
            made = made + _sinusoid(64.0 * level, self.features)
        if self.classes is not None:
            if label is None:
                raise ValueError("this network is conditioned on a class, so it takes a label")
            made = made + self.classes(_per_item(label, n, device, torch.int64))
        elif label is not None:
            raise ValueError("this network is not conditioned on a class")
        return self.mlp(made)


class _Conv(nn.Module):
    """A spatial convolution, followed along the frame axis by a temporal one when there is one.

    The frame axis is the one in front of the spatial axes; a (2+1)D or (3+1)D
    convolution factorised this way costs the spatial kernel plus three taps,
    against the spatial kernel times three for the full convolution.
    """

    def __init__(
        self,
        cin: int,
        cout: int,
        spatial: int,
        frames: bool,
        periodic: bool,
        stride: int = 1,
        kernel: int = 3,
    ):
        super().__init__()
        conv = nn.Conv2d if 2 == spatial else nn.Conv3d
        self.spatial = spatial
        self.space = conv(cin, cout, kernel, stride=stride, padding=kernel // 2)
        self.time = (
            nn.Conv1d(cout, cout, 3, padding=1, padding_mode="circular" if periodic else "zeros")
            if frames
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.time is None:
            return self.space(x)
        n, c, t = x.shape[:3]
        made = self.space(x.transpose(1, 2).reshape(n * t, c, *x.shape[3:]))
        rest = made.shape[2:]
        made = made.reshape(n, t, -1, *rest).permute(0, *range(3, 3 + self.spatial), 2, 1)
        c = made.shape[-2]
        made = self.time(made.reshape(-1, c, t))
        made = made.reshape(n, *rest, c, t).permute(0, -2, -1, *range(1, 1 + self.spatial))
        return made.contiguous()


class _Residual(nn.Module):
    """Two normalised convolutions with a skip, modulated by the conditioning (FiLM)."""

    def __init__(self, cin, cout, spatial, frames, periodic, groups, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.norm1 = nn.GroupNorm(min(groups, cin), cin)
        self.conv1 = _Conv(cin, cout, spatial, frames, periodic)
        self.norm2 = nn.GroupNorm(min(groups, cout), cout)
        self.conv2 = _Conv(cout, cout, spatial, frames, periodic)
        self.film = None if features is None else nn.Linear(features, 2 * cout)
        self.skip = None if cin == cout else _Conv(cin, cout, spatial, False, False, kernel=1)
        self.frames = frames

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.film is not None:
            gamma, beta = self.film(condition).chunk(2, -1)
            shape = (*gamma.shape, *(1,) * (h.ndim - 2))
            h = (1 + gamma.reshape(shape)) * h + beta.reshape(shape)
        h = F.silu(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.conv2(h)
        return h + (x if self.skip is None else self._skip(x))

    def _skip(self, x):
        if not self.frames:
            return self.skip(x)
        n, c, t = x.shape[:3]
        made = self.skip(x.transpose(1, 2).reshape(n * t, c, *x.shape[3:]))
        return made.reshape(n, t, -1, *made.shape[2:]).transpose(1, 2)


class UNet(nn.Module):
    """Residual UNet on real ``(n, channels, *spatial)`` tensors, for denoising and restoration.

    Each level holds ``blocks`` residual blocks of two 3-wide convolutions with
    group normalization and SiLU; levels are joined by strided convolutions on
    the way down and by nearest-neighbour upsampling and a convolution on the
    way up.  With ``residual`` the network returns its input plus a correction,
    and the last convolution starts at zero, so that an untrained network is
    the identity and an unrolled iteration around it starts as the plain
    iteration.

    With ``frames`` the input is ``(n, channels, frames, *spatial)`` and every
    convolution is factorised into a spatial one per frame and a temporal one
    per voxel; the frame axis is never downsampled.  This is the network for a
    time series or any other axis that is not spatial but is correlated from
    one entry to the next.

    Conditioning on the iteration index (``steps``), the noise level
    (``noise``) and a class label (``classes``) is embedded, summed and
    applied in every block as a feature-wise affine modulation (FiLM),
    ``(1 + gamma) h + beta``.  A network shared by all the iterations of an
    unrolled reconstruction is conditioned on the index; a denoiser applied
    over an annealed schedule, on the noise level; one serving several
    contrasts, on the contrast.

    Parameters
    ----------
    in_channels : int
        Input channels: 2 for one complex image as real and imaginary parts,
        ``2 k`` for ``k`` complex images such as subspace coefficients.
    out_channels : int, default=None
        Output channels; ``in_channels`` when omitted.
    spatial : {2, 3}, default=3
        Number of spatial axes.
    widths : sequence of int, default=(32, 64, 128, 256)
        Channels at each level, from full resolution down.  Every spatial
        extent is padded to a multiple of ``2 ** (len(widths) - 1)``.
    blocks : int, default=1
        Residual blocks per level on each side.
    groups : int, default=8
        Groups of the group normalization.
    frames : bool, default=False
        Whether the axis in front of the spatial ones is a frame axis.
    periodic : bool, default=False
        Whether the frame axis wraps around, as a cardiac cycle does.
    steps : bool, default=False
        Whether the network takes the iteration index, ``step``.
    noise : bool, default=False
        Whether the network takes the noise level, ``sigma``.
    classes : int, default=None
        Number of class labels the network takes, as ``label``.
    features : int, default=64
        Width of the conditioning embedding.
    residual : bool, default=True
        Whether the output is the input plus a correction.  Needs
        ``out_channels == in_channels``.
    dropout : float, default=0.0
        Dropout probability inside each residual block.  Left active at
        inference, it makes repeated reconstructions differ, which
        :func:`~bartorch.learning.moments` turns into a spread.

    Examples
    --------
    >>> net = learning.UNet(10, steps=True)              # five complex coefficients, 3D
    >>> net(planes, step=2).shape == planes.shape
    True
    >>> cine = learning.UNet(2, spatial=2, frames=True, periodic=True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        spatial: int = 3,
        widths=(32, 64, 128, 256),
        blocks: int = 1,
        groups: int = 8,
        frames: bool = False,
        periodic: bool = False,
        steps: bool = False,
        noise: bool = False,
        classes: int | None = None,
        features: int = 64,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        spatial = int(spatial)
        if spatial not in (2, 3):
            raise ValueError(f"a UNet here is 2D or 3D, so spatial is 2 or 3, not {spatial}")
        widths = tuple(int(w) for w in widths)
        if 0 == len(widths):
            raise ValueError("a UNet has at least one level")
        out_channels = int(in_channels if out_channels is None else out_channels)
        if residual and out_channels != in_channels:
            raise ValueError(
                f"a residual network adds its correction to its input, so it returns "
                f"{in_channels} channels, not {out_channels}"
            )
        self.spatial = spatial
        self.frames = bool(frames)
        self.residual = bool(residual)
        self.levels = len(widths)

        conditioned = steps or noise or classes is not None
        self.conditioning = (
            _Conditioning(int(features), bool(steps), bool(noise), classes) if conditioned else None
        )
        feat = int(features) if conditioned else None

        def block(cin, cout):
            return _Residual(cin, cout, spatial, self.frames, periodic, groups, feat, dropout)

        self.head = _Conv(int(in_channels), widths[0], spatial, self.frames, periodic)
        self.down = nn.ModuleList()
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.up = nn.ModuleList()
        for level, width in enumerate(widths):
            cin = widths[level - 1] if level else widths[0]
            if level:
                self.down.append(_Conv(cin, width, spatial, False, False, stride=2))
            self.encode.append(nn.ModuleList([block(width, width) for _ in range(blocks)]))
        for level in reversed(range(self.levels - 1)):
            width = widths[level]
            self.up.append(_Conv(widths[level + 1], width, spatial, False, False))
            self.decode.append(
                nn.ModuleList(
                    [block(2 * width, width)] + [block(width, width) for _ in range(blocks - 1)]
                )
            )
        self.norm = nn.GroupNorm(min(groups, widths[0]), widths[0])
        self.tail = _Conv(widths[0], out_channels, spatial, self.frames, periodic)
        if self.residual:
            nn.init.zeros_(self.tail.space.weight)
            nn.init.zeros_(self.tail.space.bias)
            if self.tail.time is not None:
                nn.init.zeros_(self.tail.time.weight)
                nn.init.zeros_(self.tail.time.bias)

    def forward(self, x: torch.Tensor, sigma=None, step=None, label=None) -> torch.Tensor:
        """Apply the network.

        Parameters
        ----------
        x : torch.Tensor
            ``(n, channels, *spatial)``, or ``(n, channels, frames, *spatial)``
            with ``frames``.
        sigma : float or torch.Tensor, default=None
            Noise level, one or one per item; only with ``noise``.
        step : int or torch.Tensor, default=None
            Iteration index, one or one per item; only with ``steps``.
        label : int or torch.Tensor, default=None
            Class label, one or one per item; only with ``classes``.
        """
        lead = 3 if self.frames else 2
        if x.ndim != lead + self.spatial:
            raise ValueError(
                f"this network takes (n, channels{', frames' if self.frames else ''}, "
                f"{self.spatial} spatial axes), and {tuple(x.shape)} is not that"
            )
        if self.conditioning is None:
            if (sigma, step, label) != (None, None, None):
                raise ValueError("this network is not conditioned on anything")
            condition = None
        else:
            condition = self.conditioning(x.shape[0], x.device, sigma, step, label)

        extent = x.shape[lead:]
        multiple = 2 ** (self.levels - 1)
        padding = [(-n) % multiple for n in extent]
        h = x
        if any(padding):
            pad = []
            for p in reversed(padding):
                pad += [0, p]
            h = F.pad(h, pad)

        h = self.head(h)
        skips = []
        for level in range(self.levels):
            if level:
                h = self._space(self.down[level - 1], h)
            for b in self.encode[level]:
                h = b(h, condition)
            skips.append(h)
        skips.pop()
        for up, decode in zip(self.up, self.decode):
            skip = skips.pop()
            h = self._space(up, self._upsample(h, skip.shape[lead:]))
            h = torch.cat([h, skip], 1)
            for b in decode:
                h = b(h, condition)
        h = self.tail(F.silu(self.norm(h)))
        h = h[(slice(None),) * lead + tuple(slice(0, n) for n in extent)]
        return x + h if self.residual else h

    def _space(self, conv: _Conv, h: torch.Tensor) -> torch.Tensor:
        """A purely spatial convolution, applied frame by frame when there are frames."""
        if not self.frames:
            return conv(h)
        n, c, t = h.shape[:3]
        made = conv(h.transpose(1, 2).reshape(n * t, c, *h.shape[3:]))
        return made.reshape(n, t, -1, *made.shape[2:]).transpose(1, 2)

    def _upsample(self, h: torch.Tensor, extent) -> torch.Tensor:
        if not self.frames:
            return F.interpolate(h, size=tuple(extent), mode="nearest")
        n, c, t = h.shape[:3]
        made = F.interpolate(
            h.transpose(1, 2).reshape(n * t, c, *h.shape[3:]), size=tuple(extent), mode="nearest"
        )
        return made.reshape(n, t, c, *extent).transpose(1, 2)
