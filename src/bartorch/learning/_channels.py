"""Complex tensors in the channel-first real layout networks and ``torchio`` use."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ComplexNet", "as_complex", "as_real"]


def as_real(input: torch.Tensor) -> torch.Tensor:
    """Stack the real and imaginary parts of ``input`` along a new leading axis.

    The channel-first counterpart of :func:`torch.view_as_real`, as
    convolutional networks and ``torchio.ScalarImage`` take channels.

    Parameters
    ----------
    input : torch.Tensor
        Complex, of any shape.  A real tensor is given a zero imaginary part.

    Returns
    -------
    torch.Tensor
        Real, of shape ``(2, *input.shape)``.

    Notes
    -----
    ``torchio.ScalarImage`` takes four axes, ``(channels, x, y, z)``, so a
    two-dimensional image is given a trailing axis of length one first.

    Examples
    --------
    >>> volume = learning.as_real(image)                      # (2, x, y, z)
    >>> subject = torchio.Subject(image=torchio.ScalarImage(tensor=volume))
    """
    if input.is_complex():
        return torch.stack([input.real, input.imag])
    return torch.stack([input, torch.zeros_like(input)])


def as_complex(input: torch.Tensor) -> torch.Tensor:
    """Combine a leading axis of real and imaginary parts into a complex tensor.

    Inverse of :func:`as_real`.

    Parameters
    ----------
    input : torch.Tensor
        Real, with a leading axis of length two.

    Returns
    -------
    torch.Tensor
        Complex, of shape ``input.shape[1:]``.

    Raises
    ------
    TypeError
        ``input`` is complex.
    ValueError
        The leading axis is not of length two.
    """
    if input.is_complex():
        raise TypeError(f"as_complex takes the real pair, and {input.dtype} is complex already")
    if 0 == input.ndim or 2 != input.shape[0]:
        raise ValueError(
            f"the real and imaginary parts are the leading axis, so it has length two, "
            f"and {tuple(input.shape)} does not"
        )
    return torch.complex(input[0], input[1])


#: Lower bound on a scale or an eigenvalue, so that an all-zero image is not divided by zero.
_TINY = 1e-12

_NORMALIZE = (None, "peak", "whiten")


class ComplexNet(nn.Module):
    """Network on real channels applied to complex images whose leading axes are channels.

    The input is ``(batch, *channels, *spatial)``, or
    ``(batch, *channels, frames, *spatial)`` with ``frames``.  The real and
    imaginary parts and the ``channels`` axes become the network's
    ``2 * prod(channels)`` real channels, real parts first; a frame axis stays
    in front of the spatial ones, as :class:`UNet` with ``frames=True`` takes
    it.  Subspace coefficient maps are thereby denoised jointly as one
    multi-channel image.

    Parameters
    ----------
    net : callable
        Called on real ``(batch, 2 * prod(channels), [frames,] *spatial)``
        tensors as ``net(x)`` or ``net(x, sigma)``, with any further keyword
        this module is called with, such as ``step``.  Returns its input's
        shape.
    spatial : {2, 3}, default=3
        Number of trailing spatial axes.
    channels : int, default=0
        Number of axes in front of the spatial ones (and of the frame axis)
        folded into the channels.
    frames : bool, default=False
        Whether the axis in front of the spatial ones is a frame axis.
    normalize : {"peak", "whiten", None}, default="peak"
        Scaling of each item around the call, with statistics taken without
        gradient.  ``"peak"`` divides by the peak modulus, so ``sigma`` is
        relative to it; ``"whiten"`` subtracts each real channel's mean and
        decorrelates the channels to unit variance, balancing coefficient maps
        whose energies differ by orders of magnitude, so ``sigma`` has no fixed
        unit; ``None`` leaves the item as it is.

    Raises
    ------
    TypeError
        ``net`` is not callable.
    ValueError
        An unknown ``normalize``.

    Examples
    --------
    >>> net = learning.ComplexNet(learning.UNet(10, steps=True), channels=1, normalize="whiten")
    >>> prior = priors.ImplicitPrior(net, step=True)      # (5, z, y, x) coefficient maps
    """

    def __init__(
        self,
        net,
        *,
        spatial: int = 3,
        channels: int = 0,
        frames: bool = False,
        normalize: str | None = "peak",
    ):
        super().__init__()
        if not callable(net):
            raise TypeError(f"a network is called as net(x), and {net!r} is not")
        if normalize not in _NORMALIZE:
            raise ValueError(f"normalize is one of {_NORMALIZE}, not {normalize!r}")
        self.net = net
        self.spatial = int(spatial)
        self.channels = int(channels)
        self.frames = bool(frames)
        self.normalize = normalize

    def forward(self, input: torch.Tensor, sigma=None, **keywords) -> torch.Tensor:
        """Apply the network.

        Parameters
        ----------
        input : torch.Tensor
            ``(batch, *channels, [frames,] *spatial)``.  A real tensor is taken as
            complex with a zero imaginary part.
        sigma : float or torch.Tensor, default=None
            Passed to the network as its second argument when given.
        **keywords
            Passed to the network.

        Returns
        -------
        torch.Tensor
            Of ``input``'s shape and dtype; the real part for a real ``input``.
        """
        inner = self.channels + int(self.frames) + self.spatial
        if input.ndim != 1 + inner:
            raise ValueError(
                f"this network takes (batch, {self.channels} channel axes"
                f"{', frames' if self.frames else ''}, {self.spatial} spatial axes), and "
                f"{tuple(input.shape)} is not that"
            )
        real = not input.is_complex()
        x = input.to(torch.complex64) if real else input
        n = x.shape[0]
        rest = x.shape[1 + self.channels :]
        planes = torch.stack([x.real, x.imag], 1).reshape(n, -1, *rest).float()

        planes, undo = self._normalize(planes)
        made = (
            self.net(planes, **keywords) if sigma is None else self.net(planes, sigma, **keywords)
        )
        if made.shape != planes.shape:
            raise ValueError(
                f"the network answered {tuple(made.shape)} for {tuple(planes.shape)}; it returns "
                "the channels and shape it was given"
            )
        made = undo(made).reshape(n, 2, *x.shape[1:])
        out = torch.complex(made[:, 0], made[:, 1])
        return out.real.to(input.dtype) if real else out.to(input.dtype)

    def _normalize(self, planes: torch.Tensor):
        """``planes`` brought to order unity, and the map taking the network's output back."""
        if self.normalize is None:
            return planes, lambda made: made
        n, c = planes.shape[:2]
        flat = planes.detach().reshape(n, c, -1)
        shape = (n, c, *(1,) * (planes.ndim - 2))
        if "peak" == self.normalize:
            modulus = flat.reshape(n, 2, -1).square().sum(1).sqrt()
            scale = modulus.amax(-1).clamp_min(_TINY).reshape(n, *(1,) * (planes.ndim - 1))
            return planes / scale, lambda made: made * scale

        mean = flat.mean(-1)
        centred = flat - mean[..., None]
        covariance = centred @ centred.transpose(1, 2) / flat.shape[-1]
        values, vectors = torch.linalg.eigh(covariance)
        values = values.clamp_min(_TINY * values.amax(-1, keepdim=True).clamp_min(_TINY))
        white = vectors @ torch.diag_embed(values.rsqrt()) @ vectors.transpose(1, 2)
        colour = vectors @ torch.diag_embed(values.sqrt()) @ vectors.transpose(1, 2)

        def mix(matrix, v):
            return (matrix @ v.reshape(n, c, -1)).reshape(v.shape)

        return (
            mix(white, planes - mean.reshape(shape)),
            lambda made: mix(colour, made) + mean.reshape(shape),
        )
