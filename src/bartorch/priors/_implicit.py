"""A denoiser in place of a regularization term."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ImplicitPrior"]


def _batched(apply, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Apply ``apply`` item by item over a leading batch axis, when ``x`` has one."""
    if x.ndim == len(shape) + 1:
        return torch.stack([apply(item) for item in x])
    return apply(x)


#: Lower bound on a modulus, so that an all-zero image is not divided by zero.
_TINY = 1e-12

_PARTS = ("channels", "separate", "magnitude")


class _Planes(nn.Module):
    """A network on real ``(n, channels, *spatial)`` planes, applied to a complex image.

    The last ``spatial`` axes are the network's image; the leading axis is a
    batch, and the axes between are folded into the network's batch axis.  The
    complex values are laid out as real planes according to ``parts``; a
    single plane is repeated across three channels for an RGB network and the
    three returned channels averaged; with ``normalize`` each item of the
    leading axis is divided by its own peak modulus before the call and
    multiplied by it afterwards.  The scale is detached, and ``sigma`` is
    passed in the units of the scaled image.  A real input is treated as a
    complex image with zero imaginary part, and the real part is returned.
    """

    def __init__(
        self,
        net,
        *,
        spatial: int = 2,
        channels: int = 1,
        parts: str | None = None,
        normalize: bool = True,
    ):
        super().__init__()
        if not callable(net):
            raise TypeError(f"a denoiser is called as net(x) or net(x, sigma), and {net!r} is not")
        spatial = int(spatial)
        if spatial not in (2, 3):
            raise ValueError(
                f"a network takes planes or volumes, so spatial is 2 or 3, not {spatial}"
            )
        channels = int(channels)
        if parts is None:
            parts = "channels" if 2 == channels else "separate"
        if parts not in _PARTS:
            raise ValueError(f"parts is one of {_PARTS}, not {parts!r}")
        if "channels" == parts and 2 != channels:
            raise ValueError(
                "parts='channels' hands the real and imaginary planes to the network's "
                f"channels, so it takes two of them, not {channels}"
            )
        if "channels" != parts and channels not in (1, 3):
            raise ValueError(
                f"parts={parts!r} hands the network one plane, which a grayscale network takes "
                f"as it is and an RGB one takes repeated; {channels} channels is neither"
            )

        self.net = net
        self.spatial = spatial
        self.channels = channels
        self.parts = parts
        self.normalize = bool(normalize)

    def forward(self, input: torch.Tensor, sigma=None, **keywords) -> torch.Tensor:
        """Denoise ``input``, returning its own shape and dtype.

        Parameters
        ----------
        input : torch.Tensor
            Complex or real, of shape ``(batch, *rest, *spatial)``.  The
            leading axis is the batch over which each scale is measured; a
            tensor of exactly ``spatial`` axes is a single image.
        sigma : float or torch.Tensor, default=None
            Passed to the network as its second argument when supplied, and
            omitted from the call otherwise.
        **keywords
            Passed to the network as they are, such as ``step``.
        """
        if input.ndim < self.spatial:
            raise ValueError(
                f"an image for this denoiser carries at least its {self.spatial} spatial axes, "
                f"and {tuple(input.shape)} has {input.ndim}"
            )
        real = not input.is_complex()
        x = input.to(torch.complex64) if real else input
        scale = self._scale(x)
        out = self._denoise(x / scale, sigma, keywords) * scale
        return out.real.to(input.dtype) if real else out.to(input.dtype)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Peak modulus of each item of the leading axis, shaped to divide ``x``."""
        if not self.normalize:
            return torch.ones((), dtype=x.real.dtype, device=x.device)
        lead = 1 if x.ndim > self.spatial else 0
        peak = x.detach().abs().reshape(*x.shape[:lead], -1).amax(-1).clamp_min(_TINY)
        return peak.reshape(*x.shape[:lead], *(1,) * (x.ndim - lead))

    def _denoise(self, x: torch.Tensor, sigma, keywords) -> torch.Tensor:
        """Apply the network to the planes ``parts`` specifies and recombine the result."""
        spatial = tuple(x.shape[-self.spatial :])

        if "magnitude" == self.parts:
            modulus = x.abs()
            phase = x / modulus.clamp_min(_TINY).to(x.dtype)
            made = self._net(modulus.reshape(-1, 1, *spatial), sigma, 1, keywords)
            return made.reshape(x.shape).to(x.dtype) * phase

        if "separate" == self.parts:
            pair = torch.stack([x.real, x.imag]).reshape(-1, 1, *spatial)
            made = self._net(pair, sigma, 1, keywords).reshape(2, -1)
            return torch.complex(made[0], made[1]).reshape(x.shape)

        planes = torch.stack([x.real, x.imag], -self.spatial - 1).reshape(-1, 2, *spatial)
        made = self._net(planes, sigma, 2, keywords).movedim(1, 0).reshape(2, -1)
        return torch.complex(made[0], made[1]).reshape(x.shape)

    def _net(self, planes: torch.Tensor, sigma, wanted: int, keywords) -> torch.Tensor:
        """A single call, matching the network's channel count and checking its output."""
        if 3 == self.channels and 1 == planes.shape[1]:
            planes = planes.repeat(1, 3, *(1,) * self.spatial)
        made = (
            self.net(planes, **keywords) if sigma is None else self.net(planes, sigma, **keywords)
        )
        if 3 == self.channels and 3 == made.shape[1]:
            made = made.mean(1, keepdim=True)
        if made.shape[1] != wanted or made.shape[2:] != planes.shape[2:]:
            raise ValueError(
                f"the network answered {tuple(made.shape)} for {tuple(planes.shape)}; a denoiser "
                f"returns {wanted} channel(s) on the shape it was given"
            )
        return made


class ImplicitPrior(nn.Module):
    """Denoiser in place of a regularization term's proximal operator (plug-and-play).

    The proximal step is ``denoiser(x, sigma)``, or ``denoiser(x)`` without a
    ``sigma``, independently of the solver's step size.  Unlike BART's proximal
    operators it is differentiable, so a solve containing it can be
    differentiated end to end.  The proximal solvers of :mod:`bartorch.optim`
    take it in place of a term.

    With ``transform``, the term is :math:`g(G x)`: the denoiser is applied on
    the codomain of :math:`G`, and the alternating-direction and primal-dual
    iterations split the variable at :math:`G x`, as for a BART term with a
    transform.  This places a prior learned in another representation, such as
    contrast-weighted images synthesized from subspace coefficients.

    Parameters
    ----------
    denoiser : callable
        Called as ``denoiser(x)`` or ``denoiser(x, sigma)``, with ``step=k``
        added under ``step``.  Without ``spatial``, ``x`` is the complex tensor
        ``(batch, *shape)``, with ``shape`` the image's or the codomain of
        :math:`G`; with ``spatial``, a network on real
        ``(n, channels, *spatial)`` tensors of order unity, such as a
        ``deepinv`` or ``monai`` model.
    sigma : float or sequence of float, default=None
        Noise level passed to the denoiser, in its own units.  A sequence is a
        schedule, one per iteration, the last repeated beyond its end.  Held as
        a frozen :class:`torch.nn.Parameter`.
    transform : LinearOperator, default=None
        :math:`G`, with domain the image; the identity when omitted.
    spatial : {2, 3}, default=None
        Trailing axes the network takes; the axes in front are folded into its
        batch axis.  ``None`` passes the complex tensor unconverted.
    channels : {1, 2, 3}, default=1
        Input channels of the network: 1 for grayscale, 3 for RGB (the plane is
        repeated and the three outputs averaged), 2 for the real and imaginary
        parts jointly.  Only with ``spatial``.
    parts : {"channels", "separate", "magnitude"}, default=None
        Real and imaginary parts as two channels, each part denoised separately
        in one call, or the modulus denoised with the phase kept.
        ``"channels"`` for two channels and ``"separate"`` otherwise when
        omitted.  Only with ``spatial``.
    normalize : bool, default=True
        Divide each image by its peak modulus around the call, so ``sigma`` is
        relative to that peak.  Only with ``spatial``.
    step : bool, default=False
        Pass the iteration index to the denoiser as ``step``.

    Raises
    ------
    TypeError
        ``denoiser`` is not callable.
    ValueError
        ``channels``, ``parts`` or ``normalize`` without ``spatial``, or an
        inconsistent combination of them; a ``sigma`` that is empty or not
        one-dimensional.

    Notes
    -----
    A ``sigma`` schedule or ``step=True`` makes each iteration a different
    map (annealed plug-and-play, or a network shared by the iterations of an
    unrolled reconstruction, such as :class:`bartorch.learning.UNet` with
    ``steps=True``); :class:`bartorch.optim.FixedPoint` refuses both.

    Examples
    --------
    >>> from deepinv.models import DRUNet
    >>> prior = priors.ImplicitPrior(DRUNet(in_channels=1, out_channels=1), sigma=0.05, spatial=2)
    >>> image = optim.fista(y, A, prior)
    >>> priors.ImplicitPrior(denoiser, transform=linop.MultiplySum(basis, ...))
    """

    #: Takes a batch whole rather than item by item.
    _batches = True
    #: Told the iteration index by the blocks.
    _iterates = True

    def __init__(
        self,
        denoiser,
        sigma: float | None = None,
        *,
        transform=None,
        spatial: int | None = None,
        channels: int = 1,
        parts: str | None = None,
        normalize: bool = True,
        step: bool = False,
    ):
        super().__init__()
        if not callable(denoiser):
            raise TypeError(f"a denoiser is called as denoiser(x, sigma), and {denoiser!r} is not")
        if spatial is not None:
            denoiser = _Planes(
                denoiser, spatial=spatial, channels=channels, parts=parts, normalize=normalize
            )
        elif (channels, parts, normalize) != (1, None, True):
            raise ValueError(
                "channels, parts and normalize describe a network's planes, so they need spatial"
            )
        self.denoiser = denoiser
        self.transform = transform
        if sigma is not None:
            sigma = torch.as_tensor(sigma, dtype=torch.float32)
            if sigma.ndim > 1 or 0 == sigma.numel():
                raise ValueError("sigma is one noise level, or a sequence of one per iteration")
            sigma = nn.Parameter(sigma.clone(), requires_grad=False)
        self.sigma = sigma
        self.step = bool(step)

    @property
    def stationary(self) -> bool:
        """Whether every iteration applies the same denoiser: no schedule and no index."""
        return not self.step and (self.sigma is None or 0 == self.sigma.ndim)

    def prox(
        self, x: torch.Tensor, gamma=1.0, *, image_shape=None, iteration: int = 0
    ) -> torch.Tensor:
        """Apply the denoiser to ``x``; ``gamma`` is ignored.

        Parameters
        ----------
        x : tensor
            Of :meth:`prox_shape`, or with a leading batch axis.
        gamma : float, default=1.0
            Ignored: the denoising strength is ``sigma``.
        image_shape : tuple of int, default=None
            Image shape, which tells a single ``x`` from a batch; without it ``x``
            is a single item.
        iteration : int, default=0
            Iteration index, which selects the ``sigma`` of a schedule and is passed
            as ``step`` under ``step=True``.

        Returns
        -------
        torch.Tensor
            Of ``x``'s shape.
        """
        shape = None if image_shape is None else self.prox_shape(image_shape)
        single = shape is None or x.ndim == len(shape)
        batch = x[None] if single else x
        keywords = {"step": int(iteration)} if self.step else {}
        if self.sigma is None:
            out = self.denoiser(batch, **keywords)
        else:
            sigma = (
                self.sigma
                if 0 == self.sigma.ndim
                else self.sigma[min(int(iteration), self.sigma.numel() - 1)]
            )
            sigma = sigma if sigma.requires_grad else float(sigma)
            out = self.denoiser(batch, sigma, **keywords)
        return out[0] if single else out

    def prox_shape(self, image_shape) -> tuple[int, ...]:
        """Shape the denoiser is applied on: the image's, or the codomain of ``transform``."""
        if self.transform is None:
            return tuple(image_shape)
        self._check(image_shape)
        return tuple(self.transform.oshape)

    def apply_transform(self, x: torch.Tensor, image_shape=None, mode: str = "forward"):
        """Apply :math:`G`, :math:`G^H` or :math:`G^H G` to ``x``, recorded for autograd.

        The identity without ``transform``.

        Parameters
        ----------
        x : torch.Tensor
            Input of the mode, with or without a leading batch axis, applied item
            by item.
        image_shape : tuple of int, default=None
            Image shape, checked against the domain of :math:`G`.
        mode : {"forward", "adjoint", "normal"}, default="forward"
            :math:`G`, :math:`G^H` or :math:`G^H G`.

        Returns
        -------
        torch.Tensor
        """
        if self.transform is None:
            return x
        from bartorch.linop._autograd import apply_adjoint, apply_forward, apply_normal

        if image_shape is None:
            image_shape = tuple(self.transform.ishape)
        self._check(image_shape)
        G = self.transform
        one, shape = {
            "forward": (apply_forward, tuple(G.ishape)),
            "adjoint": (apply_adjoint, tuple(G.oshape)),
            "normal": (apply_normal, tuple(G.ishape)),
        }[mode]
        return _batched(lambda v: one(G, v), x, shape)

    def transform_is_identity(self, image_shape) -> bool:
        """Whether the term has no ``transform``."""
        return self.transform is None

    def rewind(self, image_shape) -> None:
        """Do nothing: the term draws no random shifts."""
        return None

    def _check(self, image_shape) -> None:
        """Check that the transform's domain is the image the solve is over."""
        if tuple(self.transform.ishape) != tuple(image_shape):
            raise ValueError(
                f"{self!r}'s transform takes {tuple(self.transform.ishape)}, and the image is "
                f"{tuple(image_shape)}"
            )
