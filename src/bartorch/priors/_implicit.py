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
    """A denoiser substituted for a :mod:`bartorch.priors` regularizer.

    The proximal operator is ``denoiser(x, sigma)`` independently of the step,
    as plug-and-play regularization defines it, or ``denoiser(x)`` where no
    ``sigma`` is given.  Unlike a BART proximal operator this one is
    differentiable, so a solve containing it can be differentiated end to end.

    The denoiser receives a leading batch axis, of length one for a single
    image.  ``sigma`` is a :class:`torch.nn.Parameter`, fixed until
    ``requires_grad_()`` is called.

    A sequence of ``sigma`` is a schedule, one noise level per iteration and
    the last repeated beyond its end: the annealed plug-and-play iteration,
    which starts with a strong denoiser and weakens it as the data-consistent
    estimate improves.  With ``step`` the denoiser is also given the iteration
    index, as ``denoiser(x, sigma, step=k)`` or ``denoiser(x, step=k)``, which
    is how a network shared by the iterations of an unrolled reconstruction
    adapts to each (:class:`bartorch.learning.UNet` with ``steps=True``).
    Either makes each iteration a different map, which
    :class:`bartorch.optim.FixedPoint` refuses.

    ``transform`` is the linear operator :math:`G` of a term :math:`g(G x)`,
    as a regularizer built by BART also carries.  The denoiser is then applied
    on the codomain of :math:`G` rather than to the image, and the
    alternating-direction and primal-dual iterations split the variable at
    :math:`G x`, introducing one auxiliary variable and one dual variable per
    term and adding :math:`G^H G` to the x-update.  This accommodates a prior
    learned in a representation other than the optimization variable, such as
    contrast-weighted images obtained from subspace coefficient maps.  Without
    it :math:`G` is the identity and the denoiser is applied to the image.

    Parameters
    ----------
    denoiser : callable
        Called as ``denoiser(x)``, or as ``denoiser(x, sigma)`` when a
        ``sigma`` is given.  Without ``spatial`` it receives the complex
        ``(batch, *shape)`` tensor as it stands, where ``shape`` is the image's
        shape or, with a ``transform``, the codomain of :math:`G`.  With
        ``spatial`` it is an image-restoration network on real
        ``(n, channels, *spatial)`` planes of order unity -- a ``deepinv``,
        ``monai`` or local :class:`torch.nn.Module` -- and the conversion is
        done here.
    sigma : float or sequence of float, default=None
        The noise level the denoiser is asked for, in the units its own
        convention states; a sequence gives one per iteration.
    transform : LinearOperator, default=None
        :math:`G`, mapping the image to the domain the denoiser is applied on.
        Only the iterations given a term's transform use it; see
        :meth:`bartorch.priors.Regularizer.transform_is_identity`.
    spatial : {2, 3}, default=None
        Trailing axes the network operates on: 2 for a network trained on
        slices, 3 for one trained on volumes.  The axes in front of them are
        folded into the network's batch axis.  ``None`` passes the complex
        tensor unconverted.
    channels : {1, 2, 3}, default=1
        Input channels of the network: 1 for grayscale, 3 for RGB (a plane is
        repeated and the three outputs averaged), 2 for a network taking the
        real and imaginary planes jointly, as MoDL's does.  Only with
        ``spatial``.
    parts : {"channels", "separate", "magnitude"}, default=None
        How the complex values become real planes: the real and imaginary
        parts as the network's two channels, each part denoised on its own in
        one call over a doubled batch, or the modulus denoised with the phase
        kept.  Defaults to ``"channels"`` for two channels and ``"separate"``
        otherwise.  Only with ``spatial``.
    normalize : bool, default=True
        Scale each image to unit peak modulus around the call; ``sigma`` is
        then in units of that peak.  Only with ``spatial``.
    step : bool, default=False
        Pass the iteration index to the denoiser as ``step``.

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
        if self.transform is None:
            return tuple(image_shape)
        self._check(image_shape)
        return tuple(self.transform.oshape)

    def apply_transform(self, x: torch.Tensor, image_shape=None, mode: str = "forward"):
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
        return self.transform is None

    def rewind(self, image_shape) -> None:
        return None

    def _check(self, image_shape) -> None:
        """Check that the transform's domain is the image the solve is over."""
        if tuple(self.transform.ishape) != tuple(image_shape):
            raise ValueError(
                f"{self!r}'s transform takes {tuple(self.transform.ishape)}, and the image is "
                f"{tuple(image_shape)}"
            )
