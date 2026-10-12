"""The regularization terms, one class per BART regularizer.

Total generalized variation and the two infimal convolutions extend the
optimization variable, which BART counts across the whole set of terms, so
they cannot be built one at a time; :func:`bartorch.apps.pics`,
:class:`bartorch.optim.ADMM` and :class:`bartorch.optim.PRIDU` take them.
"""

from __future__ import annotations

from bartorch.priors._base import Regularizer

__all__ = [
    "FourierL1",
    "ImageNIHT",
    "ImaginaryL1",
    "ImaginaryL2",
    "InfimalConvolutionTGV",
    "InfimalConvolutionTV",
    "L1",
    "L2",
    "Laplace",
    "LocallyLowRank",
    "NonNegative",
    "TotalGeneralizedVariation",
    "TotalVariation",
    "Wavelet",
    "WaveletNIHT",
]

#: The wavelet families ``opt_reg_configure`` accepts.
_FAMILIES = ("haar", "dau2", "cdf44")


def _axes(axes) -> tuple[int, ...]:
    return (axes,) if isinstance(axes, int) else tuple(axes)


def _family(family: str) -> str:
    if family not in _FAMILIES:
        raise ValueError(f"wavelet family must be one of {_FAMILIES}, not {family!r}")
    return family


class _Weighted(Regularizer):
    """A term over ``axes``, acting jointly along ``joint_axes``, with a weight."""

    def __init__(self, axes, weight: float, joint_axes=()):
        self.axes = _axes(axes)
        self.joint_axes = _axes(joint_axes)
        self.weight = float(weight)


class Wavelet(_Weighted):
    r""":math:`\ell_1` norm of the wavelet coefficients (BART's ``W``).

    :math:`\lambda \| \Psi x \|_1`.  The wavelet transform :math:`\Psi` is
    inside the proximal operator, so :math:`G` is the identity.

    Parameters
    ----------
    axes : int or tuple of int
        Axes to transform, as indices into the image's shape.
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which a coefficient is kept or zeroed together.
    family : {'haar', 'dau2', 'cdf44'}, default='dau2'
        Wavelet family.
    randshift : bool, default=True
        Cycle-spin the transform by a random shift at each application.

    Examples
    --------
    >>> Wavelet(axes=(-1, -2), weight=0.005)
    """

    kind = "W"

    def __init__(
        self, axes, weight: float, joint_axes=(), *, family: str = "dau2", randshift: bool = True
    ):
        super().__init__(axes, weight, joint_axes)
        self.family = _family(family)
        self.randshift = bool(randshift)

    def _options(self) -> tuple[int, str, int]:
        return 8, self.family, int(self.randshift)

    def _settings(self) -> dict[str, object]:
        return {"family": self.family, "randshift": self.randshift}


class TotalVariation(_Weighted):
    r"""Isotropic total variation, :math:`\lambda \sum_r \| (\nabla x)_r \|_2` (BART's ``T``).

    :math:`G = \nabla` stacks the first differences along ``axes``; the
    :math:`\ell_2` norm is over the derivative directions.

    Parameters
    ----------
    axes : int or tuple of int
        Axes to differentiate, as indices into the image's shape.
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    """

    kind = "T"


class LocallyLowRank(_Weighted):
    r"""Locally low-rank penalty, the nuclear norm of image blocks (BART's ``L``).

    Parameters
    ----------
    axes : int or tuple of int
        Axes the blocks span, as indices into the image's shape.
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes forming the columns of each block's matrix.
    block : int, default=8
        Block edge length.
    randshift : bool, default=True
        Shift the block grid by a random offset at each application.
    overlapping : bool, default=False
        Fully overlapping blocks in place of a shifted grid.
    """

    kind = "L"

    def __init__(
        self,
        axes,
        weight: float,
        joint_axes=(),
        *,
        block: int = 8,
        randshift: bool = True,
        overlapping: bool = False,
    ):
        super().__init__(axes, weight, joint_axes)
        self.block = int(block)
        self.randshift = bool(randshift)
        self.overlapping = bool(overlapping)

    def _options(self) -> tuple[int, str, int]:
        return self.block, "dau2", 2 if self.overlapping else int(self.randshift)

    def _settings(self) -> dict[str, object]:
        return {"block": self.block, "randshift": self.randshift, "overlapping": self.overlapping}


class Laplace(_Weighted):
    r""":math:`\ell_1` norm of a Laplacian-filtered image (BART's ``P``).

    :math:`\lambda \| L x \|_1`, where :math:`G = L` is a 3 x 3 convolution
    with the kernel ``[[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]]``.

    Parameters
    ----------
    axes : tuple of int
        Exactly two axes to filter, as indices into the image's shape.
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    """

    kind = "P"


class FourierL1(_Weighted):
    r""":math:`\ell_1` norm of the Fourier coefficients, :math:`\lambda \| F x \|_1` (BART's ``F``).

    Parameters
    ----------
    axes : int or tuple of int
        Axes to transform, as indices into the image's shape.
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    """

    kind = "F"


class _Joint(Regularizer):
    """A term with no axes of its own."""

    def __init__(self, weight: float, joint_axes=()):
        self.joint_axes = _axes(joint_axes)
        self.weight = float(weight)


class L1(_Joint):
    r""":math:`\ell_1` norm of the image, :math:`\lambda \| x \|_1` (BART's ``I``).

    Parameters
    ----------
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    """

    kind = "I"


class ImaginaryL1(_Joint):
    r""":math:`\ell_1` norm of the imaginary part (BART's ``R1``).

    :math:`\lambda \| \operatorname{Im} x \|_1`.

    Parameters
    ----------
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    """

    kind = "R1"


class ImaginaryL2(_Joint):
    r"""Squared :math:`\ell_2` norm of the imaginary part (BART's ``R2``).

    :math:`\tfrac{\lambda}{2} \| \operatorname{Im} x \|_2^2`.

    Parameters
    ----------
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    """

    kind = "R2"


class L2(Regularizer):
    r"""Squared :math:`\ell_2` norm of the image (BART's ``Q``).

    :math:`\tfrac{\lambda}{2} \| x \|_2^2`: Tikhonov regularization as a
    proximal term.  :class:`~bartorch.optim.CG` takes the same weight as
    ``lambda_``.

    Parameters
    ----------
    weight : float
        :math:`\lambda`.
    """

    kind = "Q"

    def __init__(self, weight: float):
        self.weight = float(weight)


class NonNegative(Regularizer):
    """Projection onto images whose real and imaginary parts are non-negative (BART's ``S``).

    Each part is clamped at zero separately: ``-1 + 2i`` is projected to
    ``2i``.  The term takes no parameters.
    """

    kind = "S"


class _Counted(Regularizer):
    """A term that keeps a percentage of its entries, the largest, instead of thresholding."""

    def __init__(self, axes, count: int, joint_axes=()):
        self.axes = _axes(axes)
        self.joint_axes = _axes(joint_axes)
        self.count = int(count)


class WaveletNIHT(_Counted):
    """Keep the ``count`` per cent largest wavelet coefficients over ``axes`` (BART's ``H``).

    Parameters
    ----------
    axes : int or tuple of int
        Axes to transform, as indices into the image's shape.
    count : int
        Percentage of the coefficients kept, 0 to 100.
    joint_axes : int or tuple of int, default=()
        Axes along which a coefficient is kept or zeroed together.
    family : {'haar', 'dau2', 'cdf44'}, default='dau2'
        Wavelet family.
    randshift : bool, default=True
        Cycle-spin the transform by a random shift at each application.
    """

    kind = "H"

    def __init__(
        self, axes, count: int, joint_axes=(), *, family: str = "dau2", randshift: bool = True
    ):
        super().__init__(axes, count, joint_axes)
        self.family = _family(family)
        self.randshift = bool(randshift)

    def _options(self) -> tuple[int, str, int]:
        return 8, self.family, int(self.randshift)

    def _settings(self) -> dict[str, object]:
        return {"family": self.family, "randshift": self.randshift}


class ImageNIHT(_Counted):
    """Keep the ``count`` per cent largest image entries (BART's ``N``)."""

    kind = "N"


#: BART's `FFT_FLAGS`: the three spatial axes, which are the image's last
#: three in C order.
_FFT_FLAGS = 7


def _both_kinds_of_axis(term, ndim: int | None) -> None:
    """Refuse an infimal convolution that has no axis of each kind.

    ``ictv_reg`` and ``ictgv_reg`` (``iter/tgv.c``) each assert
    ``flags & FFT_FLAGS`` and ``flags & ~FFT_FLAGS``: the convolution splits
    the image into a part smooth over the spatial axes and a part smooth over
    the rest, so both must exist.  An assertion aborts the process, so the
    same question is asked here.  Plain total generalized variation asserts
    neither.
    """
    xflags, _ = term._flags(ndim)
    if 0 != (xflags & _FFT_FLAGS) and 0 != (xflags & ~_FFT_FLAGS):
        return
    kind = "spatial" if xflags & _FFT_FLAGS else "non-spatial"
    over = "" if ndim is None else f" of an image of {ndim} axes"
    raise ValueError(
        f"{term!r} needs at least one of the image's last three axes and at least one "
        f"axis before them: the infimal convolution separates what is smooth over the "
        f"one from what is smooth over the other, so it needs both to exist. "
        f"axes={term.axes}{over} gives only {kind} axes. BART states this as an "
        f"assertion in iter/tgv.c, which would end the process rather than raise"
    )


def _pair(values, name: str) -> tuple[float, float]:
    pair = tuple(float(v) for v in values)
    if len(pair) != 2:
        raise ValueError(f"{name} is a pair of weights, not {values!r}")
    return pair


class TotalGeneralizedVariation(_Weighted):
    r"""Second-order total generalized variation over ``axes`` (BART's ``G``).

    Minimized jointly over the image and an auxiliary vector field :math:`z`:

    .. math::

        \min_{z} \; \lambda \bigl( \alpha_1 \| \nabla x + z \|_1
                                 + \alpha_0 \| \mathcal{E} z \|_1 \bigr)

    with :math:`\mathcal{E}` the symmetrized gradient (BART's ``tgv_reg``).

    Accepted by :func:`bartorch.apps.pics`, :class:`bartorch.optim.ADMM` and
    :class:`bartorch.optim.PRIDU` only: the auxiliary variables extend the
    optimization variable, and BART counts them across the whole set of terms,
    so the term cannot be built alone.

    Parameters
    ----------
    axes : int or tuple of int
        Axes to differentiate, as indices into the image's shape.
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    alpha : tuple of float, default=(1.0, sqrt(3))
        :math:`(\alpha_1, \alpha_0)`.
    """

    kind = "G"
    _extends = True

    def __init__(self, axes, weight: float, joint_axes=(), *, alpha=(1.0, 3.0**0.5)):
        super().__init__(axes, weight, joint_axes)
        self.alpha = _pair(alpha, "alpha")

    def _settings(self) -> dict[str, object]:
        return {"alpha": self.alpha}


class InfimalConvolutionTV(_Weighted):
    r"""Infimal convolution of total variation over ``axes`` (BART's ``C``).

    The image is split into two components, each penalized by total variation:

    .. math::

        \min_{z} \; \lambda \bigl( \gamma_1 \| \nabla (x + z) \|_1
                                 + \gamma_2 \| \nabla z \|_1 \bigr)

    (BART's ``ictv_reg``).  The two gradients are the same operator: the
    anisotropic derivative scalings of ``pics --tvscales`` and
    ``--tvscales2`` are not carried by the term.

    Accepted by :func:`bartorch.apps.pics`, :class:`bartorch.optim.ADMM` and
    :class:`bartorch.optim.PRIDU` only: the auxiliary variables extend the
    optimization variable, and BART counts them across the whole set of terms,
    so the term cannot be built alone.

    ``axes`` must include at least one of the image's last three axes (BART's
    spatial axes) and at least one before them, such as subspace coefficients
    or the frames of a time series.

    Parameters
    ----------
    axes : int or tuple of int
        Axes to differentiate, as indices into the image's shape.
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    gamma : tuple of float, default=(1.0, 1.0)
        :math:`(\gamma_1, \gamma_2)`.

    Raises
    ------
    ValueError
        ``axes`` lacks a spatial or a non-spatial axis (raised when the term is
        used in a solve).
    """

    kind = "C"
    _extends = True

    def __init__(self, axes, weight: float, joint_axes=(), *, gamma=(1.0, 1.0)):
        super().__init__(axes, weight, joint_axes)
        self.gamma = _pair(gamma, "gamma")

    def _settings(self) -> dict[str, object]:
        return {"gamma": self.gamma}

    def _check(self, ndim: int | None) -> None:
        _both_kinds_of_axis(self, ndim)


class InfimalConvolutionTGV(_Weighted):
    r"""Infimal convolution of total generalized variation over ``axes`` (BART's ``V``).

    The image is split into two components as for
    :class:`InfimalConvolutionTV`, each penalized by second-order TGV with its
    own auxiliary vector field:

    .. math::

        \min_{z, u, w} \; \lambda \Bigl[
            \gamma_1 \bigl( \alpha_1 \| \nabla (x + z) + u \|_1
                          + \alpha_0 \| \mathcal{E} u \|_1 \bigr)
          + \gamma_2 \bigl( \alpha_1 \| \nabla z + w \|_1
                          + \alpha_0 \| \mathcal{E} w \|_1 \bigr) \Bigr]

    (BART's ``ictgv_reg``), with the same gradient for both components.

    Accepted by :func:`bartorch.apps.pics`, :class:`bartorch.optim.ADMM` and
    :class:`bartorch.optim.PRIDU` only: the auxiliary variables extend the
    optimization variable, and BART counts them across the whole set of terms,
    so the term cannot be built alone.

    ``axes`` must include at least one of the image's last three axes (BART's
    spatial axes) and at least one before them, such as subspace coefficients
    or the frames of a time series.

    Parameters
    ----------
    axes : int or tuple of int
        Axes to differentiate, as indices into the image's shape.
    weight : float
        :math:`\lambda`.
    joint_axes : int or tuple of int, default=()
        Axes along which entries are thresholded jointly.
    alpha : tuple of float, default=(1.0, sqrt(3))
        :math:`(\alpha_1, \alpha_0)`.
    gamma : tuple of float, default=(1.0, 1.0)
        :math:`(\gamma_1, \gamma_2)`.

    Raises
    ------
    ValueError
        ``axes`` lacks a spatial or a non-spatial axis (raised when the term is
        used in a solve).
    """

    kind = "V"
    _extends = True

    def __init__(
        self, axes, weight: float, joint_axes=(), *, alpha=(1.0, 3.0**0.5), gamma=(1.0, 1.0)
    ):
        super().__init__(axes, weight, joint_axes)
        self.alpha = _pair(alpha, "alpha")
        self.gamma = _pair(gamma, "gamma")

    def _settings(self) -> dict[str, object]:
        return {"alpha": self.alpha, "gamma": self.gamma}

    def _check(self, ndim: int | None) -> None:
        _both_kinds_of_axis(self, ndim)
