"""Denoisers, for plug-and-play priors."""

from __future__ import annotations

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = ["nlmeans", "rof", "tgv"]

# BART's three spatial dims, the last three axes of a C-order tensor.
_FFT_FLAGS = 7
# `tgv` keeps its auxiliary variables along BART's last dim.
_MAX_NDIM = 15
# `--tvscales` holds at most this many values.
_MAX_SCALES = 5


def _axes_tuple(axes: int | tuple[int, ...]) -> tuple[int, ...]:
    return (axes,) if isinstance(axes, int) else tuple(axes)


def _check_ndim(input: torch.Tensor, command: str) -> None:
    if input.ndim > _MAX_NDIM:
        raise ValueError(f"bart {command} takes at most {_MAX_NDIM} axes, got {input.ndim}")


def _scales(values, axes, ndim: int, option: str) -> tuple[float, ...] | None:
    """Per-axis scales, reordered from ``axes`` to BART's bit order."""
    if values is None:
        return None
    axes = _axes_tuple(axes)
    values = tuple(float(v) for v in values)
    if len(values) != len(axes):
        raise ValueError(f"{option} needs one value per axis in {axes}, got {len(values)}")
    if len(values) > _MAX_SCALES:
        raise ValueError(f"{option} takes at most {_MAX_SCALES} axes")
    if 0.0 in values:
        # BART drops trailing zeros and then fails an assertion on the count.
        raise ValueError(f"{option} values must be nonzero")
    bits = [axes_flags(a, ndim) for a in axes]
    return tuple(v for _, v in sorted(zip(bits, values, strict=True)))


def _image(out: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """The denoised image: the first block of an output that appends auxiliary variables."""
    return out.reshape(-1)[: input.numel()].reshape(input.shape)


@curated("rof")
def rof(input: torch.Tensor, lamda: float, axes: int | tuple[int, ...]) -> torch.Tensor:
    r"""Total-variation (Rudin-Osher-Fatemi) denoising along ``axes``, BART's ``rof``.

    Approximates

    .. math::

        \min_x \; \tfrac12 \| x - y \|_2^2 + \lambda \sum_r \| (D x)_r \|_2

    with :math:`D` the periodic first differences along ``axes`` and the inner
    norm over the derivative directions (isotropic total variation).  BART runs
    50 ADMM iterations with :math:`\rho = 0.1`, so a large :math:`\lambda` does
    not reach the constant minimizer; the mean of ``input`` is preserved.

    Parameters
    ----------
    input : torch.Tensor
        Image :math:`y`.
    lamda : float
        :math:`\lambda`, in the units of ``input``'s values.
    axes : int or tuple of int
        Axes along which differences are taken.

    Returns
    -------
    torch.Tensor
        Denoised image, of ``input``'s shape.
    """
    return dispatch("rof", [input], None, _pos=[float(lamda), axes_flags(axes, input.ndim)])


@curated("tgv")
def tgv(
    input: torch.Tensor,
    lamda: float,
    axes: int | tuple[int, ...],
    *,
    alpha: tuple[float, float] | None = None,
    tvscales: tuple[float, ...] | None = None,
) -> torch.Tensor:
    r"""Second-order total generalized variation denoising along ``axes``, BART's ``tgv``.

    Approximates

    .. math::

        \min_{x, z} \; \tfrac12 \| x - y \|_2^2
            + \lambda \bigl( \alpha_1 \| D x - z \|_1 + \alpha_0 \| E z \|_1 \bigr)

    with :math:`D` the periodic gradient along ``axes``, :math:`z` a vector
    field and :math:`E z = (D z + (D z)^T) / 2` its symmetrized gradient; each
    :math:`\| \cdot \|_1` sums over voxels an :math:`\ell_2` norm over
    derivative directions.  BART runs 100 ADMM iterations with
    :math:`\rho = 0.1`.  The mean of ``input`` is preserved.

    Parameters
    ----------
    input : torch.Tensor
        Image :math:`y`, of at most 15 axes.
    lamda : float
        :math:`\lambda`, in the units of ``input``'s values.
    axes : int or tuple of int
        Axes along which differences are taken.
    alpha : tuple of float, default=None
        :math:`(\alpha_1, \alpha_0)`; BART's ``(1, sqrt(3))`` when omitted.
    tvscales : tuple of float, default=None
        Nonzero weight of the derivative along each axis of ``axes``, in the
        same order, applied in both :math:`D` and :math:`E`.

    Returns
    -------
    torch.Tensor
        Denoised image :math:`x`, of ``input``'s shape.

    Raises
    ------
    ValueError
        More than 15 axes; ``tvscales`` not one nonzero value per axis, or more
        than five.
    """
    _check_ndim(input, "tgv")
    out = dispatch(
        "tgv",
        [input],
        None,
        _pos=[float(lamda), axes_flags(axes, input.ndim)],
        alpha=None if alpha is None else tuple(float(a) for a in alpha),
        tvscales=_scales(tvscales, axes, input.ndim, "tvscales"),
    )
    return _image(out, input)


@curated("nlmeans")
def nlmeans(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    patch_length: int | None = None,
    patch_distance: int | None = None,
    h: float | None = None,
    a: float | None = None,
) -> torch.Tensor:
    r"""Non-local means filter along ``axes``, BART's ``nlmeans``.

    Each voxel becomes :math:`\sum_j w_{ij} y_j / \sum_j w_{ij}` over the
    voxels :math:`j` within ``patch_distance`` of it along every axis of
    ``axes``, with

    .. math::

        w_{ij} = \exp\Bigl( -\frac{\sum_o g_o |y_{i+o} - y_{j+o}|^2}{2 h^2 \|g\|_2} \Bigr),

    where :math:`o` runs over a patch of ``patch_length`` per axis and
    :math:`g` is a Gaussian of standard deviation ``a`` over it.  The image is
    reflected at its edges.

    Parameters
    ----------
    input : torch.Tensor
        Image :math:`y`.
    axes : int or tuple of int
        Axes the patches and the search span.
    patch_length : int, default=None
        Odd patch length per axis; BART's 5 when omitted.
    patch_distance : int, default=None
        Search radius per axis; BART's 5 when omitted.  Zero returns ``input``.
    h : float, default=None
        Filter strength, in the units of ``input``'s values; BART's 0.04 when
        omitted.  A large ``h`` makes the filter a box mean.
    a : float, default=None
        Standard deviation of :math:`g`, in voxels; ``(patch_length - 1) / 4``
        when omitted.

    Returns
    -------
    torch.Tensor
        Filtered image, of ``input``'s shape.

    Raises
    ------
    ValueError
        An even ``patch_length``.
    """
    if patch_length is not None and int(patch_length) % 2 != 1:
        raise ValueError(f"patch_length must be odd, got {patch_length}")
    return dispatch(
        "nlmeans",
        [input],
        None,
        _pos=[axes_flags(axes, input.ndim)],
        patch_length=None if patch_length is None else int(patch_length),
        patch_dist=None if patch_distance is None else int(patch_distance),
        H=None if h is None else float(h),
        a=None if a is None else float(a),
    )
