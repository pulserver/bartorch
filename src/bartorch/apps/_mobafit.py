"""``mobafit`` as a pipeline: a signal model, and Gauss-Newton per voxel."""

from __future__ import annotations

import math
from typing import Any

import torch

from bartorch import nlop, optim
from bartorch.apps._moba import _inside

__all__ = ["mobafit"]

#: Gauss-Newton steps.  The command's own default is five (mobafit.c:155),
#: which it can afford because its coefficients are the model's own variables
#: scaled to order one by ``--scale``.  BlochSim's parameterisation is bounded
#: instead, and a bounded variable moves slowly while the Tikhonov weight is
#: large: on a mono-exponential decay of a known T2, five steps answer 430 ms
#: for 60 and for 110 alike, ten come within three per cent, and twenty are
#: exact to the millisecond.  So the default is what converges rather than
#: what the command writes.
_ITERATIONS = 20

#: BART's own ``--liniter`` default (mobafit.c:246), which is both the inner
#: solver's iteration count and the Gauss-Newton step's ``cgiter``.
_CG_MAXITER = 50

#: The channel whose value scales with the data.
_AMPLITUDE = "amplitude.real"


def _amplitude(model: nlop.SignalModel, x: torch.Tensor, factor: float) -> torch.Tensor:
    """``x`` with its amplitude channels multiplied by ``factor``."""
    if factor == 1.0:
        return x
    weights = torch.tensor(
        [factor if name.startswith("amplitude.") else 1.0 for name in model.names]
    )
    return x * weights.reshape(-1, *[1] * (x.dim() - 1)).to(x.dtype)


def mobafit(
    images: torch.Tensor,
    model: nlop.SignalModel,
    *,
    iterations: int = _ITERATIONS,
    cg_maxiter: int = _CG_MAXITER,
    inner: Any = None,
    alpha: float = 1.0,
    alpha_min: float = 0.0,
    redu: float = 2.0,
    magnitude: bool = False,
    start: torch.Tensor | None = None,
    reference: dict[str, Any] | None = None,
    **values: Any,
) -> dict[str, torch.Tensor]:
    """Fit a signal model to reconstructed contrast images, voxel by voxel.

    The method of BART's ``mobafit``: Gauss-Newton (:class:`bartorch.nlop.IRGNM`)
    over a :class:`~bartorch.nlop.SignalModel`.  The model is BlochSim's, in
    its bounded parameterization, so the result is named maps in physical
    units rather than the command's coefficients; the two minimize the same
    objective.

    Parameters
    ----------
    images : torch.Tensor
        Contrast images ``(contrasts, *voxels)``, C order, in the order of the
        model's acquisition, in any intensity units.
    model : bartorch.nlop.SignalModel
        Signal model of the acquisition, on the voxel shape of ``images``:
        :func:`~bartorch.nlop.MultiEcho`,
        :func:`~bartorch.nlop.InversionRecovery` or :func:`~bartorch.nlop.Bloch`.
    iterations : int, default=20
        Gauss-Newton steps.  The command's 5 suffice for its scaled
        coefficients, not for the bounded parameterization.
    cg_maxiter : int, default=50
        Conjugate-gradient iterations per linearized problem (``--liniter``).
    inner : solver, default=None
        Solver from :mod:`bartorch.optim` for the linearized problem, whose
        regularizers penalize the maps; ``None`` is conjugate gradients.
    alpha : float, default=1.0
        Tikhonov weight of the first Gauss-Newton step.
    alpha_min : float, default=0.0
        Limit the weight decays towards.
    redu : float, default=2.0
        Decay factor of the weight per step.
    magnitude : bool, default=False
        Fit the model's magnitude to the images' magnitude (``-a``).
    start : torch.Tensor, default=None
        Starting point in the model's domain, as
        :meth:`~bartorch.nlop.SignalModel.initial` packs it, with the amplitude
        in the units of ``images``; built from ``**values`` when ``None``.
    reference : dict of str, default=None
        Values each step is regularized towards, ``{name: value}`` in physical
        units with an amplitude in the units of ``images``; unnamed properties
        take the model's defaults, and values are clamped strictly inside the
        model's bounds.  ``None`` regularizes towards zero in the model's
        variables, as the command does: the middle of each bound and no
        amplitude.
    **values
        Starting values ``name=value`` in physical units, as
        :meth:`~bartorch.nlop.SignalModel.initial` takes them.

    Returns
    -------
    dict of str to torch.Tensor
        Fitted maps in physical units, keyed by unknown.  Voxels whose data is
        zero in every contrast keep their starting values.

    Notes
    -----
    For a model with an amplitude, the images are divided by their peak
    magnitude for the fit and the amplitude is scaled back.

    Examples
    --------
    >>> M = nlop.MultiEcho([12.5 * (echo + 1) for echo in range(8)], (128, 128))
    >>> maps = mobafit(images, M, T2=80.0)
    >>> maps["T2"]
    """
    forward: Any = model
    if magnitude:
        forward = nlop.Abs(model.oshape) @ model

    x0 = model.initial(**values) if start is None else start
    if reference is not None:
        reference = _inside(model, reference)

    # The steps start from an amplitude of order one and are regularized in
    # the model's variables, so the data is fitted at unit peak; an amplitude
    # given in data units, and the one fitted, cross by the same factor.
    peak = float(images.abs().max())
    scale = peak if _AMPLITUDE in model.names and math.isfinite(peak) and peak > 0 else 1.0
    given = start is not None or "amplitude" in values
    begin = _amplitude(model, x0, 1.0 / scale) if given else x0
    if reference is not None and "amplitude" in reference:
        reference["amplitude"] = torch.as_tensor(reference["amplitude"]) / scale
    xref = None if reference is None else model.initial(**reference)

    solver = nlop.IRGNM(
        iterations=iterations,
        alpha=alpha,
        alpha_min=alpha_min,
        redu=redu,
        cg_maxiter=cg_maxiter,
        inner=optim.CG(maxiter=cg_maxiter) if inner is None else inner,
    )
    data = images.reshape(forward.oshape) / scale
    if magnitude:
        data = data.abs().to(data.dtype)
    fitted = _amplitude(model, solver(data, forward, x0=begin, xref=xref), scale)

    # A voxel with no signal at all constrains nothing, and a Gauss-Newton
    # step on it walks wherever the bounds allow; the command skips such a
    # patch and leaves it at zero, so the fit is put back to where it started.
    empty = images.reshape(forward.oshape).abs().amax(dim=0) == 0
    if bool(empty.any()):
        fitted = torch.where(empty, x0.expand_as(fitted), fitted)

    return model.split(fitted)
