"""``mobafit`` as a pipeline: a signal model, fitted voxel by voxel."""

from __future__ import annotations

import math
from typing import Any

import torch

from bartorch import nlop

__all__ = ["mobafit"]

#: Levenberg-Marquardt steps.  A voxel leaves the loop as soon as it has
#: converged, so the count bounds the slowest voxel only.
_ITERATIONS = 20

#: First damping, as a multiple of the largest diagonal entry of each voxel's
#: Gauss-Newton curvature at the start.
_TAU = 1e-2

#: A voxel has converged when its gradient falls below this, or its step below
#: this relative to its variables.
_TOLERANCE = 1e-8

_TINY = torch.finfo(torch.float32).tiny

#: The channel whose value scales with the data.
_AMPLITUDE = "amplitude.real"


def _amplitude(names: tuple[str, ...], x: torch.Tensor, factor: float) -> torch.Tensor:
    """``x``, ``(..., channels)``, with its amplitude channels multiplied by ``factor``."""
    if factor == 1.0:
        return x
    weights = torch.tensor(
        [factor if name.startswith("amplitude.") else 1.0 for name in names],
        dtype=x.dtype,
        device=x.device,
    )
    return x * weights


def _evaluate(model: Any, x: torch.Tensor, data: torch.Tensor, magnitude: bool):
    """The residual and the Jacobian at ``x``, as real rows per voxel.

    ``x`` is ``(*voxels, channels)``, the shape a model's per-voxel maps are
    bound in; the residual comes back ``(voxels, rows)`` and the Jacobian
    ``(voxels, channels, rows)``.  The Jacobian is the model's own
    ``jacobian`` when it has one, otherwise one product per channel.
    """
    channels = x.shape[-1]
    predicted = model.A(x)
    build = getattr(model, "jacobian", None)
    if build is not None:
        blocks = build(x)
    else:
        eye = torch.eye(channels, dtype=x.dtype, device=x.device)
        blocks = torch.stack([model.A_jvp(x, eye[c].expand_as(x)) for c in range(channels)], -2)
    predicted = predicted.reshape(-1, predicted.shape[-1])
    blocks = blocks.reshape(-1, channels, blocks.shape[-1])
    if magnitude:
        # d|a| = Re(conj(a) da) / |a|.
        modulus = predicted.abs()
        unit = torch.where(modulus > 0, predicted / modulus.clamp_min(_TINY), 1.0)
        blocks = (unit.conj()[:, None, :] * blocks).real
        predicted = modulus
    residual = predicted - data
    if torch.is_complex(residual) or torch.is_complex(blocks):
        residual = residual.to(torch.complex64)
        blocks = blocks.to(torch.complex64)
        residual = torch.cat((residual.real, residual.imag), -1)
        blocks = torch.cat((blocks.real, blocks.imag), -1)
    return residual, blocks


def _levenberg_marquardt(
    model: Any,
    data: torch.Tensor,
    x: torch.Tensor,
    frozen: torch.Tensor,
    iterations: int,
    magnitude: bool,
) -> torch.Tensor:
    """Fit ``x``, ``(*voxels, channels)``, to ``data``, ``(voxels, contrasts)``.

    Each voxel takes a damped Gauss-Newton step, solved directly as a small
    least-squares system, accepted only where it lowers that voxel's residual,
    and moves its damping by the gain ratio (Nielsen's rule).  A voxel in
    ``frozen``, or one that has converged, takes no further step.  A model with
    ``select`` -- a copy over a voxel mask, maps bound on it included -- drops
    those voxels, so the late steps cost what is left; without it they stay in
    every evaluation.
    """
    voxels, channels = x.shape[:-1], x.shape[-1]
    select = getattr(model, "select", None)
    answer = x.reshape(-1, channels).clone()
    live = torch.arange(answer.shape[0], device=x.device)
    x = answer.clone()
    eye = torch.eye(channels, device=x.device)
    state: dict[str, torch.Tensor] = {}

    def retire():
        nonlocal model, x, data, frozen, live, voxels
        if select is None or not bool(frozen.any()):
            return
        answer[live[frozen]] = x[frozen]
        keep = ~frozen
        model = model.select(keep.reshape(voxels))
        x, data, live, frozen = x[keep], data[keep], live[keep], frozen[keep]
        for name, value in state.items():
            state[name] = value[keep]
        voxels = (int(keep.sum()),)

    residual, blocks = _evaluate(model, x.reshape(*voxels, channels), data, magnitude)
    damping = _TAU * (blocks @ blocks.mT).diagonal(dim1=-2, dim2=-1).amax(-1)
    state.update(
        residual=residual,
        blocks=blocks,
        cost=residual.square().sum(-1),
        damping=damping,
        rising=torch.full_like(damping, 2.0),
    )

    for _ in range(iterations):
        retire()
        if x.shape[0] == 0 or bool(frozen.all()):
            break
        residual, blocks, cost = state["residual"], state["blocks"], state["cost"]
        damping, rising = state["damping"], state["rising"]
        gradient = torch.einsum("vpt,vt->vp", blocks, residual)

        root = damping.sqrt()[:, None, None] * eye.to(blocks.dtype)
        system = torch.cat((blocks.mT, root), dim=-2)
        target = torch.cat((-residual, torch.zeros_like(x)), dim=-1)
        step = torch.linalg.lstsq(system, target[..., None]).solution.squeeze(-1)
        singular = ~torch.isfinite(step).all(-1)
        step = torch.where((singular | frozen)[:, None], 0.0, step)

        trial = x + step
        trial_residual, trial_blocks = _evaluate(
            model, trial.reshape(*voxels, channels), data, magnitude
        )
        trial_cost = trial_residual.square().sum(-1)
        predicted = (step * (damping[:, None] * step - gradient)).sum(-1)
        gain = torch.where(predicted > 0, (cost - trial_cost) / predicted.clamp_min(_TINY), 0.0)
        better = (gain > 0) & ~singular & ~frozen
        lowered = damping * (1.0 - (2.0 * gain - 1.0).pow(3)).clamp_min(1.0 / 3.0)
        damping = torch.where(better, lowered, torch.where(frozen, damping, damping * rising))
        rising = torch.where(better, 2.0, torch.where(frozen, rising, rising * 2.0))

        x = torch.where(better[:, None], trial, x)
        state.update(
            residual=torch.where(better[:, None], trial_residual, residual),
            blocks=torch.where(better[:, None, None], trial_blocks, blocks),
            cost=torch.where(better, trial_cost, cost),
        )

        short = step.norm(dim=-1) < _TOLERANCE * (x.norm(dim=-1) + _TOLERANCE)
        flat = gradient.abs().amax(-1) < _TOLERANCE
        # A voxel whose damping has overflowed takes no step a float can hold.
        frozen = frozen | (short & ~singular) | flat | ~torch.isfinite(damping)
        state.update(damping=torch.where(frozen, 1.0, damping), rising=rising)

    answer[live] = x
    return answer


@torch.no_grad()
def mobafit(
    images: torch.Tensor,
    model: nlop.ModelOperator,
    *,
    iterations: int = _ITERATIONS,
    magnitude: bool = False,
    start: torch.Tensor | None = None,
    **values: Any,
) -> dict[str, torch.Tensor]:
    """Fit a signal model to reconstructed contrast images, voxel by voxel.

    Levenberg-Marquardt over every voxel at once, each with its own damping and
    its own small linear system solved directly, over the model of a
    :class:`~bartorch.nlop.ModelOperator`.  The model is BlochSim's, in its
    bounded parameterization, so the result is named maps in physical units
    rather than BART's coefficients, and no iterate leaves the bounds.  The
    answer is a local minimum that depends on the starting point.

    Parameters
    ----------
    images : torch.Tensor
        Contrast images ``(contrasts, *voxels)``, C order, in the order of the
        model's acquisition, in any intensity units.
    model : bartorch.nlop.ModelOperator
        Signal model of the acquisition, on the voxel shape of ``images``:
        :func:`~bartorch.nlop.MultiEcho`,
        :func:`~bartorch.nlop.InversionRecovery` or :func:`~bartorch.nlop.Bloch`.
    iterations : int, default=20
        Levenberg-Marquardt steps; a voxel stops earlier once it has converged.
    magnitude : bool, default=False
        Fit the model's magnitude to the images' magnitude (``-a``).
    start : torch.Tensor, default=None
        Starting point in the model's domain, as
        :meth:`~bartorch.nlop.ModelOperator.initial` packs it, with the amplitude
        in the units of ``images``; built from ``**values`` when ``None``.
    **values
        Starting values ``name=value`` in physical units, as
        :meth:`~bartorch.nlop.ModelOperator.initial` takes them.

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
    inner = model.model
    names = tuple(model.names)
    x0 = model.initial(**values) if start is None else start

    # The damping is scaled per voxel, but a start of order one is closer to
    # data at unit peak; an amplitude given in data units, and the one fitted,
    # cross by the same factor.
    peak = float(images.abs().max())
    scale = peak if _AMPLITUDE in names and math.isfinite(peak) and peak > 0 else 1.0
    given = start is not None or "amplitude" in values

    # BART's ``(channels, *voxels)`` complex to the model's ``(*voxels, channels)`` real.
    begin = x0.real.movedim(0, -1).contiguous()
    if given:
        begin = _amplitude(names, begin, 1.0 / scale)
    data = images.reshape(images.shape[0], -1).mT / scale
    if magnitude:
        data = data.abs()
    data = data.to(begin.device)

    # A voxel with no signal at all constrains nothing; the command skips such
    # a patch, so the fit leaves it where it started.
    empty = data.abs().amax(-1) == 0
    fitted = _levenberg_marquardt(inner, data, begin, empty, iterations, magnitude)
    fitted = _amplitude(names, fitted, scale)
    original = x0.real.movedim(0, -1).reshape(fitted.shape)
    fitted = torch.where(empty[:, None], original, fitted)

    out = fitted.reshape(begin.shape).movedim(-1, 0).to(x0.dtype)
    return model.split(out)
