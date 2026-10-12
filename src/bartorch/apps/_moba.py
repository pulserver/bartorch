"""``moba`` as a pipeline: a signal model inside the encoding, and Gauss-Newton over k-space."""

from __future__ import annotations

from typing import Any

import torch

from bartorch import linop, nlop, optim
from bartorch._operator import Shape
from bartorch.nlop._base import NonlinearOperator, _chain
from bartorch.nlop._basic import Multiply

__all__ = ["moba"]

#: Gauss-Newton steps.  The command takes eight (moba/moba.c:17) over its own
#: coefficients, which ``--scale`` brings to order one; a bounded variable
#: moves slowly while the Tikhonov weight is large, so BlochSim's
#: parameterisation needs more.
_ITERATIONS = 20

#: Conjugate-gradient steps per linearized problem, the command's ``--liniter``
#: default (mobafit.c:246).
_CG_MAXITER = 50

#: Conjugate-gradient steps of the contrast images the data scaling is read off.
_SCALE_CG_MAXITER = 10

#: ``(a, b)`` of the coil weighting, the command's ``--sobolev_a`` and
#: ``--sobolev_b`` defaults (moba/moba.c:28-29).
_SOBOLEV = (880.0, 32.0)


def _spatial(voxels: Shape) -> Shape:
    """The voxel shape with BART's third spatial axis written out."""
    return (1, *voxels) if 2 == len(voxels) else tuple(voxels)


def _estimated_pattern(kspace: torch.Tensor) -> torch.Tensor:
    """Ones where any coil has a sample, as the command estimates it without ``-p``."""
    return (kspace != 0).any(dim=1, keepdim=True).to(torch.complex64)


def _joint(sense: nlop.NonlinearSense, model: NonlinearOperator) -> NonlinearOperator:
    """``noir2_join`` with the signal model in front of the image.

    The same four stages :class:`~bartorch.nlop.NonlinearSense` is made of --
    the image's linear part, the Sobolev-weighted coils, their product and the
    transform -- with the model's contrast images as the image.  Written as
    a product ahead of a linear transform, which is the composition a
    Gauss-Newton step recognises and applies through the transform's normal.
    Inputs are the maps, then the coil coefficients.
    """
    image, coils = sense.image, sense.coils
    product = Multiply(image.oshape, coils.oshape)
    made = _chain(coils.to_nonlinear(), product, output=0, input=1)
    made = _chain(image.to_nonlinear(), made, output=0, input=0)
    made = _chain(model._reshape_output(0, image.ishape), made, output=0, input=1)
    made = made._permute_inputs([1, 0])
    return _chain(made, sense.transform.to_nonlinear(), output=0, input=0)


def _scale(model: nlop.ModelOperator, images) -> float:
    """What the data is divided by: BART's scaling rule over images of the data.

    The rule is the one :func:`bartorch.optim.data_scaling` applies, so that
    the amplitude the fit starts from and the one it is fitted to are of the
    same order.  Only a model carrying an amplitude can take its data scaled,
    since the amplitude is what absorbs the factor; one without is fitted as
    it stands.
    """
    if "amplitude.real" not in model.names:
        return 1.0
    made = images()
    return optim.data_scaling(made, A=linop.Identity(tuple(made.shape))) or 1.0


#: How far inside its bound a reference value is clamped, as a fraction of the
#: interval (or of the bound's magnitude, for a one-sided one).  A value on a
#: bound has no image in the bounded parameterisation.
_INSIDE = 1e-3


def _inside(model: nlop.ModelOperator, named: dict[str, Any]) -> dict[str, Any]:
    """``named`` with each bounded property clamped strictly inside its bound."""
    clamped = dict(named)
    for name, value in named.items():
        low, high = getattr(model.model, "bounds", {}).get(name, (None, None))
        if low is None and high is None:
            continue
        width = high - low if low is not None and high is not None else None
        value = torch.as_tensor(value, dtype=torch.float32)
        if low is not None:
            value = value.clamp(min=low + _INSIDE * (width or max(abs(low), 1.0)))
        if high is not None:
            value = value.clamp(max=high - _INSIDE * (width or max(abs(high), 1.0)))
        clamped[name] = value
    return clamped


def _smoothing(model: nlop.ModelOperator, smooth) -> linop.LinearOperator | None:
    """The model's input from the variables solved for: the smooth properties
    through the Sobolev weighting, the rest as they are."""
    if not smooth:
        return None
    names = model.names
    missing = [name for name in smooth if name not in names]
    if missing:
        raise ValueError(f"smooth names {missing}, which the model does not solve for: {names}")
    voxels = tuple(model.voxels)
    axes = tuple(range(1, 1 + len(voxels)))
    parts: list[linop.LinearOperator] = []
    run = 0
    for name in names:
        if name not in smooth:
            run += 1
            continue
        if run:
            parts.append(linop.Identity((run, *voxels)))
            run = 0
        a, b = smooth[name]
        shape = (1, *voxels)
        parts.append(linop.Real(shape) @ linop.Sobolev(shape, axes, a, b))
    if run:
        parts.append(linop.Identity((run, *voxels)))
    return linop.block_diag(parts)


def _given(kspace, sensitivities, model, traj, pattern):
    """The linear encoding over known sensitivities, and the data in its layout."""
    voxels, contrasts = model.voxels, model.contrasts
    coils = sensitivities.shape[0]
    if traj is None:
        encoding = linop.CartesianSense(sensitivities, (contrasts, *voxels), ndim=len(voxels))
        if pattern is None:
            pattern = _estimated_pattern(kspace)
    else:
        encoding = linop.NoncartesianSense(sensitivities, (contrasts, *voxels), traj=traj)

    # A trajectory shared by the contrasts makes them a batch, which the
    # encoding puts in front of the coils; one with a contrast axis of its own
    # makes them a sample axis, which it puts behind them.
    samples = tuple(kspace.shape[2:])
    swapped = tuple(encoding.oshape) != (contrasts, coils, *samples)
    if swapped and tuple(encoding.oshape) != (coils, contrasts, *samples):
        raise ValueError(
            f"kspace is (contrasts, coils, *samples) = {tuple(kspace.shape)}, which the "
            f"encoding's samples {tuple(encoding.oshape)} are not"
        )

    def laid_out(t: torch.Tensor) -> torch.Tensor:
        return t.transpose(0, 1) if swapped else t

    data = kspace
    A = encoding
    if pattern is not None:
        pattern = torch.as_tensor(pattern).to(torch.complex64)
        pattern = pattern.reshape((1,) * (kspace.ndim - pattern.ndim) + tuple(pattern.shape))
        data = kspace * pattern
        A = linop.Diagonal(laid_out(pattern).contiguous(), encoding.oshape) @ encoding
    return A, laid_out(data).contiguous()


def moba(
    kspace: torch.Tensor,
    model: nlop.ModelOperator,
    sensitivities: torch.Tensor | None = None,
    *,
    traj: torch.Tensor | None = None,
    pattern: torch.Tensor | None = None,
    iterations: int = _ITERATIONS,
    cg_maxiter: int = _CG_MAXITER,
    inner: Any = None,
    alpha: float = 1.0,
    alpha_min: float = 0.0,
    redu: float = 2.0,
    sobolev: tuple[float, float] = _SOBOLEV,
    smooth: dict[str, tuple[float, float]] | None = None,
    start: torch.Tensor | None = None,
    reference: dict[str, Any] | None = None,
    scaling: float | None = None,
    return_sensitivities: bool = False,
    **values: Any,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Fit a signal model to k-space: model-based reconstruction of parameter maps.

    The method of BART's ``moba``: the model ``y = P F (S . M(theta))``,
    fitted to the samples by Gauss-Newton (:class:`bartorch.nlop.IRGNM`).
    Without ``sensitivities`` the coils are a second unknown, shared by the
    contrasts and represented as in :class:`~bartorch.nlop.NonlinearSense`,
    under a Sobolev weighting.  The model is BlochSim's
    (:class:`~bartorch.nlop.ModelOperator`), so the result is named maps in
    physical units rather than the command's coefficients.

    Parameters
    ----------
    kspace : torch.Tensor
        Samples ``(contrasts, coils, *samples)``, C order, contrasts in the
        order of the model's acquisition.  ``*samples`` is the voxel shape on
        a grid, and ``traj.shape[:-1]`` without a leading contrast axis along
        a trajectory.
    model : bartorch.nlop.ModelOperator
        Signal model on the voxel shape, ``(y, x)`` or ``(z, y, x)``.
    sensitivities : torch.Tensor, default=None
        Coil sensitivities ``(coils, *voxels)``, shared by the contrasts;
        estimated jointly with the maps when ``None``.
    traj : torch.Tensor, default=None
        Trajectory ``(..., samples, 3)`` in grid units, shared by the contrasts
        or with a leading contrast axis.  Requires ``sensitivities``.
    pattern : torch.Tensor, default=None
        Sampling pattern or weights broadcastable against ``kspace``, of size
        one along the coil axis; may differ between contrasts.  ``None`` on a
        grid is one wherever any coil has a sample.
    iterations : int, default=20
        Gauss-Newton steps.  The command's 8 suffice for its scaled
        coefficients, not for the bounded parameterization.
    cg_maxiter : int, default=50
        Conjugate-gradient iterations per linearized problem.
    inner : solver, default=None
        Solver from :mod:`bartorch.optim` for the linearized problem, whose
        regularizers penalize the maps; requires ``sensitivities``.  ``None``
        is conjugate gradients.
    alpha : float, default=1.0
        Tikhonov weight of the first Gauss-Newton step.
    alpha_min : float, default=0.0
        Limit the weight decays towards.
    redu : float, default=2.0
        Decay factor of the weight per step.
    sobolev : tuple of float, default=(880.0, 32.0)
        ``(a, b)`` of the coil weighting ``(1 + a |k|^2)^(-b/2)`` when the
        coils are estimated; the command's defaults.
    smooth : dict of str to tuple of float, default=None
        ``{name: (a, b)}``: properties solved for as smooth maps, through the
        weighting ``(1 + a |k|^2)^(-b/2)`` of :class:`~bartorch.linop.Sobolev`,
        as the command treats its B1 and B0 maps.  The weighting applies to
        the model's variable, the transformed value for a bounded property.
    start : torch.Tensor, default=None
        Starting point in the model's domain, with the amplitude divided by
        ``scaling``; built from ``**values`` when ``None``.  A ``smooth``
        property enters through the weighting's adjoint, which keeps a
        constant map and smooths any other.
    reference : dict of str, default=None
        Values each step is regularized towards, ``{name: value}`` in physical
        units with an amplitude in the units of ``kspace``; unnamed properties
        take the model's defaults, and values are clamped strictly inside the
        model's bounds.  ``None`` uses the start.  The coil coefficients are
        regularized towards zero.
    scaling : float, default=None
        Factor the data is divided by for the solve; the fitted amplitude is
        multiplied by it again.  ``None`` applies the rule of
        :func:`bartorch.optim.data_scaling` to least-squares contrast images
        (known coils) or zero-filled coil images (estimated coils).  A model
        without an amplitude is fitted to unscaled data.
    return_sensitivities : bool, default=False
        Also return the estimated sensitivities.
    **values
        Starting values ``name=value`` in physical units, as
        :meth:`~bartorch.nlop.ModelOperator.initial` takes them, with an
        amplitude in the units of ``kspace``.

    Returns
    -------
    maps : dict of str to torch.Tensor
        Fitted maps in physical units, keyed by unknown.  With estimated coils,
        amplitude and sensitivities share a complex factor the data does not
        determine.
    sensitivities : torch.Tensor
        Estimated sensitivities ``(coils, *voxels)``; only with
        ``return_sensitivities``.

    Raises
    ------
    ValueError
        If ``traj`` or ``inner`` is given without ``sensitivities``, if
        ``return_sensitivities`` is set with ``sensitivities``, if ``kspace``
        does not match the encoding's samples, or if ``smooth`` names a
        property the model does not solve for.

    Notes
    -----
    BART regularizes each step towards zero.  Zero in the bounded
    parameterization is the middle of each bound with no amplitude, and with
    the coils starting at zero the data has no derivative by the maps there,
    so the default centre is the start instead.  Its influence decays with
    the Tikhonov weight.

    Examples
    --------
    >>> M = nlop.MultiEcho([12.5 * (echo + 1) for echo in range(8)], (128, 128))
    >>> maps = moba(kspace, M, sensitivities, T2=80.0)
    >>> maps, sensitivities = moba(kspace, M, return_sensitivities=True, T2=80.0)
    """
    kspace = torch.as_tensor(kspace)
    S = _smoothing(model, smooth)
    solved = model if S is None else model @ S

    def lifted(x: torch.Tensor) -> torch.Tensor:
        return x if S is None else S.H(x)

    def fit(F, data, images, coefficients=None):
        scale = _scale(model, images) if scaling is None else float(scaling)

        def packed(named):
            amplitude = named.get("amplitude")
            return model.initial(
                **(named if amplitude is None else {**named, "amplitude": amplitude / scale})
            ).to(kspace.device)

        x0 = lifted(packed(values) if start is None else start)
        xref = x0 if reference is None else lifted(packed(_inside(model, reference)))
        solver = nlop.IRGNM(
            iterations=iterations,
            alpha=alpha,
            alpha_min=alpha_min,
            redu=redu,
            cg_maxiter=cg_maxiter,
            inner=optim.CG(maxiter=cg_maxiter) if inner is None else inner,
        )
        if coefficients is None:
            fitted = solver(data * (1.0 / scale), F, x0=x0, xref=xref)
        else:
            fitted, coefficients = solver(
                data * (1.0 / scale), F, x0=(x0, coefficients), xref=(xref, coefficients)
            )
        maps = model.split(fitted if S is None else S(fitted))
        if "amplitude" in maps:
            maps["amplitude"] = maps["amplitude"] * scale
        return maps, coefficients

    if sensitivities is not None:
        if return_sensitivities:
            raise ValueError("return_sensitivities returns estimated sensitivities, not given ones")
        A, data = _given(kspace, sensitivities, model, traj, pattern)
        return fit(A @ solved, data, lambda: optim.CG(maxiter=_SCALE_CG_MAXITER)(data, A))[0]

    if traj is not None:
        raise ValueError(
            "off the grid the coils are not estimated here; pass sensitivities, from "
            "bartorch.tools.nlinv or bartorch.tools.ecalib"
        )
    if inner is not None:
        raise ValueError(
            "an inner solver's regularizers act on the whole state, which with estimated "
            "coils is the maps and the coils laid end to end; pass sensitivities to "
            "regularize the maps"
        )

    voxels, contrasts = tuple(model.voxels), model.contrasts
    coils = kspace.shape[1]
    spatial = _spatial(voxels)
    if pattern is None:
        pattern = _estimated_pattern(kspace)
    pattern = torch.as_tensor(pattern).to(torch.complex64)
    pattern = torch.broadcast_to(pattern, (contrasts, 1, *voxels))
    data = (kspace * pattern).transpose(0, 1)

    sense = nlop.NonlinearSense(
        (coils, contrasts, *spatial),
        pattern=pattern.reshape(1, contrasts, *spatial).contiguous(),
        coil_shape=(coils, 1, *spatial),
        sobolev=sobolev,
    )
    data = data.reshape(sense.kspace_shape).contiguous()
    # A binary pattern on a grid makes the zero-filled coil images the
    # least-squares ones.
    maps, coefficients = fit(
        _joint(sense, solved),
        data,
        lambda: sense.transform.adjoint(data),
        torch.zeros(sense.ishapes[1], dtype=torch.complex64, device=kspace.device),
    )
    if not return_sensitivities:
        return maps
    return maps, sense.coils(coefficients).reshape(coils, *voxels)
