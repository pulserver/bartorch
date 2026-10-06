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
#: coefficients; BlochSim's bounded parameterisation needs more, as
#: :func:`~bartorch.apps.mobafit` sets out for the same reason.
_ITERATIONS = 20

#: Conjugate-gradient steps per linearized problem, as :func:`~bartorch.apps.mobafit`
#: takes them.
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


def _scale(model: nlop.SignalModel, images) -> float:
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


def _inside(model: nlop.SignalModel, named: dict[str, Any]) -> dict[str, Any]:
    """``named`` with each bounded property clamped strictly inside its bound."""
    clamped = dict(named)
    for name, value in named.items():
        low, high = model.model.bounds.get(name, (None, None))
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


def _smoothing(model: nlop.SignalModel, smooth) -> linop.LinearOperator | None:
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
    model: nlop.SignalModel,
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
    """Fit a signal model to k-space: parameter maps straight from the samples.

    The method of the ``moba`` command, assembled here: the signal model
    inside the encoding, ``y = P F (S . M(theta))``, fitted to the k-space by
    the Gauss-Newton loop of :class:`bartorch.nlop.IRGNM`, and the fitted
    variables read back into their own units.  Without ``sensitivities`` the
    coils are a second unknown, estimated jointly with the maps as the command
    estimates them: a k-space representation under the Sobolev weighting of
    :class:`~bartorch.nlop.NonlinearSense`, shared by every contrast.

    The model is BlochSim's rather than BART's -- a
    :class:`~bartorch.nlop.SignalModel` -- so this does not reproduce the
    command's coefficients.  It solves the same problem with the same method
    and answers in named maps.

    Parameters
    ----------
    kspace : torch.Tensor
        Samples, ``(contrasts, coils, *samples)``, C order: ``*samples`` is the
        voxel shape on a grid, and the trajectory's sample axes off it --
        ``traj.shape[:-1]``, without a leading contrast axis the trajectory
        may carry.  Contrasts are in the order the model's acquisition lists
        them.
    model : bartorch.nlop.SignalModel
        The signal model, on the voxel shape of the image -- ``(y, x)`` or
        ``(z, y, x)``.
    sensitivities : torch.Tensor, default=None
        Coil sensitivities, ``(coils, *voxels)``, shared by the contrasts.
        Estimated jointly with the maps when not given.
    traj : torch.Tensor, default=None
        Non-Cartesian trajectory ``(..., samples, 3)`` in grid units, shared by
        the contrasts or with a leading contrast axis.  Requires
        ``sensitivities``.
    pattern : torch.Tensor, default=None
        Sampling pattern or weights, broadcastable against ``kspace``, with one
        entry along the coil axis; it may differ between contrasts.  On a grid
        it is read off ``kspace`` when not given: one wherever a coil has a
        sample.
    iterations : int, default=20
        Gauss-Newton steps.  The command takes eight over its own scaled
        coefficients; twenty are what a bounded parameterisation needs.
    cg_maxiter : int, default=50
        Conjugate-gradient steps per linearized problem.
    inner : solver, default=None
        A configured solver from :mod:`bartorch.optim` for the linearized
        problem, whose regularizers then penalize the maps; only with
        ``sensitivities``, where the maps are the only unknown.  The default
        is plain conjugate gradients.
    alpha : float, default=1.0
        Initial Tikhonov weight on the Gauss-Newton step.
    alpha_min : float, default=0.0
        What that weight decays towards.
    redu : float, default=2.0
        Factor the weight is divided by after each step.
    sobolev : tuple of float, default=(880.0, 32.0)
        ``(a, b)`` of the coil weighting ``(1 + a |k|^2)^(-b/2)`` when the
        coils are estimated; the command's defaults.
    smooth : dict of str to tuple of float, default=None
        Properties fitted as smooth maps, ``{name: (a, b)}``: each is solved
        for as k-space coefficients under the weighting ``(1 + a |k|^2)^(-b/2)``
        of :class:`~bartorch.linop.Sobolev`, as the command solves for its B1
        and B0 maps.  The weighting acts on the model's variable for the
        property, which for a bounded one is its transformed value.  A start or
        reference map for a smooth property is taken through the weighting's
        adjoint and back, which leaves a constant as it is and smooths
        anything else.
    start : torch.Tensor, default=None
        Maps to start from, of the model's input shape, in the units the solve
        works in -- with the amplitude divided by ``scaling``.  Built from
        ``**values`` when it is not given.
    reference : dict of str, default=None
        Maps each Gauss-Newton step is regularized towards, ``{name: value}``
        in each property's own units as ``**values`` takes them, with an
        amplitude in the data's units; a name left out takes the model's
        default.  The start when not given.  A value outside the model's bounds
        is clamped to them, so a limit such as a vanishing rate is approached
        from inside.
    scaling : float, default=None
        The factor the data is divided by for the solve; the fitted amplitude
        is multiplied by it again.  Estimated when not given, by the rule of
        :func:`bartorch.optim.data_scaling` applied to least-squares contrast
        images with known coils and zero-filled coil images without, which makes
        ``alpha`` and the start independent of the data's overall scale.  A
        model without an amplitude is fitted to the data as it stands.
    return_sensitivities : bool, default=False
        Also return the estimated sensitivities; only when they are estimated.
    **values
        Starting values per unknown, in that property's own units, as
        :meth:`~bartorch.nlop.SignalModel.initial` takes them; an amplitude
        in the data's units.

    Returns
    -------
    dict of str to torch.Tensor
        The fitted maps in their own units, by the name each unknown carries.
        With estimated coils the amplitude and the sensitivities share one
        complex scale between them, which the data does not fix.
    torch.Tensor
        With ``return_sensitivities``: the sensitivities, ``(coils, *voxels)``.

    Raises
    ------
    ValueError
        ``inner``, ``traj`` or ``return_sensitivities`` without
        ``sensitivities``, a ``kspace`` the encoding's samples are not, or a
        ``smooth`` property the model does not solve for.

    Notes
    -----
    Each step is regularized towards ``reference`` -- by default the maps
    towards where they started, the coil coefficients towards zero -- where the command
    regularizes towards zero.  Zero in BlochSim's parameterisation is the
    middle of each bound and no amplitude, not a plausible map, and the coils start
    at zero, as the command starts them: at that point the data has no
    derivative by the maps, so a first step centred on zero would take the
    maps to the middle of their bounds and leave the data with no derivative
    by the coils.  The weight decays by ``redu`` each step, and the start's
    influence with it.

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
