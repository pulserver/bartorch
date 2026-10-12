"""``pics`` as a pipeline: the encoding, the terms and the iteration."""

from __future__ import annotations

from collections.abc import Iterable

import torch

import bartorch
from bartorch import linop, optim
from bartorch.linop import _basic as basic
from bartorch.optim._linear import NIHT
from bartorch.priors import Regularizer
from bartorch.tools import _sampling as sampling

__all__ = ["image_shape", "pics"]

#: BART's own ``-i`` default (pics.c:82).
_MAXITER = 30

#: Which iteration ``italgo_choose`` (grecon/italgo.c) picks from the terms,
#: by the letter each term carries as its ``kind``.  A term not named here makes the
#: iteration FISTA when it is the first and ADMM when it is not.
_ALWAYS_ADMM = frozenset({"T", "G", "C", "V", "R1", "R2"})
_ALWAYS_NIHT = frozenset({"H", "N"})
#: An l2 penalty on the image leaves the choice where it was.
_KEEPS = frozenset({"Q"})

_SOLVERS = {
    "cg": optim.CG,
    "ist": optim.IST,
    "fista": optim.FISTA,
    "admm": optim.ADMM,
    "pridu": optim.PRIDU,
    "niht": NIHT,
}


def _terms(regularizers: Regularizer | Iterable[Regularizer] | None) -> list[Regularizer]:
    if regularizers is None:
        return []
    if isinstance(regularizers, Regularizer):
        return [regularizers]
    return list(regularizers)


def _chosen(terms: list[Regularizer]) -> str:
    """The iteration ``pics`` selects for these terms, when none was named."""
    algorithm = "cg"
    for position, term in enumerate(terms):
        letter = term.kind
        if letter in _KEEPS:
            continue
        if letter in _ALWAYS_NIHT:
            algorithm = "niht"
        elif letter in _ALWAYS_ADMM:
            algorithm = "admm"
        else:
            algorithm = "fista" if position == 0 else "admm"
    return algorithm


def image_shape(
    kspace: torch.Tensor, sensitivities: torch.Tensor, traj: torch.Tensor | None = None
) -> tuple[int, ...]:
    """Image shape :func:`pics` reconstructs for these arguments.

    On a grid, the shape of ``kspace`` without the coil axis (and without a
    sets axis of one); along a trajectory, the shape of ``sensitivities``
    without the coil axis.
    """
    if traj is not None:
        maps = sensitivities.squeeze(1) if sensitivities.ndim > 3 else sensitivities
        return tuple(maps.shape[1:])
    if kspace.ndim > 3 and kspace.shape[1] == 1:
        return tuple(kspace.shape[2:])
    return tuple(kspace.shape[1:])


def pics(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    *,
    regularizers: Regularizer | Iterable[Regularizer] | None = None,
    l2: float | None = None,
    solver: str | None = None,
    maxiter: int | None = None,
    step: float | None = None,
    admm_rho: float | None = None,
    cg_maxiter: int | None = None,
    traj: torch.Tensor | None = None,
    pattern: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    initial: torch.Tensor | None = None,
    toeplitz: bool | None = None,
    eigen_step: bool = False,
    scaling: float | None = None,
    rescale: bool = True,
) -> torch.Tensor:
    """Parallel-imaging compressed-sensing reconstruction, BART's ``pics -S``.

    The steps of the command: the sampling pattern applied to the k-space, on
    a grid the modulation into BART's uncentred convention, division by the
    data scaling, the solve with an encoding from :mod:`bartorch.linop` and an
    iteration from :mod:`bartorch.optim`, and multiplication of the image by
    the scaling.  Regularization weights are therefore relative to the scaled
    data, and the image is in the units of ``kspace``.  On a grid the result
    equals the command's to the bit.

    Parameters
    ----------
    kspace : torch.Tensor
        k-space, C order: ``(coils, z, y, x)`` on a grid,
        ``(coils, *encoding, shots, samples)`` along a trajectory.
    sensitivities : torch.Tensor
        Coil sensitivities ``(coils, [1,] z, y, x)``, as
        :func:`~bartorch.tools.ecalib` returns them.
    regularizers : Regularizer or iterable of Regularizer, default=None
        Terms from :mod:`bartorch.priors`; their axes index the image.
    l2 : float, default=None
        Tikhonov weight (``-r``).
    solver : {'cg', 'ist', 'fista', 'admm', 'pridu', 'niht'}, default=None
        Iteration.  ``None`` chooses as the command does: ADMM when
        ``admm_rho`` is given, otherwise from the terms.
    maxiter : int, default=None
        Iterations (``-i``); ``None`` is BART's 30.
    step : float, default=None
        Step size of the gradient iterations (``-s``).
    admm_rho : float, default=None
        ADMM penalty (``-u``).
    cg_maxiter : int, default=None
        Conjugate-gradient iterations inside each ADMM step (``-C``).
    traj : torch.Tensor, default=None
        Trajectory ``(..., samples, 3)`` in grid units; selects the
        non-Cartesian encoding.
    pattern : torch.Tensor, default=None
        Sampling pattern (``-p``).  On a grid, ``None`` takes the support of
        ``kspace``.  Along a trajectory it weights the data term,
        ``||W (A x - y)||^2``, and is typically the square root of
        :func:`~bartorch.estimate_density`.
    basis : torch.Tensor, default=None
        Subspace basis (``-B``); used along a trajectory only.
    initial : torch.Tensor, default=None
        Starting image (``-W``), in the units of the result: of ``kspace``
        with ``rescale``, of the scaled data without.
    toeplitz : bool, default=None
        Along a trajectory, ``False`` applies the NUFFT and its adjoint instead
        of the normal operator's Toeplitz convolution (``--no-toeplitz``);
        ``None`` keeps the convolution.  Ignored on a grid.
    eigen_step : bool, default=False
        Step size from the largest eigenvalue of the normal operator, by 30
        power iterations (``-e``).  The start vector comes from BART's
        process-global generator, so repeated runs in one process differ.
    scaling : float, default=None
        Data scaling (``-w``); ``None`` or zero estimates it with
        :func:`bartorch.optim.data_scaling`, and an estimate of zero is
        replaced by one.
    rescale : bool, default=True
        Multiply the image by the scaling (``-S``).  ``False`` returns the
        image of the scaled data, as ``pics`` without ``-S``.

    Returns
    -------
    torch.Tensor
        Image of the shape :func:`image_shape` gives.

    Raises
    ------
    ValueError
        If ``solver`` is unknown; if ``'cg'`` gets more than one L2 term, or
        both ``l2`` and an L2 term; if ``eigen_step`` is set with ``'cg'``; if
        ``'ist'`` or ``'fista'`` gets a first term over a transform other than
        the identity, such as :class:`~bartorch.priors.FourierL1`, which the
        command would threshold on the image itself (``'admm'`` and
        ``'pridu'`` take it).

    Notes
    -----
    Without ``solver``, the command's rule (``italgo_choose``) applies: no
    term or an L2 term alone gives CG; :class:`~bartorch.priors.WaveletNIHT`
    and :class:`~bartorch.priors.ImageNIHT` give normalized iterative hard
    thresholding; total variation, the other terms ``italgo_choose`` assigns
    to ADMM, and any second term give ADMM; any other first term gives FISTA.

    Examples
    --------
    >>> image = pics(kspace, maps, l2=0.01, maxiter=50)
    >>> image = pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))
    """
    terms = _terms(regularizers)
    if solver is None:
        solver = "admm" if admm_rho is not None else _chosen(terms)
    if solver not in _SOLVERS:
        raise ValueError(f"solver must be one of {sorted(_SOLVERS)}, got {solver!r}")

    # Conjugate gradients has no proximal step, so an l2 penalty on the image
    # is its Tikhonov weight rather than a term of its own.  `opt_reg_configure`
    # (grecon/optreg.c:386) makes the two the same thing from the other side:
    # `-r` without a `-R` becomes exactly the `L2IMG` that `-R Q:` names.
    tikhonov = [term for term in terms if term.kind == "Q"]
    if solver == "cg" and tikhonov:
        if len(tikhonov) > 1:
            raise ValueError("cg takes one l2 penalty, not several")
        if l2 is not None:
            raise ValueError("the l2 weight is given once: as l2, or as an L2 term")
        l2 = tikhonov[0].weight
        terms = [term for term in terms if term.kind != "Q"]

    maps = sensitivities.squeeze(1) if sensitivities.ndim > 3 else sensitivities

    shape = image_shape(kspace, sensitivities, traj)

    if traj is None:
        if pattern is None:
            pattern = sampling.pattern(kspace)
        measured = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
        encoding = linop.CartesianSense(maps, shape, coil_batch=0, modulated=True)
        A = basic.Sampling(pattern.squeeze(), encoding.oshape) @ encoding
        scale = scaling or optim.data_scaling(measured) or 1.0
        data = (measured * (1.0 / scale)).squeeze(1)
    else:
        # `toeplitz` defaults to the normal operator's convolution, which is
        # what the application uses; None means nobody asked.
        # Off the grid the weights are a diagonal of the encoding, as `pics -p`
        # hands them to `nufft_create2`, so the data term is |W (A x - y)|^2.
        off_grid: dict[str, object] = {"traj": traj, "basis": basis, "weights": pattern}
        if toeplitz is not None:
            off_grid["toeplitz"] = toeplitz
        A = linop.NoncartesianSense(maps, shape, **off_grid)
        measured = kspace if pattern is None else kspace * pattern
        # Off the grid the scaling comes from the spread of the adjoint
        # reconstruction, so it is estimated over the samples in the layout
        # the application hands them in: a trailing readout axis of one.
        scale = scaling or optim.data_scaling(measured[..., None], A=A) or 1.0
        data = measured * (1.0 / scale)

    extra: dict[str, object] = {}
    if step is not None:
        extra["step"] = step
    if admm_rho is not None:
        extra["rho"] = admm_rho
    if cg_maxiter is not None:
        extra["cg_maxiter"] = cg_maxiter
    if solver == "pridu":
        extra["sigma_tau_ratio"] = scale
    if eigen_step:
        if solver == "cg":
            raise ValueError("eigen_step scales a gradient step, which cg does not take")
        extra["eigen"] = True

    if solver in ("ist", "fista") and terms and not terms[0].transform_is_identity(A.ishape):
        # `pics` hands these two iterations no transforms (`trafos_cond` in
        # pics.c), so the application thresholds the image itself for a term
        # over a transform; the iteration refuses instead.
        raise ValueError(
            f"{solver} applies the proximal operator of {terms[0]!r} to the image, without "
            "the transform the term penalizes; solver='admm' or solver='pridu' applies it"
        )
    iteration = _SOLVERS[solver]
    arguments = [] if solver == "cg" else [terms]
    if solver == "cg" and l2 is not None:
        arguments = [l2]
    iterate = iteration(*arguments, maxiter=_MAXITER if maxiter is None else maxiter, **extra)
    if not rescale:
        return iterate(data, A, initial)
    if initial is not None:
        initial = initial * (1.0 / scale)
    return iterate(data, A, initial) * scale
