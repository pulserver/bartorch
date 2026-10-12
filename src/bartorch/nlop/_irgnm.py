"""Nonlinear inverse problems by iteratively regularized Gauss-Newton.

The method comes in two forms, and :class:`IRGNMBlock` takes one step of
either.  ``irgnm`` solves each linearized problem by conjugate gradients inside
the library, as ``noir``'s step does, which is the block without an inner
solver.  ``irgnm2`` hands the problem to a generic regularized least-squares
solver, which is what ``inner=`` selects.  :class:`IRGNM` loops the block to
BART's schedule, as the solvers in :mod:`bartorch.optim` loop theirs.
"""

from __future__ import annotations

import copy
import dataclasses
import weakref

import torch
from torch import nn

from bartorch._dispatch import BartError, _lock, _on_device
from bartorch._lib import library
from bartorch._operator import as_operand
from bartorch.optim._blocks import _setting, _value

__all__ = ["IRGNM", "IRGNMBlock", "irgnm"]


def _inner_solver(inner):
    """``inner`` checked to be a configured solver, which is all it may be."""
    from bartorch.optim import _linear as linear

    if isinstance(inner, str):
        raise TypeError(
            f"inner takes a configured solver, not the name of one: pass "
            f"optim.{inner.upper()}() rather than {inner!r}, so that its terms and "
            "settings are visible where the solve is written"
        )
    if not isinstance(inner, linear._Solver):
        raise TypeError(f"inner takes a solver from bartorch.optim, not {type(inner).__name__}")
    return inner


def _at(solver, alpha: float):
    """``solver`` as this Newton step needs it.

    Two things change per step.  The Tikhonov weight goes on the normal
    operator, which is where ``lsqr2_create(..., lambda = alpha, ...)`` puts
    it and where ``cclambda`` puts it.

    And a step-size solver power-iterates.  The normal operator of a Newton
    step is ``DF^H DF + alpha I``, whose largest eigenvalue changes with the
    linearization point *and* with alpha, so a fixed step diverges the moment
    it exceeds ``2 / L``.  BART's own inner FISTA does not offer the choice --
    ``moba/iter_l1.c``'s ``inverse_fista`` computes ``alpha + power(20,
    normal)`` and rescales at every step -- so neither does this.
    """
    step = copy.copy(solver)
    step.cclambda = float(alpha)
    if hasattr(step, "eigen"):
        step.eigen = True
    return step


class IRGNMBlock(nn.Module):
    """One iteratively regularized Gauss-Newton step, as a :class:`torch.nn.Module`.

    Without ``inner`` the step is BART's ``irgnm`` form (``noir``'s
    Gauss-Newton step),

    ``x = xn + (DF^H DF + alpha)^-1 [DF^H (y - F(xn)) - alpha (xn - xref)]``,

    with the inverse computed by conjugate gradients inside the library and
    differentiated implicitly.  With ``inner`` it is the ``irgnm2`` form: the
    solver minimizes ``||DF u - r||^2 + alpha ||u||^2 + R(u)`` with ``DF``
    taken at ``xn`` and ``r = y - F(xn) + DF (xn - xref)``, and
    ``x = u + xref``.

    ``state = block.start(y, F, x0, xref)`` prepares a run,
    ``state = block(state, F)`` takes one step, and ``block.output(state, F)``
    returns the unknowns.  Step ``k`` (counted from zero in the state) uses
    ``alpha`` decayed ``k`` times by ``alpha <- (alpha - alpha_min) / redu +
    alpha_min``, so a looped block, or a stack of blocks with equal
    ``alpha``, follows BART's schedule.

    Parameters
    ----------
    alpha : float, default=1.0
        Tikhonov weight of the first step.
    alpha_min : float, default=0.0
        Limit the weight decays towards.
    alpha_min0 : float, default=0.0
        Lower bound on the decayed weight; ``irgnm2`` form only.
    redu : float, default=2.0
        Decay factor of the weight per step.
    cg_maxiter : int, default=30
        Conjugate-gradient iterations of the ``irgnm`` form
        (``iter_conjgrad_conf.maxiter``).
    cg_tol : float, default=0.0
        Their tolerance (``iter_conjgrad_conf.tol``); BART refuses a nonzero
        value when the step is differentiated.
    cg_lambda : float, default=0.0
        Their Tikhonov weight (``iter_conjgrad_conf.l2lambda``).
    inner : solver, default=None
        Configured solver from :mod:`bartorch.optim` for the linearized
        problem, selecting the ``irgnm2`` form; its regularizers are ``R``.
    fuse : bool, default=True
        Apply the encoding of a coil model (a product of two unknowns before a
        linear encoding) as its normal operator; see :meth:`plan`.

    Raises
    ------
    ValueError
        If ``alpha_min0`` is nonzero without ``inner``.
    TypeError
        If ``inner`` is not a solver from :mod:`bartorch.optim`.

    Notes
    -----
    ``F`` must supply its derivative as a function of the point; an operator
    from :meth:`~bartorch.nlop.NonlinearOperator.from_callbacks` does not, and
    is refused with ``TypeError``.  A leading batch axis on ``y`` holds
    independent items, each stepped on its own; a model that holds its items,
    such as ``CoilSense(..., items=True)``, is applied to all at once, with
    each item's inner problem solved on its own.

    ``alpha`` and ``redu`` are :class:`torch.nn.Parameter` objects, frozen until
    ``requires_grad_()``.  The ``irgnm`` form is differentiable by the data,
    the iterate, ``xref`` and ``alpha``.  The ``irgnm2`` form is differentiable
    by the data, the iterate, ``xref`` and the inner solver's settings and
    priors, and passes ``alpha`` to the solver as a number.
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        """State of a run: iterate ``x``, centre ``xref``, prepared data, model and step ``k``."""

        x: torch.Tensor
        xref: torch.Tensor
        data: torch.Tensor
        space: object
        k: int = 0

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        alpha_min: float = 0.0,
        alpha_min0: float = 0.0,
        redu: float = 2.0,
        cg_maxiter: int = 30,
        cg_tol: float = 0.0,
        cg_lambda: float = 0.0,
        inner=None,
        fuse: bool = True,
    ):
        super().__init__()
        self.inner = None if inner is None else _inner_solver(inner)
        if self.inner is None and alpha_min0:
            raise ValueError(
                "alpha_min0 is the second form's floor; BART's first form has none, so it "
                "takes an inner solver"
            )
        self.alpha = _setting(alpha)
        self.redu = _setting(redu)
        self.alpha_min = float(alpha_min)
        self.alpha_min0 = float(alpha_min0)
        self.cg_maxiter = int(cg_maxiter)
        self.cg_tol = float(cg_tol)
        self.cg_lambda = float(cg_lambda)
        self.fuse = bool(fuse)
        self._spaces = weakref.WeakKeyDictionary()

    def _space(self, F):
        """``F`` prepared for steps, built once per model."""
        from bartorch.linop._base import LinearOperator
        from bartorch.nlop._step import Linearized

        made = self._spaces.get(F)
        if made is None:
            model = F.to_nonlinear() if isinstance(F, LinearOperator) else F
            made = Linearized(
                model,
                cg_maxiter=self.cg_maxiter,
                cg_tol=self.cg_tol,
                cg_lambda=self.cg_lambda,
                fuse=self.fuse,
                inverse=self.inner is None,
            )
            self._spaces[F] = made
        return made

    def plan(self, F):
        """Report how the steps apply ``F``.

        Parameters
        ----------
        F : NonlinearOperator or LinearOperator
            Forward model.

        Returns
        -------
        Plan
            ``plan.derivative``: where the derivative comes from (``'declared'``,
            ``'chain rule'``, ``'linear'`` or ``'torch'``); ``plan.domain``:
            ``'normal'`` when the step applies the encoding as ``E^H E`` to
            data taken through ``E^H``, ``'paired'`` when it applies ``E`` and
            ``E^H``; ``plan.encoding``: the linear encoding's own plan, or
            ``None``; ``plan.fused``: whether the coil model was rewritten.
        """
        return self._space(F).plan

    def start(self, y: torch.Tensor, F, x0=None, xref=None) -> State:
        """Prepare a run.

        Parameters
        ----------
        y : torch.Tensor
            Data of the model's output shape, with an optional leading batch
            axis.
        F : NonlinearOperator or LinearOperator
            Forward model.
        x0, xref : torch.Tensor or tuple of torch.Tensor, default=None
            Start and regularization centre: a tuple with one tensor per
            input, the inputs flattened and concatenated, or for a model of
            one input that input at its shape.  ``x0=None`` starts as
            ``nlinv`` does, with the first input ones and the others zero;
            ``xref=None`` regularizes towards zero.

        Returns
        -------
        IRGNMBlock.State
            Iterate, centre, prepared data and step count ``k = 0``.

        Raises
        ------
        ValueError
            If ``inner`` is set and the model holds several items.
        """
        space = self._space(F)
        if self.inner is not None and 1 < space.items:
            raise ValueError(
                "a model holding several items is stepped by the first form; a solver from "
                "bartorch.optim takes a batch one run per item, so give the batch as a leading "
                "axis on the data of a single-item model instead"
            )
        data = space.prepare(torch.as_tensor(y))
        batch = data.shape[:1] if space.batched(data, space.data_shape) else ()
        if x0 is None:
            x = space.start(batch, device=data.device)
        else:
            x = space.state(x0).expand(*batch, *space.state_shape)
        centre = torch.zeros_like(x) if xref is None else space.state(xref).expand_as(x)
        return self.State(x, centre, data, space)

    def forward(self, state: State, F) -> State:
        """Take one Gauss-Newton step.

        Parameters
        ----------
        state : IRGNMBlock.State
            State from :meth:`start` or a previous step.
        F : NonlinearOperator or LinearOperator
            Forward model the state was started with.

        Returns
        -------
        IRGNMBlock.State
            The new iterate, with ``k`` incremented.
        """
        space = state.space
        take = self._first if self.inner is None else self._second
        if space.batched(state.x, space.state_shape):
            items = zip(state.x, state.xref, state.data)
            x = torch.stack([take(space, *item, state.k) for item in items])
        else:
            x = take(space, state.x, state.xref, state.data, state.k)
        return dataclasses.replace(state, x=x, k=state.k + 1)

    def output(self, state: State, F):
        """Return the unknowns of a state.

        Parameters
        ----------
        state : IRGNMBlock.State
            State of the run.
        F : NonlinearOperator or LinearOperator
            Forward model.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            One tensor per input of ``F``, with the batch axis of the data.
        """
        parts = state.space.split(state.x)
        return parts[0] if 1 == len(parts) else parts

    def _first(self, space, x, xref, data, k: int):
        """``noir_gauss_newton_step_create_s``'s expression, one item."""
        from bartorch.nlop._derivative import _evaluate

        alpha = self._weight(k, x)
        residual = data - _evaluate(space.operator, x)
        rhs = _evaluate(space.adjoint, residual, x) - alpha * (x - xref)
        return x + _evaluate(space.inverse(x.device), rhs, x, alpha)

    def _second(self, space, x, xref, data, k: int):
        """``irgnm2``'s step, one item: the linearization carried to the centre."""
        from bartorch.nlop._derivative import _evaluate

        residual = data - _evaluate(space.operator, x)
        derivative = space.flat.at(x)
        residual = residual + derivative.forward(x - xref)
        return _at(self.inner, self._number(k))(residual, derivative) + xref

    def _weight(self, k: int, like: torch.Tensor) -> torch.Tensor:
        """The first form's weight at step ``k``: a vector as long as the state.

        Decayed one step at a time, in the arithmetic BART's schedule uses.
        """
        if self.alpha.requires_grad:
            weight = self.alpha.float().to(torch.complex64) * torch.ones_like(like)
        else:
            weight = torch.full_like(like, float(self.alpha))
        redu = _value(self.redu)
        for _ in range(k):
            weight = (weight - self.alpha_min) / redu + self.alpha_min
        return weight

    def _number(self, k: int) -> float:
        """The second form's weight at step ``k``, which ``irgnm2`` keeps in a double."""
        alpha, redu = float(self.alpha), float(self.redu)
        for _ in range(k):
            alpha = (alpha - self.alpha_min) / redu + self.alpha_min
            if alpha < self.alpha_min0:
                alpha = self.alpha_min0
        return alpha

    def __repr__(self) -> str:
        inner = "" if self.inner is None else f", inner={type(self.inner).__name__}(...)"
        return f"IRGNMBlock(alpha={float(self.alpha)}, redu={float(self.redu)}{inner})"


class IRGNM:
    """Iteratively regularized Gauss-Newton solver for ``F(x) = y``.

    Each step linearizes ``F`` at the current iterate and solves

    ``min_u ||DF u - r||^2 + alpha ||u||^2 + R(u)``

    for the residual ``r`` carried to the linearization; ``alpha`` is then
    decayed towards ``alpha_min`` by ``redu``.  ``R`` is the regularization of
    the inner solver, and absent for conjugate gradients.  The solver loops
    :class:`IRGNMBlock`, which documents the two forms.

    Parameters
    ----------
    iterations : int, default=8
        Gauss-Newton steps.
    alpha : float, default=1.0
        Tikhonov weight of the first step.
    alpha_min : float, default=0.0
        Limit the weight decays towards.
    alpha_min0 : float, default=0.0
        Lower bound on the decayed weight; used only with ``inner``.
    redu : float, default=2.0
        Decay factor of the weight per step.
    cg_maxiter : int, default=30
        Conjugate-gradient iterations per step without ``inner``.
    cg_tol : float, default=0.0
        Conjugate-gradient tolerance per step without ``inner``.
    inner : solver, default=None
        Configured solver from :mod:`bartorch.optim` for the linearized
        problem, whose regularizers are ``R``.  ``None`` solves it by
        conjugate gradients inside the library, as ``nlinv`` does.
    fuse : bool, default=True
        Apply the encoding of a coil model as its normal operator; see
        :meth:`IRGNMBlock.plan`.

    Notes
    -----
    The two forms agree in exact arithmetic but not to the bit: without
    ``inner`` the result matches BART's ``iter4_irgnm``, and with
    ``inner=optim.CG()`` it matches ``iter4_irgnm2``.

    Examples
    --------
    Plain, which is ``nlinv``:

    >>> nlop.IRGNM(iterations=8)(kspace, F, x0=start)

    The same method with the linearized problem written out here, which is
    ``iter4_irgnm2`` to the bit:

    >>> nlop.IRGNM(iterations=8, inner=optim.CG())(kspace, F, x0=start)

    Wavelet-regularized:

    >>> nlop.IRGNM(inner=optim.FISTA(priors.Wavelet((-1, -2), 0.001), maxiter=30))(
    ...     kspace, F, x0=start
    ... )

    Several terms at once, which is ADMM's job:

    >>> nlop.IRGNM(
    ...     inner=optim.ADMM(
    ...         [priors.Wavelet((-1, -2), 0.001), priors.TotalVariation((-1, -2), 0.01)]
    ...     )
    ... )(kspace, F, x0=start)
    """

    def __init__(
        self,
        *,
        iterations: int = 8,
        alpha: float = 1.0,
        alpha_min: float = 0.0,
        alpha_min0: float = 0.0,
        redu: float = 2.0,
        cg_maxiter: int = 30,
        cg_tol: float = 0.0,
        inner=None,
        fuse: bool = True,
    ):
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_min0 = float(alpha_min0)
        self.redu = float(redu)
        self.cg_maxiter = int(cg_maxiter)
        self.cg_tol = float(cg_tol)
        self.inner = None if inner is None else _inner_solver(inner)
        self.fuse = bool(fuse)

    def _block(self) -> IRGNMBlock:
        """The step this solver loops over, built from its settings."""
        return IRGNMBlock(
            alpha=self.alpha,
            alpha_min=self.alpha_min,
            alpha_min0=self.alpha_min0 if self.inner is not None else 0.0,
            redu=self.redu,
            cg_maxiter=self.cg_maxiter,
            cg_tol=self.cg_tol,
            inner=self.inner,
            fuse=self.fuse,
        )

    def __call__(self, y: torch.Tensor, F, x0=None, xref=None):
        """Solve ``F(x) = y``.

        Parameters
        ----------
        y : torch.Tensor
            Data of ``F.oshape``, with an optional leading batch axis.
        F : NonlinearOperator or LinearOperator
            Forward model; a linear one is converted with
            :meth:`~bartorch.linop.LinearOperator.to_nonlinear`.
        x0 : torch.Tensor or tuple of torch.Tensor, default=None
            Start; see :meth:`IRGNMBlock.start`.
        xref : torch.Tensor or tuple of torch.Tensor, default=None
            Regularization centre; ``None`` regularizes towards zero, as BART
            does.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Complex64 solution, one tensor per input of ``F``.
        """
        block = self._block()
        state = block.start(y, F, x0, xref)
        for _ in range(self.iterations):
            state = block(state, F)
        return block.output(state, F)

    # --- BART's own loops, the references the block is held against ---------

    def _first_in_library(
        self, y: torch.Tensor, F, x0: torch.Tensor, xref: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``iter4_irgnm``, BART's first form, run inside the library: the reference
        the block without an inner solver is held against."""
        from bartorch.linop._base import LinearOperator

        if isinstance(F, LinearOperator):
            F = F.to_nonlinear()
        op = F._bart()
        y = as_operand(y, op.oshape, "y")
        x = as_operand(x0, op.ishape, "x0").clone()
        ref = as_operand(xref, op.ishape, "xref") if xref is not None else None
        with _lock, _on_device(op.device or y.device):
            code = library().bartorch_irgnm(
                op._h.ptr,
                self.iterations,
                self.alpha,
                self.alpha_min,
                self.redu,
                self.cg_maxiter,
                self.cg_tol,
                x.data_ptr(),
                y.data_ptr(),
                ref.data_ptr() if ref is not None else None,
            )
        if code != 0:
            raise BartError("Gauss-Newton solve failed; see the log for BART's message")
        return x

    def _in_library(
        self, y: torch.Tensor, F, x0: torch.Tensor, xref: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``iter4_irgnm2`` with BART's own conjugate gradients: the reference
        ``inner=optim.CG()`` is held against."""
        from bartorch.linop._base import LinearOperator

        if isinstance(F, LinearOperator):
            F = F.to_nonlinear()
        op = F._bart()
        y = as_operand(y, op.oshape, "y")
        x = as_operand(x0, op.ishape, "x0").clone()
        ref = as_operand(xref, op.ishape, "xref") if xref is not None else None
        with _lock, _on_device(op.device or y.device):
            code = library().bartorch_irgnm2(
                op._h.ptr,
                self.iterations,
                self.alpha,
                self.alpha_min,
                self.alpha_min0,
                self.redu,
                self.cg_maxiter,
                self.cg_tol,
                x.data_ptr(),
                y.data_ptr(),
                ref.data_ptr() if ref is not None else None,
            )
        if code != 0:
            raise BartError("Gauss-Newton solve failed; see the log for BART's message")
        return x

    def __repr__(self) -> str:
        inner = "" if self.inner is None else f", inner={type(self.inner).__name__}(...)"
        return (
            f"IRGNM(iterations={self.iterations}, alpha={self.alpha}, "
            f"alpha_min={self.alpha_min}, redu={self.redu}{inner})"
        )


def irgnm(y: torch.Tensor, F, *, x0=None, xref=None, inner=None, **settings):
    """Solve ``F(x) = y`` by iteratively regularized Gauss-Newton.

    Functional form of :class:`IRGNM`.

    Parameters
    ----------
    y : torch.Tensor
        Data of ``F.oshape``, with an optional leading batch axis.
    F : NonlinearOperator or LinearOperator
        Forward model.
    x0 : torch.Tensor or tuple of torch.Tensor, default=None
        Start; ``None`` starts with the first input ones and the others zero,
        as ``nlinv`` does.  See :meth:`IRGNMBlock.start`.
    xref : torch.Tensor or tuple of torch.Tensor, default=None
        Regularization centre; ``None`` regularizes towards zero.
    inner : solver, default=None
        Solver from :mod:`bartorch.optim` for the linearized problem; ``None``
        uses conjugate gradients inside the library.
    **settings
        Other parameters of :class:`IRGNM`: ``iterations``, ``alpha``,
        ``alpha_min``, ``alpha_min0``, ``redu``, ``cg_maxiter``, ``cg_tol``,
        ``fuse``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Complex64 solution, one tensor per input of ``F``.
    """
    return IRGNM(inner=inner, **settings)(y, F, x0=x0, xref=xref)
