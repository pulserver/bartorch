"""Least squares by BART's iterative algorithms.

The proximal solvers loop a block from :mod:`bartorch.optim._blocks` to BART's
schedule; conjugate gradients go to BART's ``lsqr2``, configured through
``italgo_config``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import library
from bartorch._operator import as_operand
from bartorch.optim._blocks import (
    ADMMBlock,
    FISTABlock,
    ISTBlock,
    PRIDUBlock,
    _empty,
    _preconditioner,
    _refuse_preconditioner,
    _refuse_transform,
)
from bartorch.priors._base import Regularizer, _as_terms
from bartorch.priors._terms import L2

__all__ = ["ADMM", "CG", "PRIDU", "FISTA", "IST", "NIHT", "Tikhonov"]


@dataclasses.dataclass(frozen=True)
class Tikhonov:
    r"""Quadratic penalty :math:`w \| G x - b \|^2` for :class:`CG`.

    Generalized Tikhonov regularization.  :class:`CG` stacks the terms under the
    encoding, so the problem stays linear least squares, the normal operator is
    :math:`A^H A + \sum_i w_i G_i^H G_i`, and a Toeplitz encoding keeps its
    point-spread-function convolution.

    Parameters
    ----------
    weight : float
        :math:`w`, the weight of the squared norm (not its square root); at
        least zero.
    operator : LinearOperator, default=None
        :math:`G`, with domain ``A.ishape``; the identity when omitted.
    bias : tensor, default=None
        :math:`b`, of ``operator.oshape``; zero when omitted.

    Raises
    ------
    ValueError
        ``weight`` is negative.

    Examples
    --------
    Pull towards a prior image rather than towards zero:

    >>> CG(terms=Tikhonov(0.1, bias=prior))(y, A)

    Penalize the first differences (quadratic total variation):

    >>> CG(terms=Tikhonov(0.1, operator=linop.Gradient(A.ishape, (-1, -2))))(y, A)
    """

    weight: float
    operator: object | None = None
    bias: torch.Tensor | None = None

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(
                f"a quadratic penalty has a weight of at least zero, not {self.weight}"
            )

    def _operator(self, ishape: tuple[int, ...]):
        """This term's operator, or the identity on the image."""
        if self.operator is not None:
            return self.operator

        from bartorch.linop import Identity

        return Identity(ishape)


Terms = Tikhonov | Iterable["Tikhonov"] | None


def _as_quadratics(terms: Terms) -> list[Tikhonov]:
    """``terms`` -- None, one term or an iterable of them -- as a list of terms."""
    if terms is None:
        return []
    if isinstance(terms, Tikhonov):
        return [terms]

    def refuse(what):
        return TypeError(
            f"CG takes Tikhonov terms, not {what!r}; a term with a proximal operator "
            "rather than a quadratic one goes to a solver that has one, such as FISTA"
        )

    try:
        out = list(terms)
    except TypeError:
        raise refuse(terms) from None
    for term in out:
        if not isinstance(term, Tikhonov):
            raise refuse(term)
    return out


def _stacked(A, y: torch.Tensor, terms: Sequence[Tikhonov]):
    """``(A~, y~)`` for ``min ||A x - y||^2 + sum_i w_i ||G_i x - b_i||^2``.

    Written as one least-squares problem, which conjugate gradients solves::

        A~ = [A; sqrt(w_1) G_1; ...]        y~ = [y; sqrt(w_1) b_1; ...]

    The codomains have nothing in common -- samples against images against
    differences -- so each is read as the line of numbers it is and they are
    laid end to end, as ``linop_stack_cod`` does, which makes the normal of
    the whole the sum of the parts' own normals.

    That last part matters.  A^H A for a Toeplitz encoding is a
    convolution rather than two transforms, and it stays one here: the normal
    is built as ``A.gram() + sum_i w_i G_i.gram()`` and attached with
    ``linop_from_ops``, so the encoding is asked for its normal rather than
    for its forward and its adjoint.
    """
    from bartorch.linop import Reshape, concatenate
    from bartorch.linop._base import _WithNormal

    op = A._bart()
    y = as_operand(y, op.oshape, "y")

    def flat(operator):
        size = math.prod(operator.oshape)
        return Reshape((size,), operator.oshape) @ operator

    pieces = [flat(A)]
    data = [y.reshape(-1)]
    normal = A.gram()

    for term in terms:
        G = term._operator(op.ishape)
        if G.ishape != op.ishape:
            raise ValueError(f"a term over {G.ishape} does not fit an encoding over {op.ishape}")
        # The weight meets a complex64 tensor, so its square root is taken in
        # the precision that tensor is held at.
        root = float(np.float32(math.sqrt(term.weight)))
        pieces.append(flat(root * G) if 1.0 != root else flat(G))

        if term.bias is None:
            data.append(torch.zeros(math.prod(G.oshape), dtype=y.dtype, device=y.device))
        else:
            data.append(root * as_operand(term.bias, G.oshape, "bias").reshape(-1))

        normal = normal + (term.weight * G.gram() if 1.0 != term.weight else G.gram())

    return _WithNormal(concatenate(pieces), normal), torch.cat(data)


Regularizers = Regularizer | Iterable[Regularizer] | None


def maxeigen(
    A, terms: Terms = None, *, cclambda: float = 0.0, precond=None, iterations: int = 30
) -> float:
    r"""Estimate the largest eigenvalue of the normal operator by power iteration.

    The operator is :math:`M (A^H A + \lambda I) + \sum_j G_j^H G_j`, as BART's
    ``lsqr2_create`` and ``iter2_chambolle_pock`` form it, with :math:`M` the
    preconditioner and :math:`G_j` the terms' transforms.  A solver given
    ``eigen=True`` divides its step by this value.

    Parameters
    ----------
    A : LinearOperator
        Encoding operator.
    terms : Regularizer or iterable of Regularizer, default=None
        Terms whose :math:`G_j^H G_j` is added.  The primal-dual iteration
        passes its dual terms; the proximal-gradient iterations pass none.
    cclambda : float, default=0.0
        :math:`\lambda`.
    precond : LinearOperator, default=None
        :math:`M`, from ``A.ishape`` to itself.
    iterations : int, default=30
        Power iterations; BART uses 30.

    Returns
    -------
    float
        The estimated eigenvalue.

    Raises
    ------
    ~bartorch.BartError
        The library could not form the operator or run the iteration.

    Notes
    -----
    The starting vector is drawn from BART's process-wide random generator.  An
    iteration written outside the library calls this where BART's own loop
    would, so that later draws -- a wavelet term's random shifts, for
    instance -- follow the same sequence.
    """
    op = A._bart()
    _ensure_ready()
    lib = library()

    handles = [term.build(op.ishape) for term in _as_terms(terms)]
    conditioner = _conditioner(precond, op.ishape)
    value = _marshal.double_out()
    with _lock, _on_device(op.device or torch.device("cpu")):
        code = lib.bartorch_maxeigen(
            op._h.ptr,
            None if conditioner is None else conditioner._h.ptr,
            float(cclambda),
            len(handles),
            _marshal.pointers(handles) if handles else None,
            int(iterations),
            _marshal.by_reference(value),
        )
    if code != 0:
        raise BartError("the largest eigenvalue could not be estimated")
    return float(value.value)


def _conditioner(precond, shape):
    """``precond`` as a BART operator, checked to map the image to itself."""
    if precond is None:
        return None
    _preconditioner(precond, shape)
    return precond._bart()


def _solve(
    A,
    y: torch.Tensor,
    x0: torch.Tensor | None,
    algorithm: str,
    terms: list[Regularizer],
    *,
    maxiter: int,
    cclambda: float,
    step: float = -1.0,
    eigen: bool = False,
    hogwild: bool = False,
    rho: float = -1.0,
    cg_maxiter: int = 0,
    cg_tol: float = 0.0,
    dynamic_rho: bool = False,
    dynamic_tau: bool = False,
    relative_norm: bool = False,
    fast: bool = False,
    pqr: tuple[float, float, float] | None = None,
    sigma_tau_ratio: float = 1.0,
    adaptive_step: bool = False,
    precond=None,
    steps: list | None = None,
) -> torch.Tensor:
    """Run ``bartorch_solve``.  A negative ``step`` or ``rho``, a zero
    ``cg_maxiter`` and ``pqr=None`` keep BART's defaults.

    ``steps``, when given, has the number of iterations the algorithm took
    appended to it."""
    op = A._bart()
    y = as_operand(y, op.oshape, "y")
    if x0 is None:
        x = torch.zeros(op.ishape, dtype=torch.complex64, device=y.device)
    else:
        x = as_operand(x0, op.ishape, "x0").clone()

    _ensure_ready()
    lib = library()
    ndim = len(op.ishape)
    flags = [term._flags(ndim) for term in terms]
    # A term with auxiliary variables cannot be built on its own -- the offsets its
    # transforms sit at are worked out across the whole set -- so on that path
    # the solve configures the set itself and there is nothing to hand over.
    extends = _extending(terms)
    if extends:
        _refuse_preconditioner(precond, "BART's loop")
    for term in terms:
        term._check(ndim)
    handles = [] if extends else [term.build(op.ishape) for term in terms]
    p, q, r = pqr if pqr is not None else (-1.0, -1.0, -1.0)
    # The preconditioner is one more BART operator, and it has to outlive the
    # call; the wrapper a Python-defined one produces is kept here for that.
    conditioner = _conditioner(precond, op.ishape)

    # `opt_reg_configure` takes one block size, one wavelet family and one
    # shift mode for the whole set, and reaches for them only on the path that
    # configures the set -- which is the path an extending term forces.
    block, family, shift_mode = _shared_options(terms)
    alpha, gamma = _shared_pairs(terms)

    counter = _marshal.int64_out()
    with _lock, _on_device(op.device or y.device):
        code = lib.bartorch_solve(
            op._h.ptr,
            algorithm.encode(),
            _marshal.argv([term.kind for term in terms]) if terms else None,
            _marshal.uint64s([f for f, _ in flags]) if terms else None,
            _marshal.uint64s([j for _, j in flags]) if terms else None,
            _marshal.floats([term.weight for term in terms]) if terms else None,
            _marshal.ints([term.count for term in terms]) if terms else None,
            _marshal.pointers(handles) if handles else None,
            len(terms),
            float(cclambda),
            int(maxiter),
            float(step),
            int(eigen),
            int(hogwild),
            float(rho),
            int(cg_maxiter),
            float(cg_tol),
            int(dynamic_rho),
            int(dynamic_tau),
            int(relative_norm),
            int(fast),
            float(p),
            float(q),
            float(r),
            float(sigma_tau_ratio),
            int(adaptive_step),
            int(x0 is not None),
            None if conditioner is None else conditioner._h.ptr,
            block,
            family.encode(),
            shift_mode,
            _marshal.floats(alpha),
            _marshal.floats(gamma),
            x.data_ptr(),
            y.data_ptr(),
            _marshal.by_reference(counter) if steps is not None else None,
        )
    if code != 0:
        said = lib.bartorch_solve_error(code).decode(errors="replace")
        raise BartError(f"the solve failed: {said}")
    if steps is not None:
        steps.append(int(counter.value))
    return x


#: The two iterations that take a term's transform, and so the only two BART
#: lets an extending term reach.  `italgo_choose` sends every one of the three
#: to the alternating directions.
_TAKES_A_TRANSFORM = ("admm", "pridu")

#: What `opt_reg_configure` falls back on, and what a term that does not read
#: these answers with.
_SHARED_OPTIONS = (8, "dau2", 1)


#: The pairs BART takes once for the whole set, and its own values for
#: them (`opt_reg_init`, optreg.c:309-313).  Only the terms with auxiliary variables
#: read them.
_SHARED_PAIRS: dict[str, tuple[float, float]] = {
    "alpha": (1.0, 3.0**0.5),
    "gamma": (1.0, 1.0),
}


def _shared_pairs(terms) -> tuple[tuple[float, float], tuple[float, float]]:
    """The ``alpha`` and ``gamma`` pairs for the whole set.

    They live on ``struct opt_reg_s`` rather than on a term, so the set carries
    one of each; two terms disagreeing is refused rather than one of them
    silently winning, as it is for the block size.
    """
    resolved = dict(_SHARED_PAIRS)
    asked: dict[str, tuple[float, float]] = {}
    for term in terms:
        for name, value in term._settings().items():
            if name not in resolved:
                continue
            if asked.setdefault(name, value) != value:
                raise ValueError(
                    f"{name} is one pair for the whole set, and these terms ask for "
                    f"{asked[name]!r} and {value!r}; "
                    "give them the same, or solve for them separately"
                )
            resolved[name] = value
    return resolved["alpha"], resolved["gamma"]


def _extending(terms) -> bool:
    """Whether any of ``terms`` extends the optimization variable with auxiliary ones."""
    return any(getattr(term, "_extends", False) for term in terms)


#: `NUM_REGS`, the most penalties `opt_reg_configure` makes of a set.
_MAX_PENALTIES = 10


def _penalties(terms, image_shape: tuple[int, ...]):
    """The penalties BART splits such a set into, and how many variables it adds.

    ``opt_reg_configure`` works the offsets out across the whole set, as
    ``bartorch_solve`` does; each penalty's transform maps from the image's
    entries followed by the supporting ones.
    """
    from bartorch.priors._base import _Frozen, _Penalty

    ndim = len(image_shape)
    for term in terms:
        term._check(ndim)
    flags = [term._flags(ndim) for term in terms]
    block, family, shift_mode = _shared_options(terms)
    alpha, gamma = _shared_pairs(terms)

    _ensure_ready()
    handles = _marshal.pointer_buffer(_MAX_PENALTIES)
    count, svars = _marshal.int_out(), _marshal.int64_out()
    with _lock:
        code = library().bartorch_prox_set_create(
            len(terms),
            _marshal.argv([term.kind for term in terms]),
            _marshal.uint64s([f for f, _ in flags]),
            _marshal.uint64s([j for _, j in flags]),
            _marshal.floats([term.weight for term in terms]),
            _marshal.ints([term.count for term in terms]),
            block,
            family.encode(),
            shift_mode,
            _marshal.floats(alpha),
            _marshal.floats(gamma),
            _marshal.padded_dims(tuple(image_shape)),
            _MAX_PENALTIES,
            handles,
            _marshal.by_reference(count),
            _marshal.by_reference(svars),
        )
    if code != 0:
        said = library().bartorch_solve_error(code).decode(errors="replace")
        raise BartError(f"the terms could not be configured together: {said}")

    frozen = all(isinstance(term, _Frozen) for term in terms)
    shape = (math.prod(image_shape) + int(svars.value),)
    return [_Penalty(handles[i], shape, frozen) for i in range(count.value)], int(svars.value)


def _shared_options(terms) -> tuple[int, str, int]:
    """The one block size, wavelet family and shift mode for the whole set.

    ``opt_reg_configure`` takes one of each and hands them to whichever terms
    read them, so the set carries a single block size and a single family.
    A term that reads none answers with the defaults, so what is looked for is
    the terms that said something, and two of those disagreeing is refused
    rather than silently resolved.
    """
    asked = {term._options() for term in terms if term._options() != _SHARED_OPTIONS}
    if not asked:
        return _SHARED_OPTIONS
    if 1 < len(asked):
        raise ValueError(
            "BART configures a set of terms with one block size, one wavelet family and "
            f"one shift mode -- BART takes one of each for the whole set -- and these ask "
            f"for "
            f"{sorted(asked)}; give them the same, or solve for them separately"
        )
    return next(iter(asked))


def _item_by_item(solve, y, A, x0, **kwargs):
    """A batch in front of ``y`` solved one item at a time; ``None`` for a single item.

    BART runs one solve per item: its stopping rule, its adaptive steps and a
    term's random shifts all belong to the run, so a batch taken as one would
    share them between items.
    """
    y = torch.as_tensor(y)
    if y.ndim != len(A.oshape) + 1:
        return None
    if x0 is None:
        starts = [None] * len(y)
    else:
        x0 = torch.as_tensor(x0)
        starts = list(x0) if x0.ndim == len(A.ishape) + 1 else [x0] * len(y)
    return torch.stack([solve(item, A, start, **kwargs) for item, start in zip(y, starts)])


class _Solver:
    """Base of the solvers: a block looped to BART's schedule, or BART's ``lsqr2``."""

    #: The name ``bartorch_solve`` selects the iteration by.
    _algorithm = ""

    def __init__(self, regularizers: Regularizers, maxiter: int, cclambda: float, precond=None):
        self.regularizers = _as_terms(regularizers)
        if _extending(self.regularizers) and self._algorithm not in _TAKES_A_TRANSFORM:
            raise TypeError(
                f"{type(self).__name__} cannot take a term with auxiliary variables in the "
                "optimization: total generalized variation and the two infimal convolutions "
                "split into several penalties over the enlarged variable, and only the "
                "alternating-direction and primal-dual iterations are given a term's "
                "transform at all; optim.ADMM takes them"
            )
        self.maxiter = int(maxiter)
        self.cclambda = float(cclambda)
        self.precond = precond

    @property
    def _foreign(self) -> list:
        """The terms BART could not have been given, such as an ``ImplicitPrior``."""
        return [t for t in self.regularizers if not isinstance(t, Regularizer)]

    def _settings(self) -> dict:
        return {}

    def _block(self):
        """The step this solver loops over, built from its settings."""
        raise NotImplementedError

    def _declines(self, state) -> bool:
        """Whether BART would leave the start as it is without iterating."""
        return False

    def _stops(self, state) -> bool:
        return getattr(state, "done", False)

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve for the image.

        Parameters
        ----------
        y : tensor
            Data of ``A.oshape``, or ``(batch, *A.oshape)``; a batch is solved
            one item at a time, each as its own run.
        A : LinearOperator
            Encoding operator.  A BART-backed operator is applied inside the
            library; a Python-defined one is called back once per application.
        x0 : tensor, default=None
            Starting image of ``A.ishape``, or one per batch item; zero when omitted.

        Returns
        -------
        torch.Tensor
            Complex64 image of ``A.ishape``, with ``y``'s batch axis in front.

        Raises
        ------
        ValueError
            ``y`` requires a gradient and a BART term has not been detached
            (:meth:`~bartorch.priors.Regularizer.detach`): its proximal operator has
            no backward pass.

        Notes
        -----
        The iterations are torch operations, so a solve is differentiated by
        unrolling.  Step sizes and weights are fixed; :mod:`bartorch.optim`'s
        blocks and :class:`~bartorch.learning.Unrolled` expose them as parameters.
        """
        made = _item_by_item(self, y, A, x0)
        if made is not None:
            return made
        block = self._block()
        state = block.start(y, A, x0)
        if self._declines(state):
            return state.x
        for _ in range(self.maxiter):
            state = block(state, A)
            if self._stops(state):
                break
        return block.output(state, A)

    def _in_library(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """BART's own loop: the reference a block is held against, and records nothing."""
        if self._foreign:
            raise ValueError(
                f"{self._foreign[0]!r} is not a term BART can be given, so there is no "
                "library route for this solve"
            )
        return _solve(
            A,
            y,
            x0,
            self._algorithm,
            self.regularizers,
            maxiter=self.maxiter,
            cclambda=self.cclambda,
            precond=self.precond,
            **self._settings(),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.regularizers!r}, maxiter={self.maxiter})"


class CG(_Solver):
    r"""Conjugate gradients on the normal equations of a quadratic problem.

    Solves

    .. math::

        \min_x \; \| A x - y \|^2 + \lambda \| x \|^2
              + \sum_i w_i \| G_i x - b_i \|^2

    where the sum is over the :class:`Tikhonov` terms.

    Parameters
    ----------
    lambda_ : float, default=0.0
        :math:`\lambda`, the Tikhonov weight on the image.
    terms : Tikhonov or iterable of Tikhonov, default=None
        Quadratic penalties :math:`w_i \| G_i x - b_i \|^2`.
    maxiter : int, default=30
        Maximum number of iterations.
    tol : float, default=0.0
        Stop once the residual of the normal equations is at most
        ``tol * ||A^H y||``; zero, BART's default, runs every iteration.
    cclambda : float, default=0.0
        Weight of an identity added to the normal operator (``lsqr``'s
        ``cclambda``).
    precond : LinearOperator, default=None
        Left preconditioner :math:`M` from ``A.ishape`` to itself, positive
        definite (``lsqr2_create``'s ``precond_op``); BART's reconstructions use none.

    Raises
    ------
    TypeError
        A term is not a :class:`Tikhonov` penalty, such as a proximal
        regularizer.

    Notes
    -----
    BART's ``iter2_conjgrad`` takes the weight :math:`\lambda` and no
    regularizing operators or biases.  The terms are therefore stacked under the
    encoding, :math:`[A; \sqrt{w_i} G_i] x \approx [y; \sqrt{w_i} b_i]`, and the
    stacked operator is solved; its normal is applied as the sum of the parts'
    normals.

    Examples
    --------
    >>> CG(maxiter=30)(kspace, A)
    >>> CG(terms=Tikhonov(0.1, bias=prior))(kspace, A)
    >>> CG(terms=[Tikhonov(0.1, operator=G), Tikhonov(0.01)])(kspace, A)
    """

    _algorithm = "cg"

    def __init__(
        self,
        lambda_: float = 0.0,
        *,
        terms: Terms = None,
        maxiter: int = 30,
        tol: float = 0.0,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(L2(lambda_) if lambda_ else None, maxiter, cclambda, precond)
        self.lambda_ = float(lambda_)
        self.terms = _as_quadratics(terms)
        self.tol = float(tol)

    def _settings(self) -> dict:
        return {"cg_tol": self.tol}

    def __call__(
        self,
        y: torch.Tensor,
        A,
        x0: torch.Tensor | None = None,
        *,
        steps: list | None = None,
    ) -> torch.Tensor:
        """Solve for the image.

        Parameters
        ----------
        y : tensor
            Data of ``A.oshape``, or ``(batch, *A.oshape)``; a batch is solved
            one item at a time.
        A : LinearOperator
            Encoding operator.
        x0 : tensor, default=None
            Starting image of ``A.ishape``, or one per batch item; zero when omitted.
        steps : list, default=None
            When given, the number of iterations each solve took is appended to it.

        Returns
        -------
        torch.Tensor
            Complex64 image of ``A.ishape``, with ``y``'s batch axis in front.

        Notes
        -----
        When ``y`` requires a gradient the solve is recorded as one operation:
        the forward pass is the library's iteration, and the backward pass a
        second solve with the same normal operator
        (:doc:`/explanation/learned-reconstruction`).  The warm start receives no
        gradient.  For a linearization of a nonlinear operator the gradient also
        reaches the point it was linearized at.  The solve can therefore serve as
        the data-consistency layer of an unrolled network.
        """
        from bartorch.linop._base import _tracking

        made = _item_by_item(self, y, A, x0, steps=steps)
        if made is not None:
            return made

        if self.terms:
            A, y = _stacked(A, y, self.terms)

        def forward(data: torch.Tensor) -> torch.Tensor:
            return _solve(
                A,
                data,
                x0,
                self._algorithm,
                self.regularizers,
                maxiter=self.maxiter,
                cclambda=self.cclambda,
                precond=self.precond,
                steps=steps,
                **self._settings(),
            )

        if getattr(A, "_records", False) and not self.terms:
            return self._at_a_point(y, A, forward)

        if not _tracking(y):
            return forward(y)

        from bartorch.optim._autograd import apply_solve

        return apply_solve(y, forward, lambda g: A.forward(self._inverse(A, g)))

    def _at_a_point(self, y, A, forward):
        """The solve over an encoding linearized at a point, recorded through the encoding.

        ``b = A^H y`` is recorded by the encoding itself, which differentiates by
        the point as well; ``N(p)^-1 b`` is ADMM's resolvent with no penalty,
        whose backward pass gives the point ``-d/dp Re <w, N(p) x>``.  The
        forward pass is the same library solve either way.
        """
        from bartorch.linop._autograd import apply_normal
        from bartorch.linop._base import _tracking
        from bartorch.optim._blocks import _moving, _Resolvent

        point = _moving(A)
        if not (_tracking(y) or point is not None):
            return forward(y)
        rhs = A.adjoint(y)

        def solve(b, warm):
            return forward(y) if warm else self._inverse(A, b)

        def data(v, at):
            return apply_normal(A.at(at), v)

        rate = torch.zeros((), dtype=torch.float64)
        return _Resolvent.apply(rhs, rate, point, solve, None, data)

    def _inverse(self, A, g: torch.Tensor) -> torch.Tensor:
        """``N^-1 g``, driven from the right-hand side rather than from data.

        ``_WithNormal(Identity, A.gram())`` is an operator whose adjoint is the
        identity and whose normal is ``A^H A``, so conjugate gradients on it
        solves ``N w = g`` instead of ``N w = A^H g``.  The same terms and the
        same weight go in, so it is the same ``N`` the forward pass inverted;
        and an encoding built with ``toeplitz=True`` keeps its point-spread
        convolution through :meth:`~bartorch.linop.LinearOperator.gram`, so the
        backward pass costs the same as the forward one.
        """
        from bartorch.linop import Identity
        from bartorch.linop._base import _WithNormal

        shape = A.ishape
        return _solve(
            _WithNormal(Identity(shape), A.gram()),
            g,
            None,
            self._algorithm,
            self.regularizers,
            maxiter=self.maxiter,
            cclambda=self.cclambda,
            precond=self.precond,
            **self._settings(),
        )

    def _in_library(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        return self(y, A, x0)

    def __repr__(self) -> str:
        terms = f", terms={self.terms!r}" if self.terms else ""
        return f"CG(lambda_={self.lambda_}{terms}, maxiter={self.maxiter}, tol={self.tol})"


class IST(_Solver):
    r"""Iterative soft thresholding (proximal gradient), looping :class:`ISTBlock`.

    Solves :math:`\min_x \tfrac12 \| A x - y \|^2 + g(x)`.

    Parameters
    ----------
    regularizers : Regularizer or ImplicitPrior, default=None
        The single term :math:`g`; its transform must be the identity.
    maxiter : int, default=30
        Number of iterations.
    step : float, default=0.95
        Step size :math:`\tau` of the gradient step.
    eigen : bool, default=False
        Divide the step by :func:`maxeigen` of the normal operator.
    hogwild : bool, default=False
        Must be false: BART's IST rejects it.
    cclambda : float, default=0.0
        Weight of an identity added to the normal operator.
    precond : LinearOperator, default=None
        Left preconditioner :math:`M` from ``A.ishape`` to itself, positive
        definite (``lsqr2_create``'s ``precond_op``); BART's reconstructions use none.

    Raises
    ------
    ValueError
        Not exactly one term, or ``hogwild``; at the call, a term whose
        transform is not the identity, such as
        :class:`~bartorch.priors.TotalVariation`.
    TypeError
        A term with auxiliary variables.
    """

    _algorithm = "ist"

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        step: float = 0.95,
        eigen: bool = False,
        hogwild: bool = False,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        if 1 != len(self.regularizers):
            # `iter2_ist` and `iter2_fista` assert one, and an assertion in
            # the library takes the process rather than coming back as an error.
            raise ValueError(
                f"{type(self).__name__} takes exactly one term, not "
                f"{len(self.regularizers)}; ADMM is the one that splits several apart"
            )
        if hogwild and "ist" == self._algorithm:
            # `iter2_ist` asserts it off, and an assertion takes the process.
            raise ValueError(
                "BART's iterative soft thresholding refuses hogwild; FISTA is the one "
                "that decays its step"
            )
        self.step = float(step)
        self.eigen = bool(eigen)
        self.hogwild = bool(hogwild)

    def _settings(self) -> dict:
        return {"step": self.step, "eigen": self.eigen, "hogwild": self.hogwild}

    def _block(self):
        return ISTBlock(
            self.regularizers[0],
            step=self.step,
            eigen=self.eigen,
            cclambda=self.cclambda,
            precond=self.precond,
        )

    def _declines(self, state) -> bool:
        return _empty(state.adjoint)

    def _in_library(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        _refuse_transform(self.regularizers[0], A.ishape, type(self).__name__)
        return super()._in_library(y, A, x0)


class FISTA(IST):
    r"""Fast iterative soft thresholding, looping :class:`FISTABlock`.

    Solves :math:`\min_x \tfrac12 \| A x - y \|^2 + g(x)` with Nesterov
    momentum.

    Parameters
    ----------
    regularizers : Regularizer or ImplicitPrior, default=None
        The single term :math:`g`; its transform must be the identity.
    maxiter : int, default=30
        Number of iterations.
    step : float, default=0.95
        Step size :math:`\tau` of the gradient step.
    eigen : bool, default=False
        Divide the step by :func:`maxeigen` of the normal operator.
    hogwild : bool, default=False
        Halve the step after 10 iterations, and again after each further 20,
        40, ... iterations.
    pqr : tuple of float, default=None
        :math:`(p, q, r)` of the momentum update
        :math:`t_{k+1} = (p + \sqrt{q + r t_k^2}) / 2`; ``None`` is BART's
        ``(1, 1, 4)``.
    cclambda : float, default=0.0
        Weight of an identity added to the normal operator.
    precond : LinearOperator, default=None
        Left preconditioner :math:`M` from ``A.ishape`` to itself, positive
        definite (``lsqr2_create``'s ``precond_op``); BART's reconstructions use none.

    Raises
    ------
    ValueError
        Not exactly one term; at the call, a term whose transform is not the
        identity.
    TypeError
        A term with auxiliary variables.

    Examples
    --------
    >>> fista = optim.FISTA(priors.Wavelet((-1, -2), 0.01), maxiter=50)
    >>> x = fista(kspace, A)
    >>> x = fista(kspace, A, x0=x)                # warm start

    A term over a transform is refused, since the iteration is not given the
    transform:

    >>> optim.FISTA(priors.TotalVariation((-1, -2), 0.01))(kspace, A)
    Traceback (most recent call last):
        ...
    ValueError: FISTA applies the proximal operator to the image, ...
    """

    _algorithm = "fista"

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        step: float = 0.95,
        eigen: bool = False,
        hogwild: bool = False,
        pqr: tuple[float, float, float] | None = None,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(
            regularizers,
            maxiter=maxiter,
            step=step,
            eigen=eigen,
            hogwild=hogwild,
            precond=precond,
            cclambda=cclambda,
        )
        self.pqr = None if pqr is None else tuple(float(v) for v in pqr)

    def _settings(self) -> dict:
        return {**super()._settings(), "pqr": self.pqr}

    def _block(self):
        return FISTABlock(
            self.regularizers[0],
            step=self.step,
            eigen=self.eigen,
            hogwild=self.hogwild,
            pqr=self.pqr,
            cclambda=self.cclambda,
            precond=self.precond,
        )


class ADMM(_Solver):
    r"""Alternating direction method of multipliers, looping :class:`ADMMBlock`.

    Solves :math:`\min_x \tfrac12 \| A x - y \|^2 + \sum_j g_j(G_j x - b_j)`,
    splitting each term at :math:`G_j x`; each x-update is a conjugate-gradient
    solve.

    Parameters
    ----------
    regularizers : Regularizer or ImplicitPrior, or an iterable of them, default=None
        The terms :math:`g_j`, with any transform :math:`G_j`, including terms
        with auxiliary variables.
    maxiter : int, default=30
        Budget of conjugate-gradient iterations over the whole run, not of
        outer iterations: the run stops once the inner iterations spent exceed
        it, as BART's ``admm`` does.
    rho : float, default=0.5
        Penalty parameter :math:`\rho`.
    cg_maxiter : int, default=10
        Conjugate-gradient iterations per x-update.
    hogwild : bool, default=False
        Double :math:`\rho` after iteration 1, 3, 7, 15, ...  Excludes
        ``dynamic_rho``.
    cclambda : float, default=0.0
        Weight of an identity added to the normal operator.
    biases : sequence of tensor, default=None
        The :math:`b_j`, one per term, each of its term's
        :meth:`~bartorch.priors.Regularizer.prox_shape`.
    dynamic_rho : bool, default=False
        Multiply :math:`\rho` by ``tau`` when the primal residual exceeds ``mu``
        times the dual, divide it when the dual exceeds ``mu`` times the
        primal; the scaled duals are rescaled to match.
    dynamic_tau : bool, default=False
        Set ``tau`` from the residuals, :math:`\sqrt{r / s}` or its inverse,
        clipped to ``tau_max`` (residual balancing, Wohlberg 2017).
    relative_norm : bool, default=False
        Compare the residuals each relative to its scaling.
    fast : bool, default=False
        Skip the residuals, and with them the stopping test.  Excludes
        ``dynamic_rho``.
    alpha : float, default=1.6
        Over-relaxation parameter.
    mu : float, default=3.0
        Residual ratio beyond which ``dynamic_rho`` rescales :math:`\rho`.
    tau_max : float, default=20.0
        Upper bound of ``tau``.
    abstol : float, default=0.0
        Absolute tolerance of the residual stopping test (Boyd et al. 2011).
    reltol : float, default=0.0
        Relative tolerance of the residual stopping test.  With both
        tolerances zero, as ``italgo_config`` sets them, only ``maxiter``
        ends the run.
    cg_maxiter_first : int, default=None
        Conjugate-gradient iterations of the first x-update; ``cg_maxiter``
        when omitted.
    precond : LinearOperator, default=None
        Left preconditioner :math:`M` from ``A.ishape`` to itself, positive
        definite (``lsqr2_create``'s ``precond_op``); BART's reconstructions use none.

    Raises
    ------
    ValueError
        ``hogwild`` or ``fast`` with ``dynamic_rho``, or not one bias per term.

    Notes
    -----
    ``alpha``, ``mu``, ``tau_max``, the tolerances, the biases and
    ``cg_maxiter_first`` have no counterpart in ``italgo_config``, so BART's
    own loop runs with their defaults.

    Examples
    --------
    Several terms, each split off with its own transform:

    >>> terms = [priors.TotalVariation((-1, -2), 0.005), priors.L1(0.001)]
    >>> x = optim.ADMM(terms, maxiter=30)(kspace, A)
    """

    _algorithm = "admm"

    #: What `italgo_config` gives no way to set, and each one's value when
    #: nothing was asked for.
    _beyond_the_tool = {
        "alpha": 1.6,
        "mu": 3.0,
        "tau_max": 20.0,
        "abstol": 0.0,
        "reltol": 0.0,
        "cg_maxiter_first": None,
        "biases": None,
    }

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        rho: float = 0.5,
        cg_maxiter: int = 10,
        hogwild: bool = False,
        cclambda: float = 0.0,
        biases: Sequence[torch.Tensor] | None = None,
        dynamic_rho: bool = False,
        dynamic_tau: bool = False,
        relative_norm: bool = False,
        fast: bool = False,
        alpha: float = 1.6,
        mu: float = 3.0,
        tau_max: float = 20.0,
        abstol: float = 0.0,
        reltol: float = 0.0,
        cg_maxiter_first: int | None = None,
        precond=None,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        self.rho = float(rho)
        self.cg_maxiter = int(cg_maxiter)
        self.hogwild = bool(hogwild)
        self.dynamic_rho = bool(dynamic_rho)
        self.dynamic_tau = bool(dynamic_tau)
        self.relative_norm = bool(relative_norm)
        self.fast = bool(fast)
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.tau_max = float(tau_max)
        self.abstol = float(abstol)
        self.reltol = float(reltol)
        self.cg_maxiter_first = None if cg_maxiter_first is None else int(cg_maxiter_first)
        self.biases = None if biases is None else list(biases)
        # The block refuses what BART asserts apart, while an exception still
        # reaches the caller.
        self._block()

    def _settings(self) -> dict:
        return {
            "rho": self.rho,
            "cg_maxiter": self.cg_maxiter,
            "hogwild": self.hogwild,
            "dynamic_rho": self.dynamic_rho,
            "dynamic_tau": self.dynamic_tau,
            "relative_norm": self.relative_norm,
            "fast": self.fast,
        }

    def _block(self):
        return ADMMBlock(
            self.regularizers,
            rho=self.rho,
            alpha=self.alpha,
            cg_maxiter=self.cg_maxiter,
            cg_maxiter_first=self.cg_maxiter_first,
            cclambda=self.cclambda,
            biases=self.biases,
            dynamic_rho=self.dynamic_rho,
            dynamic_tau=self.dynamic_tau,
            relative_norm=self.relative_norm,
            fast=self.fast,
            hogwild=self.hogwild,
            mu=self.mu,
            tau_max=self.tau_max,
            abstol=self.abstol,
            reltol=self.reltol,
            precond=self.precond,
        )

    def _stops(self, state) -> bool:
        return state.done or state.invokes > self.maxiter

    def _in_library(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        asked = [
            name
            for name, default in self._beyond_the_tool.items()
            if getattr(self, name) != default
        ]
        if asked:
            raise ValueError(
                f"{', '.join(sorted(asked))} cannot be set on BART's own loop -- "
                "`italgo_config` has no way to pass them"
            )
        return super()._in_library(y, A, x0)


class PRIDU(_Solver):
    r"""Primal-dual iteration (Chambolle-Pock), looping :class:`PRIDUBlock`.

    Solves :math:`\min_x \tfrac12 \| A x - y \|^2 + \sum_j g_j(G_j x)` with no
    inner solve.

    Parameters
    ----------
    regularizers : Regularizer or ImplicitPrior, or an iterable of them, default=None
        The terms :math:`g_j`, with any transform :math:`G_j`, including terms
        with auxiliary variables.
    maxiter : int, default=30
        Maximum number of iterations.
    step : float, default=0.95
        Product of the dual and primal step sizes, :math:`\sigma \tau`.
    sigma_tau_ratio : float, default=1.0
        :math:`\sigma / \tau`, so :math:`\sigma = \sqrt{\text{step}}\,\text{ratio}`
        and :math:`\tau = \sqrt{\text{step}} / \text{ratio}`.  BART's
        reconstructions set it to the :func:`data_scaling` factor.
    adaptive_step : bool, default=False
        Adapt :math:`\sigma` and :math:`\tau` at every iteration.
    eigen : bool, default=False
        Divide :math:`\sigma` and :math:`\tau` by the square root of
        :func:`maxeigen` over the encoding and the dual terms.
    hogwild : bool, default=False
        Relax iteration :math:`k` by :math:`0.95^k`: each variable moves that
        fraction of the way to its update.
    cclambda : float, default=0.0
        Weight of an identity added to the normal operator.
    precond : LinearOperator, default=None
        Left preconditioner :math:`M` from ``A.ishape`` to itself, positive
        definite (``lsqr2_create``'s ``precond_op``); BART's reconstructions use none.

    Notes
    -----
    The first term is applied as the primal proximal step when its transform
    is the identity, and every other term through a dual variable.  The run
    stops early when the primal and dual residuals sum below 1e-4.
    """

    _algorithm = "pridu"

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        step: float = 0.95,
        sigma_tau_ratio: float = 1.0,
        adaptive_step: bool = False,
        eigen: bool = False,
        hogwild: bool = False,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        self.step = float(step)
        self.sigma_tau_ratio = float(sigma_tau_ratio)
        self.adaptive_step = bool(adaptive_step)
        self.eigen = bool(eigen)
        self.hogwild = bool(hogwild)

    def _settings(self) -> dict:
        return {
            "step": self.step,
            "sigma_tau_ratio": self.sigma_tau_ratio,
            "adaptive_step": self.adaptive_step,
            "eigen": self.eigen,
            "hogwild": self.hogwild,
        }

    def _block(self):
        return PRIDUBlock(
            self.regularizers,
            step=self.step,
            sigma_tau_ratio=self.sigma_tau_ratio,
            adaptive_step=self.adaptive_step,
            eigen=self.eigen,
            hogwild=self.hogwild,
            cclambda=self.cclambda,
            precond=self.precond,
        )


class NIHT(_Solver):
    """Normalized iterative hard thresholding.

    Cannot be run; see :meth:`__call__`.

    Parameters
    ----------
    regularizers : WaveletNIHT or ImageNIHT, or an iterable of them
        The hard-thresholding terms from :mod:`bartorch.priors`.
    maxiter : int, default=30
    cclambda : float, default=0.0
        Weight of an identity added to the normal operator.
    precond : LinearOperator, default=None
        Left preconditioner, ``lsqr2_create``'s ``precond_op``: chained onto
        the normal operator and onto the adjoint, so the iteration sees
        ``M(A^H A + lambda) x = M A^H y``.  Must be positive definite --
        BART composes it without symmetrizing.  BART's own reconstructions
        pass none.
    """

    _algorithm = "niht"

    def __init__(
        self,
        regularizers: Regularizers,
        *,
        maxiter: int = 30,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        for term in self.regularizers:
            if term.kind not in ("H", "N"):
                raise TypeError(f"NIHT takes WaveletNIHT and ImageNIHT terms, not {term!r}")

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Always raises: BART's ``niht`` cannot run against ``lsqr``'s operator.

        ``niht`` applies the normal operator in place -- ``iter_op_call(op, g,
        g)`` at ``iter/niht.c:85`` and ``:212`` -- while the operator ``lsqr2``
        supplies asserts that its arguments are not aliased, ``args[0] !=
        args[1]`` at ``iter/lsqr.c:60``.  Every NIHT solve therefore terminates
        in an assertion; here those assertions are
        ``error()`` calls that unwind rather than ending the process.
        """
        raise NotImplementedError(
            "BART's NIHT cannot run: `niht` applies the normal operator in place "
            "(iter/niht.c:85, :212) and the operator `lsqr2` hands it asserts that it "
            "is not (iter/lsqr.c:60), so every solve ends in an assertion.  Nothing here "
            "can work around it; it needs a BART fix.  "
            "WaveletNIHT and ImageNIHT terms still reach BART's pics command, which catches "
            "the assertion rather than ending the process"
        )
