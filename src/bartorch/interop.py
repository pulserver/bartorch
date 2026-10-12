"""Adapters that hand bartorch's operators to other libraries.

:func:`to_deepinv` makes an operator a ``deepinv`` physics for its losses and
samplers; ``deepinv`` is imported on first use and installed separately.
"""

from __future__ import annotations

import functools

import torch

__all__ = ["to_deepinv"]


def _batched(apply, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """``apply`` over a leading batch axis, when ``x`` has one more axis than ``shape``.

    The batch is walked rather than folded into the operator, whose data
    (trajectory, sensitivities) does not repeat over it.
    """
    if x.ndim == len(shape) + 1:
        return torch.stack([apply(item) for item in x])
    return apply(x)


@functools.lru_cache(maxsize=1)
def _physics_class() -> type:
    """The ``LinearPhysics`` subclass, built once, when ``deepinv`` is present."""
    try:
        from deepinv.physics import LinearPhysics
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "bartorch.interop.to_deepinv() hands an operator over as a deepinv "
            "LinearPhysics, and needs deepinv: pip install deepinv"
        ) from exc

    class BartPhysics(LinearPhysics):
        """A ``deepinv.physics.LinearPhysics`` over a bartorch operator.

        ``A`` and ``A_adjoint`` are the operator's and differentiate through its
        autograd functions; ``A_dagger`` is :class:`bartorch.optim.CG`.

        Parameters
        ----------
        op : LinearOperator
        maxiter, lambda_, tol : int, float, float
            ``A_dagger``'s defaults, overridable per call.
        **kwargs
            Passed to ``LinearPhysics``, for instance a ``noise_model``.
        """

        def __init__(self, op, maxiter: int = 30, lambda_: float = 0.0, tol: float = 1e-6, **kw):
            adjoint = op.H
            super().__init__(
                A=lambda x, **_: _batched(op, x, op.ishape),
                A_adjoint=lambda y, **_: _batched(adjoint, y, op.oshape),
                **kw,
            )
            self.op = op
            self.maxiter = maxiter
            self.lambda_ = lambda_
            self.tol = tol

        def A_dagger(self, y, **kwargs):  # noqa: N802
            """Least-squares solution by :class:`bartorch.optim.CG`."""
            from bartorch.optim import CG

            solver = CG(
                lambda_=kwargs.pop("lambda_", self.lambda_),
                maxiter=kwargs.pop("maxiter", self.maxiter),
                tol=kwargs.pop("tol", self.tol),
            )
            return _batched(lambda item: solver(item, self.op), y, self.op.oshape)

        def __repr__(self) -> str:
            return f"BartPhysics({self.op!r})"

    return BartPhysics


def to_deepinv(op, **kwargs):
    """Wrap a :class:`~bartorch.linop.LinearOperator` as a ``deepinv.physics.LinearPhysics``.

    ``A`` and ``A_adjoint`` apply the operator and its adjoint, differentiably;
    a tensor with one more axis than the operator's shape is a batch, applied
    item by item.  ``A_dagger`` solves by :class:`bartorch.optim.CG`.

    Parameters
    ----------
    op : LinearOperator
        Operator, kept alive as long as the physics.
    **kwargs
        ``maxiter`` (30), ``lambda_`` (0.0) and ``tol`` (1e-6), the defaults of
        ``A_dagger``, each overridable per call; anything else is passed to
        ``LinearPhysics``, such as a ``noise_model``.

    Returns
    -------
    deepinv.physics.LinearPhysics
        The physics, with the operator as its ``op`` attribute.

    Raises
    ------
    ImportError
        ``deepinv`` is not installed.

    Examples
    --------
    >>> physics = interop.to_deepinv(linop.NoncartesianSense(maps, (128, 128), traj=traj))
    >>> physics.A_dagger(kspace[None]).shape       # a batch of one
    torch.Size([1, 128, 128])
    """
    return _physics_class()(op, **kwargs)
