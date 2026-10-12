"""Projection onto convex sets: ``italgos.c``'s ``pocs``, and the sweep it repeats."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import torch
from torch import nn

from bartorch.optim._blocks import _prox

__all__ = ["POCS", "POCSBlock"]


def _projections(projections: Iterable) -> list:
    """The sets to project onto, each as a callable on a tensor.

    A :mod:`bartorch.priors` term is taken as its proximal operator at
    ``mu = 1``, which is what ``pocs`` calls a projection with.
    """
    made = []
    for projection in projections:
        if hasattr(projection, "prox"):
            term = projection
            made.append(lambda x, term=term: _prox(term, x, 1.0, tuple(x.shape)))
        elif callable(projection):
            made.append(projection)
        else:
            raise TypeError(
                f"a projection is a callable on a tensor or a bartorch.priors term, "
                f"not {type(projection).__name__}"
            )
    if not made:
        raise ValueError("a sweep of no projections computes nothing")
    return made


class POCSBlock(nn.Module):
    """One sweep of projections onto convex sets, BART's ``pocs`` (``italgos.c``).

    A step applies every projection in turn; there is no step size, momentum or
    residual, so the state is the iterate alone.  The projections carry the data
    and the encoding -- for ``pocsense``, the measured samples, the range of the
    coil sensitivities and a sparsity threshold -- so ``A`` is accepted and
    ignored, giving the block the calling convention of the others for
    :class:`bartorch.learning.Unrolled`.

    Parameters
    ----------
    projections : sequence
        Applied in the order given.  Each is a callable mapping a tensor to a
        tensor, or a :mod:`bartorch.priors` term, applied as its proximal
        operator at unit step.

    Raises
    ------
    TypeError
        A projection is neither callable nor a term.
    ValueError
        ``projections`` is empty.

    Examples
    --------
    >>> block = POCSBlock([consistency, sense, sparsity])
    >>> state = block.start(kspace)
    >>> state = block(state)
    >>> block.output(state)
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        x: torch.Tensor
        k: int = 0

    def __init__(self, projections: Sequence):
        super().__init__()
        self.projections = _projections(projections)
        modules = [p for p in projections if isinstance(p, nn.Module)]
        if modules:
            self.held = nn.ModuleList(modules)

    def start(self, y: torch.Tensor, A=None, x0: torch.Tensor | None = None) -> State:
        """Initial state: ``x0``, or zero of ``y``'s shape, as ``pocs_recon2`` starts.

        ``y`` fixes the shape, dtype and device of the zero start; its values are
        not read.
        """
        y = torch.as_tensor(y)
        if x0 is None:
            return self.State(torch.zeros_like(y))
        return self.State(torch.as_tensor(x0).clone())

    def forward(self, state: State, A=None) -> State:
        """Apply every projection once; returns the next state."""
        x = state.x
        for project in self.projections:
            x = project(x)
        return dataclasses.replace(state, x=x, k=state.k + 1)

    def output(self, state: State, A=None) -> torch.Tensor:
        """The iterate as the last projection left it."""
        return state.x


class POCS:
    """Projection onto convex sets, looping :class:`POCSBlock`.

    ``maxiter`` sweeps of the projections, with no stopping test, as BART's
    ``pocs``.

    Parameters
    ----------
    projections : sequence
        Callables on a tensor, or :mod:`bartorch.priors` terms, as
        :class:`POCSBlock` takes them.
    maxiter : int, default=50
        Number of sweeps; ``pocsense``'s default.

    Examples
    --------
    >>> POCS([consistency, sense, sparsity], maxiter=50)(kspace)
    """

    def __init__(self, projections: Sequence, *, maxiter: int = 50):
        self.projections = list(projections)
        self.maxiter = int(maxiter)

    def _block(self) -> POCSBlock:
        return POCSBlock(self.projections)

    def __call__(self, y: torch.Tensor, A=None, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Sweep the projections.

        Parameters
        ----------
        y : tensor
            Fixes the shape, dtype and device of the zero start; its values are not
            read.
        A : object, default=None
            Ignored.
        x0 : tensor, default=None
            Starting iterate; zero of ``y``'s shape when omitted.

        Returns
        -------
        torch.Tensor
            The iterate after the last sweep.
        """
        block = self._block()
        state = block.start(y, A, x0)
        for _ in range(self.maxiter):
            state = block(state, A)
        return block.output(state, A)
