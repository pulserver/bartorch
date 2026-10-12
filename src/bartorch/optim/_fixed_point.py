"""A block driven to its fixed point, differentiated implicitly at that point."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from bartorch.optim._blocks import ADMMBlock, FISTABlock, PRIDUBlock

__all__ = ["FixedPoint"]


def _stationary(block) -> None:
    """Reject a block whose step is not the same map every time: it has no fixed point."""
    if isinstance(block, FISTABlock):
        raise TypeError(
            "FISTA's momentum depends on the iteration count, so its step is a different map "
            "every time and has no fixed point; ISTBlock is the same step without it"
        )
    if isinstance(block, ADMMBlock) and (block.dynamic_rho or block.hogwild or block.rho.ndim):
        raise ValueError("a rho that moves makes every step a different map, with no fixed point")
    if isinstance(block, PRIDUBlock) and (block.adaptive_step or block.hogwild):
        raise ValueError("steps that move make every step a different map, with no fixed point")
    for term in block.modules():
        if not getattr(term, "stationary", True):
            raise ValueError(
                f"{term!r} follows the iteration, with a schedule of sigma or the index, so every "
                "step is a different map, with no fixed point"
            )


def _moving(state) -> list[str]:
    """The fields a step moves: every tensor, or tuple of tensors, but ``A^H y``."""
    names = []
    for field in dataclasses.fields(state):
        value = getattr(state, field.name)
        if "adjoint" == field.name:
            continue
        if isinstance(value, torch.Tensor) or (
            isinstance(value, tuple) and all(isinstance(t, torch.Tensor) for t in value)
        ):
            names.append(field.name)
    return names


def _pack(state, names) -> torch.Tensor:
    parts = []
    for name in names:
        value = getattr(state, name)
        parts.extend([value] if isinstance(value, torch.Tensor) else list(value))
    return torch.cat([p.reshape(-1) for p in parts])


def _unpack(state, names, z: torch.Tensor):
    """``state`` with its moving fields read back from ``z``."""
    fields, at = {}, 0
    for name in names:
        value = getattr(state, name)
        parts = [value] if isinstance(value, torch.Tensor) else list(value)
        made = []
        for part in parts:
            made.append(z[at : at + part.numel()].reshape(part.shape))
            at += part.numel()
        fields[name] = made[0] if isinstance(value, torch.Tensor) else tuple(made)
    return dataclasses.replace(state, **fields)


class FixedPoint(nn.Module):
    """Iteration block run to its fixed point and differentiated there (deep equilibrium).

    The forward iterations run without recording until the state changes by at
    most ``tol`` of its norm, or for ``max_iter`` steps.  The backward pass takes
    one recorded step from the fixed point and iterates the vector-Jacobian
    product of that step to its own fixed point, so memory does not grow with the
    iteration count.

    Parameters
    ----------
    block : nn.Module
        One of :mod:`bartorch.optim`'s iteration blocks.
    max_iter : int, default=100
        Maximum number of forward iterations.
    tol : float, default=1e-4
        Relative change of the state at which the forward iteration stops.
    backward_iter : int, default=None
        Maximum number of backward iterations; ``max_iter`` when omitted.
    backward_tol : float, default=None
        Relative change at which the backward iteration stops; ``tol`` when
        omitted.

    Attributes
    ----------
    iterations : int
        Forward iterations taken by the last call.

    Raises
    ------
    TypeError
        ``block`` is a :class:`FISTABlock`, whose momentum makes every step a
        different map.
    ValueError
        ``block`` has no fixed point for another reason: a moving or scheduled
        ``rho``, adaptive or relaxed step sizes, or an
        :class:`~bartorch.priors.ImplicitPrior` with a ``sigma`` schedule or
        ``step=True``.

    Examples
    --------
    >>> deq = optim.FixedPoint(optim.PRIDUBlock(priors.ImplicitPrior(net, 0.05)), tol=1e-4)
    >>> deq(kspace, A).abs().sub(target).square().sum().backward()
    """

    def __init__(
        self,
        block: nn.Module,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        backward_iter: int | None = None,
        backward_tol: float | None = None,
    ):
        super().__init__()
        _stationary(block)
        self.block = block
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.backward_iter = self.max_iter if backward_iter is None else int(backward_iter)
        self.backward_tol = self.tol if backward_tol is None else float(backward_tol)
        self.iterations = 0

    def forward(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Iterate the block from ``block.start(y, A, x0)`` and return the image at the fixed point.

        The result is recorded for autograd only when ``y`` requires a gradient or
        the block has trainable parameters.

        Parameters
        ----------
        y : torch.Tensor
            Data.
        A : LinearOperator
            Encoding operator.
        x0 : torch.Tensor, default=None
            Starting image; zero when omitted.

        Returns
        -------
        torch.Tensor
            The image at the fixed point.
        """
        from bartorch.linop._base import _tracking

        block = self.block
        state = block.start(y, A, x0)
        names = _moving(state)

        with torch.no_grad():
            z = _pack(state, names)
            for self.iterations in range(1, self.max_iter + 1):
                state = block(state, A)
                moved = _pack(state, names)
                done = torch.linalg.vector_norm(moved - z) <= self.tol * torch.linalg.vector_norm(
                    moved
                )
                z = moved
                if done:
                    break

        learned = any(p.requires_grad for p in block.parameters())
        if not (torch.is_grad_enabled() and (_tracking(y) or learned)):
            return block.output(state, A)

        def step(v):
            return _pack(block(_unpack(state, names, v), A), names)

        point = step(z.detach())
        start = point.detach().requires_grad_()
        again = step(start)

        def implicit(grad):
            g = grad
            for _ in range(self.backward_iter):
                (vjp,) = torch.autograd.grad(again, start, g, retain_graph=True)
                moved = vjp + grad
                size = torch.linalg.vector_norm(moved)
                if torch.linalg.vector_norm(moved - g) <= self.backward_tol * size:
                    return moved
                g = moved
            return g

        point.register_hook(implicit)
        return block.output(_unpack(state, names, point), A)
