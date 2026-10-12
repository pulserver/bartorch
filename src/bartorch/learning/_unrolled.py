"""Unrolled networks formed by repeated application of an iteration block."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ["Unrolled"]


def _fields(state) -> list[str]:
    """Names of the state fields holding a tensor, or a non-empty tuple of tensors."""
    names = []
    for field in dataclasses.fields(state):
        value = getattr(state, field.name)
        if isinstance(value, torch.Tensor) or (
            isinstance(value, tuple)
            and 0 != len(value)
            and all(isinstance(t, torch.Tensor) for t in value)
        ):
            names.append(field.name)
    return names


def _tensors(state, names) -> list[torch.Tensor]:
    made: list[torch.Tensor] = []
    for name in names:
        value = getattr(state, name)
        made.extend([value] if isinstance(value, torch.Tensor) else list(value))
    return made


def _restored(state, names, values):
    """``state`` with its tensor fields replaced by ``values``, in ``_tensors`` order."""
    fields, at = {}, 0
    for name in names:
        value = getattr(state, name)
        count = 1 if isinstance(value, torch.Tensor) else len(value)
        taken = list(values[at : at + count])
        at += count
        fields[name] = taken[0] if isinstance(value, torch.Tensor) else tuple(taken)
    return dataclasses.replace(state, **fields)


class Unrolled(nn.Module):
    """Unrolled network: an iteration block applied a fixed number of times.

    With every parameter frozen the stack reproduces the corresponding solver
    bit for bit; ``requires_grad_()`` on a step size, a penalty weight or a
    denoiser's weights makes it trainable.  One block shared by every iteration
    shares the weights; a sequence of blocks gives each iteration its own.
    MoDL is an :class:`~bartorch.optim.ADMMBlock` whose single term is an
    :class:`~bartorch.priors.ImplicitPrior`.

    ``detach`` and ``checkpoint`` select the backward pass without changing the
    value computed:

    * by default the whole stack is recorded, and memory grows with the number
      of iterations;
    * ``detach=True`` starts each iteration from a detached state; with a loss
      on each image :meth:`steps` yields, this is greedy per-iteration
      training, with memory independent of the number of iterations;
    * ``checkpoint=True`` keeps the states between iterations and recomputes
      each step in the backward pass, giving the end-to-end gradient at the
      cost of a second application of every block.

    :class:`bartorch.optim.FixedPoint` is the alternative without an iteration
    count, differentiated at the fixed point.
    :doc:`/explanation/learned-reconstruction` compares the strategies.

    Parameters
    ----------
    block : nn.Module or sequence of nn.Module
        One of :mod:`bartorch.optim`'s iteration blocks, or one per iteration.
    iterations : int, default=None
        Number of applications of a shared block; the sequence's length for a
        sequence.
    detach : bool, default=False
        Start each iteration from a detached state.
    checkpoint : bool, default=False
        Recompute each iteration in the backward pass rather than store it.

    Raises
    ------
    ValueError
        A shared block without ``iterations``, an empty sequence, an
        ``iterations`` that differs from the sequence's length, or fewer than
        one iteration.

    Notes
    -----
    A checkpointed step is correct only if its recomputation reproduces it.
    PyTorch's random state is restored for the recomputation, so dropout is
    reproduced; a BART term that draws random shifts from BART's generator,
    such as :class:`~bartorch.priors.Wavelet` or
    :class:`~bartorch.priors.LocallyLowRank` with ``randshift=True``, is not,
    and the gradient then does not belong to the forward pass.  This is not
    checked.

    Examples
    --------
    >>> block = optim.ADMMBlock(priors.ImplicitPrior(denoiser), rho=0.05, cg_maxiter=10)
    >>> block.rho.requires_grad_()
    >>> model = learning.Unrolled(block, iterations=10, checkpoint=True)
    >>> model(kspace, A).abs().sub(target).square().mean().backward()
    """

    def __init__(
        self,
        block,
        iterations: int | None = None,
        *,
        detach: bool = False,
        checkpoint: bool = False,
    ):
        super().__init__()
        shared = isinstance(block, nn.Module) and not isinstance(
            block, (nn.ModuleList, nn.Sequential)
        )
        if shared:
            if iterations is None:
                raise ValueError("one block is shared by every iteration, so say how many")
            self.blocks = nn.ModuleList([block])
            self.iterations = int(iterations)
        else:
            self.blocks = nn.ModuleList(block)
            if 0 == len(self.blocks):
                raise ValueError("a stack of no blocks computes nothing")
            if iterations is not None and int(iterations) != len(self.blocks):
                raise ValueError(
                    f"a block per iteration is {len(self.blocks)} of them, not {int(iterations)}"
                )
            self.iterations = len(self.blocks)
        if self.iterations < 1:
            raise ValueError(f"a stack runs at least one iteration, not {self.iterations}")
        self.detach = bool(detach)
        self.checkpoint = bool(checkpoint)

    def block(self, k: int) -> nn.Module:
        """The block applied at iteration ``k``."""
        return self.blocks[k % len(self.blocks)]

    def steps(self, y: torch.Tensor, A, x0: torch.Tensor | None = None):
        """Yield the image after each iteration.

        Takes the arguments of :meth:`forward`.  A loss on each yielded image, with
        ``detach=True``, is greedy per-iteration training.
        """
        state = self.blocks[0].start(y, A, x0)
        names = _fields(state)
        for k in range(self.iterations):
            block = self.block(k)
            if self.detach:
                state = _restored(state, names, [t.detach() for t in _tensors(state, names)])
            state = self._step(block, state, A, names)
            yield block.output(state, A)

    def forward(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Run the iterations and return the last image.

        Parameters
        ----------
        y : torch.Tensor
            Data, with or without a leading batch axis.
        A : LinearOperator
            Encoding operator, shared by every item of a batch.
        x0 : torch.Tensor, default=None
            Starting image; zero when omitted, as in BART.

        Returns
        -------
        torch.Tensor
            The image after the last iteration.
        """
        image = None
        for image in self.steps(y, A, x0):
            pass
        return image

    def _step(self, block, state, A, names):
        """One application of ``block``, checkpointed where that was requested."""
        tensors = _tensors(state, names)
        learning = any(p.requires_grad for p in block.parameters())
        if not (
            self.checkpoint
            and torch.is_grad_enabled()
            and (learning or any(t.requires_grad for t in tensors))
        ):
            return block(state, A)

        made: dict = {}

        def run(*values):
            step = block(_restored(state, names, values), A)
            made["state"] = step
            return tuple(_tensors(step, names))

        out = checkpoint(run, *tensors, use_reentrant=False)
        return _restored(made["state"], names, out)
