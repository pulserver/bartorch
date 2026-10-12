"""Figures of the learned reconstruction page."""

import gc

import matplotlib.pyplot as plt
import torch

from bartorch import learning, linop, optim, priors

SHAPE = (48, 48)
ITERATIONS = (1, 2, 4, 6, 8)


class _Held:
    """A tensor autograd saved for the backward pass, counted while it is alive.

    Kept detached, so that holding it creates no reference cycle through the
    graph and it is freed with the graph.
    """

    live = 0

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.detach()
        self.size = tensor.numel() * tensor.element_size()
        _Held.live += self.size

    def __del__(self):
        _Held.live -= self.size


def _held(reconstruct) -> float:
    """MiB held for the backward pass once ``reconstruct`` has returned."""
    _Held.live = 0
    with torch.autograd.graph.saved_tensors_hooks(_Held, lambda held: held.tensor):
        image = reconstruct()
    gc.collect()
    made = _Held.live / 2**20
    del image
    return made


def _memory():
    torch.manual_seed(0)
    mask = (torch.rand(SHAPE) < 0.4).to(torch.complex64)
    A = linop.Diagonal(mask, SHAPE) @ linop.FFT(SHAPE, axes=(-1, -2))
    y = A(torch.randn(SHAPE, dtype=torch.complex64))
    net = learning.UNet(1, spatial=2, widths=(16, 32))

    def block():
        return optim.ISTBlock(priors.ImplicitPrior(net, spatial=2, channels=1))

    curves = {}
    for label, keywords in (
        ("end to end", {}),
        ("detach=True", {"detach": True}),
        ("checkpoint=True", {"checkpoint": True}),
    ):
        curves[label] = [
            _held(lambda n=n: learning.Unrolled(block(), iterations=n, **keywords)(y, A))
            for n in ITERATIONS
        ]
    curves["FixedPoint"] = [
        _held(lambda n=n: optim.FixedPoint(block(), max_iter=n, tol=0.0)(y, A)) for n in ITERATIONS
    ]
    return curves


def memory():
    curves = _memory()
    figure, axes = plt.subplots(figsize=(6.0, 3.0))
    for label, values in curves.items():
        axes.plot(ITERATIONS, values, "o-", label=label)
    axes.set_ylim(bottom=0)
    axes.set_xlabel("iterations")
    axes.set_ylabel("held for backward (MiB)")
    axes.set_title("U-Net prior, 48 × 48 image")
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    return figure


FIGURES = {"learned-memory.svg": memory}
