"""FISTA, IST and ADMM on one wavelet-regularized SENSE problem, on the Solvers page."""

import matplotlib.pyplot as plt
import torch

import bartorch
import bartorch.tools as bt
from bartorch import linop, optim, priors

SIZE = 64
COILS = 4
WEIGHT = 0.003
BUDGET = 120  # applications of the normal operator


def _problem():
    image = bt.phantom(SIZE)
    maps = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
    maps = maps / bartorch.rss(maps, axes=(0,), keepdim=True)
    generator = torch.Generator().manual_seed(3)
    lines = (torch.rand(SIZE, generator=generator) < 0.3).float()
    lines[SIZE // 2 - 4 : SIZE // 2 + 4] = 1.0
    pattern = lines.reshape(SIZE, 1).expand(SIZE, SIZE).to(torch.complex64)
    A = linop.CartesianSense(maps, (SIZE, SIZE), pattern)
    y = A(image)
    noise = torch.randn(y.shape, dtype=torch.complex64, generator=generator)
    return A, y + 0.01 * float(y.abs().max()) * noise * pattern


def _trace(block, A, y, distance):
    """Distance to the minimizer against the normal-operator applications spent."""
    state = block.start(y, A)
    spent, error = [0], [1.0]
    for _ in range(BUDGET):
        if spent[-1] >= BUDGET:
            break
        state = block(state, A)
        spent.append(getattr(state, "invokes", state.k))
        error.append(distance(block.output(state, A)))
    return spent, error


def convergence():
    A, y = _problem()
    # Haar on even lengths is orthogonal, and without random shifts its
    # threshold is the exact proximal operator, so every solver has one minimizer.
    term = priors.Wavelet((-1, -2), WEIGHT, family="haar", randshift=False)
    minimizer = optim.FISTA(term, maxiter=500)(y, A)

    def distance(x):
        return float((x - minimizer).norm() / minimizer.norm())

    figure, axis = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    runs = [
        ("IST", optim.ISTBlock(term)),
        ("FISTA", optim.FISTABlock(term)),
        (r"ADMM, $\rho$ = 0.1", optim.ADMMBlock(term, rho=0.1)),
        (r"ADMM, $\rho$ = 0.5", optim.ADMMBlock(term, rho=0.5)),
    ]
    for label, block in runs:
        spent, error = _trace(block, A, y, distance)
        axis.semilogy(spent, error, lw=1.8, label=label)
    axis.set_xlim(0, BUDGET)
    axis.set_xlabel(r"applications of $A^HA$")
    axis.set_ylabel(r"$\|x_k - \hat{x}\| \,/\, \|\hat{x}\|$")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return figure


FIGURES = {"solvers-convergence.svg": convergence}
