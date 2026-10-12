"""Figures of the model-based reconstruction page."""

import matplotlib.pyplot as plt
import torch

from bartorch import nlop, optim

ECHOES = [10.0 * (echo + 1) for echo in range(8)]
T2 = torch.linspace(30.0, 200.0, 16)
STEPS = 16


def _irgnm():
    """A T2 decay fitted voxel by voxel by Gauss-Newton, each iterate kept."""
    torch.manual_seed(0)
    voxels = (4, T2.numel())
    truth = T2.expand(voxels)
    M = nlop.MultiEcho(ECHOES, voxels)
    clean = M(M.initial(T2=truth, amplitude=torch.ones(voxels)))
    data = clean + 0.02 * torch.randn_like(clean)

    block = nlop.IRGNMBlock(alpha=1.0, redu=2.0, inner=optim.CG(maxiter=50))
    state = block.start(data, M, x0=M.initial(T2=80.0))
    errors, fits = [], {}
    for k in range(1, STEPS + 1):
        state = block(state, M)
        maps = M.split(block.output(state, M))
        errors.append(float(((maps["T2"] - truth).abs() / truth).median()))
        if k in (2, 6, STEPS):
            fits[k] = maps
    return data, fits, errors


def convergence():
    data, fits, errors = _irgnm()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    voxel = (0, 12)
    t = torch.linspace(0.0, max(ECHOES), 200)
    left.plot(ECHOES, data[(slice(None), *voxel)].real, "o", label="data")
    for k, maps in fits.items():
        amplitude = maps["amplitude"][voxel].real
        decay = amplitude * torch.exp(-t / maps["T2"][voxel])
        left.plot(t, decay, label=f"step {k}")
    left.set_xlabel("TE (ms)")
    left.set_ylabel("signal (a.u.)")
    left.set_title(f"one voxel, $T_2$ = {float(T2[voxel[1]]):.0f} ms")

    steps = range(1, STEPS + 1)
    right.semilogy(steps, errors, "o-", color="C4", label="median $T_2$ error")
    weights = [0.5**k for k in range(STEPS)]
    right.semilogy(steps, weights, "--", color="C5", label=r"weight $\alpha_k$")
    right.set_xlabel("Gauss-Newton step $k$")
    right.set_title("convergence")
    handles = [*left.get_legend_handles_labels()[0], *right.get_legend_handles_labels()[0]]
    labels = [*left.get_legend_handles_labels()[1], *right.get_legend_handles_labels()[1]]
    right.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    return figure


FIGURES = {"model-irgnm.svg": convergence}
