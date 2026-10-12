"""Figures of the non-Cartesian transform page."""

import matplotlib.pyplot as plt
import numpy as np
import torch

import bartorch.tools as bt
from bartorch import _finufft, linop

N = 64
SPOKES = 48
TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _explicit_sum(traj, image):
    """The forward transform as a sum over voxels, one sample at a time."""
    k = traj.numpy().real
    x = np.arange(N) - N // 2
    phase = np.exp(
        -2j
        * np.pi
        * (k[..., 0, None, None] * x[None, None, None, :] + k[..., 1, None, None] * x[:, None])
        / N
    )
    return (phase * image.numpy().reshape(N, N)).sum(axis=(-1, -2)) / N


def _errors(traj, image, reference):
    def error():
        got = linop.NUFFT(traj, (1, N, N), toeplitz=False)(image).numpy()
        return np.linalg.norm(got.reshape(reference.shape) - reference) / np.linalg.norm(reference)

    curves = {}
    try:
        for upsampling in (1.25, 2.0):
            curves[upsampling] = []
            for tolerance in TOLERANCES:
                _finufft.use_in_tools(True, tolerance=tolerance, upsampling=upsampling)
                curves[upsampling].append(error())
    finally:
        _finufft.use_in_tools(True)
    with _finufft.barts_own_gridder():
        bart = error()
    return curves, bart


def accuracy():
    traj = bt.traj(x=N, y=SPOKES, r=True)
    image = bt.phantom([N, N]).reshape(1, N, N)
    curves, bart = _errors(traj, image, _explicit_sum(traj, image))

    psf_size = 128
    psf_traj = bt.traj(x=psf_size, y=32, r=True)
    A = linop.NUFFT(psf_traj, (1, psf_size, psf_size))
    psf = A.adjoint(torch.ones(A.oshape, dtype=torch.complex64)).abs()[0]
    psf = (psf / psf.max()).clamp_min(1e-4).log10()

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    shown = left.imshow(psf, cmap="gray", vmin=-3, vmax=0)
    left.set_axis_off()
    left.set_title("PSF, 32 spokes")
    bar = figure.colorbar(shown, ax=left, fraction=0.046, pad=0.04)
    bar.set_label(r"$\log_{10}|h|$")

    for upsampling, errors in curves.items():
        right.loglog(TOLERANCES, errors, "o-", label=f"FINUFFT, $\\sigma$ = {upsampling:g}")
    right.axhline(bart, linestyle="--", color="C2", label="BART's gridder")
    right.loglog(TOLERANCES, TOLERANCES, ":", color="C7", label="error = tolerance")
    right.invert_xaxis()
    right.set_xlabel("tolerance")
    right.set_ylabel("error against the sum")
    right.set_title(f"{N}$^2$, {SPOKES} spokes")
    right.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    return figure


FIGURES = {"nufft-accuracy.svg": accuracy}
