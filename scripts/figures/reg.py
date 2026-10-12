"""The proximal operator of an l1 penalty, on the Regularizers page."""

import matplotlib.pyplot as plt
import torch

import bartorch
from bartorch import priors

N = 256
NOISE = 0.15
WEIGHT = 0.3


def _signal():
    clean = torch.zeros(N)
    clean[40:90] = 1.0
    clean[90:150] = 0.4
    clean[180:220] = -0.7
    noise = NOISE * torch.randn(N, generator=torch.Generator().manual_seed(0))
    return clean.to(torch.complex64), (clean + noise).to(torch.complex64)


def proximal_operator():
    figure = plt.figure(figsize=(7.0, 5.6), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=(1.0, 0.9))
    curves, coefficients = top.subplots(1, 2, width_ratios=(1.0, 1.3))
    signal = bottom.subplots()

    v = torch.linspace(-3.0, 3.0, 601)
    curves.plot(v, v, color="0.5", lw=1.0, ls=":", label="identity")
    for weight in (0.5, 1.0, 2.0):
        out = bartorch.soft_thresh(weight, v.to(torch.complex64)).real
        curves.plot(v, out, lw=1.8, label=rf"$\tau\lambda$ = {weight:g}")
    curves.set_xlabel("input $v$")
    curves.set_ylabel("output")
    curves.set_aspect("equal")
    curves.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=9)

    clean, noisy = _signal()
    before = bartorch.fwt(noisy, -1, wavelet="haar")
    after = bartorch.soft_thresh(WEIGHT, before)
    index = torch.arange(N)
    coefficients.plot(index, before.real, ".", ms=3, color="C0", label="noisy")
    kept = after.abs() > 0
    coefficients.plot(
        index[kept],
        after.real[kept],
        "o",
        ms=4,
        mfc="none",
        color="C1",
        label=f"thresholded ({int(kept.sum())} kept)",
    )
    coefficients.axhspan(-WEIGHT, WEIGHT, color="0.5", alpha=0.2, lw=0)
    coefficients.set_xlabel("Haar coefficient index")
    coefficients.set_ylabel("value")
    coefficients.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=9)

    denoised = priors.Wavelet((-1,), WEIGHT, family="haar", randshift=False).prox(noisy)
    signal.plot(index, noisy.real, color="C0", lw=0.8, label="noisy")
    signal.plot(index, denoised.real, color="C1", lw=1.8, label="prox")
    signal.plot(index, clean.real, color="0.5", lw=1.0, ls="--", label="noiseless")
    signal.set_xlabel("sample")
    signal.set_ylabel("signal")
    signal.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return figure


FIGURES = {"reg-proximal-operator.svg": proximal_operator}
