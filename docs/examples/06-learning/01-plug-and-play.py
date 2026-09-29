r"""
=======================
Plug-and-play denoisers
=======================

A pretrained image denoiser used as the proximal step of BART's iterations, in
place of a specified regularization term, for an undersampled Cartesian SENSE
acquisition.

A proximal iteration such as ADMM or FISTA applies the regularization term
only through its proximal operator,

.. math::

   \operatorname{prox}_{\gamma g}(v) = \arg\min_x \; \tfrac12 \|x - v\|_2^2 + \gamma\, g(x),

which is the maximum a posteriori estimate of an image observed in white
Gaussian noise under the prior :math:`\exp(-g)`. Plug-and-play regularization
[#venkatakrishnan]_ [#ahmad]_ replaces this operator by an image denoiser
:math:`D_\sigma`, without writing down :math:`g`. The denoiser's noise level
:math:`\sigma` takes the role of the regularization weight, and the penalty
parameter :math:`\rho` of ADMM sets how far each x-update may move from the
denoised image towards data consistency.

The denoiser here is DRUNet [#zhang]_ with the weights distributed by
``deepinv``, trained for Gaussian denoising of natural grayscale images and not
on MR images. :class:`bartorch.priors.ImplicitPrior` converts between the
complex image of the reconstruction and the real planes the network takes; the
iterations are :func:`bartorch.optim.admm` and :func:`bartorch.optim.fista`,
unchanged.

The phantom is the BrainWeb slice of
:doc:`../03-regularization/01-regularized-reconstruction`; the cell that builds
it is hidden on this page and present in the downloadable script.

**Learning objectives**

- Wrap a pretrained denoiser as :class:`bartorch.priors.ImplicitPrior` and
  pass it to :func:`bartorch.optim.admm` and :func:`bartorch.optim.fista` in
  place of a :mod:`bartorch.priors` term.
- Compare the result with total-variation regularization on the same data.
- Vary the denoiser's noise level and relate it to the regularization weight.

It follows :doc:`../05-model-based/02-quantitative-models`. The next lesson,
:doc:`02-modl-with-admm`, trains the denoiser through the iteration.

The pretrained weights, about 125 MB, are downloaded on the first call.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 11,
        "figure.constrained_layout.use": True,
    }
)

PAGE_WIDTH = 8.0  # inches, the width of the documentation column


def panels(rows, columns, height=1.0):
    """A grid of square image panels filling the documentation column."""
    side = PAGE_WIDTH / columns
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(PAGE_WIDTH, rows * side * height + 0.4)
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    return figure, axes


def show(axis, values, title=None, vmax=None, cmap="gray", vmin=0.0):
    """One panel, of a magnitude by default."""
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        axis.set_title(title, fontsize=10)
    return handle


def scaled(estimate, reference):
    """``estimate`` scaled to ``reference`` in the least-squares sense."""
    a, b = estimate.abs().double(), reference.abs().double()
    return (float((a * b).sum() / (a * a).sum()) * a).float()


# sphinx_gallery_end_ignore
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri
from deepinv.models import DRUNet

import bartorch
import bartorch.tools as bt
from bartorch import linop, optim, priors

SIZE = 128
COILS = 8
ACCELERATION = 4
CALIBRATION = 16

# sphinx_gallery_start_ignore
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times
MARGIN = 0.25  # what the field of view leaves around the head

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]

# BrainWeb's volume is indexed (inferior-superior, posterior-anterior,
# left-right), so its first axis selects an axial slice; an image is drawn from
# its first row down, so flipping puts anterior at the top.
fractions = np.flipud(get_mri(sub_id=0, contrast="fuzzy")[SLICE])[..., list(TISSUES)].copy()

# A square field of view around the head, with a margin for the aliasing of an
# undersampled acquisition to fold into.
occupied = np.nonzero(fractions.sum(-1) > 0.5)
middle = [int((axis.min() + axis.max()) / 2) for axis in occupied]
half = int(round((1 + MARGIN) * max(axis.max() - axis.min() for axis in occupied) / 2))
source = tuple(
    slice(max(0, c - half), min(n, c + half)) for c, n in zip(middle, fractions.shape[:2])
)
box = np.zeros((2 * half, 2 * half, fractions.shape[-1]), dtype=np.float32)
box[tuple(slice(s.start - (c - half), s.stop - (c - half)) for s, c in zip(source, middle))] = (
    fractions[source]
)
memberships = torch.nn.functional.interpolate(
    torch.as_tensor(box).permute(2, 0, 1)[None],
    size=(SIZE, SIZE),
    mode="bilinear",
    align_corners=False,
)[0]

weights = memberships * torch.as_tensor(tissue_pd)[:, None, None]
share = weights.sum(0).clamp(min=1e-6)
T1 = (weights * torch.as_tensor(tissue_t1)[:, None, None]).sum(0) / share
T2 = (weights * torch.as_tensor(tissue_t2)[:, None, None]).sum(0) / share
proton_density = weights.sum(0) / weights.sum(0).max()

# A T1-weighted spin echo, at a repetition time of 600 ms and an echo time of
# 12 ms.
signal = (
    proton_density
    * (1 - torch.exp(-600.0 / T1.clamp(min=1e-3)))
    * torch.exp(-12.0 / T2.clamp(min=1e-3))
)
signal = torch.where(T1 > 0, signal, torch.zeros(()))
signal = signal / signal.max()

# A smooth quadratic phase, so that nothing depends on the image being real.
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)

# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

# %%
#
# Acquisition
# -----------
#
# A quarter of the phase encodes, drawn from a variable density around a fully
# sampled region of 16 lines, with complex Gaussian noise of variance
# :math:`10^{-3}` per sample of the unitary transform. The sensitivities are
# the ones the data was simulated with, so that the comparison below concerns
# the regularization alone; :doc:`../02-parallel-imaging/01-coil-calibration`
# compares their estimation.

encodes = torch.arange(SIZE) - SIZE // 2
centre = (encodes.abs() < CALIBRATION // 2).to(torch.float32)
drawn = torch.multinomial(
    (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0 * (1.0 - centre),
    SIZE // ACCELERATION - CALIBRATION,
    replacement=False,
    generator=torch.Generator().manual_seed(11),
)
lines = centre.clone()
lines[drawn] = 1.0
pattern = lines.reshape(SIZE, 1).to(torch.complex64)

A = linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=pattern)
data = bt.noise(A(image), n=1e-3, s=42) * pattern

# %%
#
# A specified regularizer
# -----------------------
#
# Total variation under ADMM, at the weight that minimizes the error against
# the phantom among 0.002, 0.005, 0.01 and 0.02, is the reference point.

total_variation = optim.admm(data, A, priors.TotalVariation((-1, -2), 0.01), maxiter=60, rho=0.1)

# %%
#
# A denoiser as the proximal step
# -------------------------------
#
# ``spatial=2`` states that the network operates on two-dimensional planes of
# shape ``(n, channels, y, x)``. :class:`~bartorch.priors.ImplicitPrior` scales
# each image to unit peak modulus, denoises its real and imaginary parts as
# two grayscale planes, and scales the result back, so ``sigma`` is in units of
# the image's peak. The network is evaluated without gradients, since nothing
# is trained here.

denoiser = DRUNet(in_channels=1, out_channels=1, pretrained="download").eval()
prior = priors.ImplicitPrior(denoiser, sigma=0.05, spatial=2)

with torch.no_grad():
    admm = optim.admm(data, A, prior, maxiter=40, rho=0.2)
    fista = optim.fista(data, A, prior, maxiter=40)

reconstructions = {
    "zero-filled": A.H(data),
    "total variation": total_variation,
    "DRUNet, ADMM": admm,
    "DRUNet, FISTA": fista,
}

for name, estimate in reconstructions.items():
    error = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
    similarity = bt.ssim(image.abs(), scaled(estimate, image))
    print(f"{name:>16}  NRMSE {error:.3f}  SSIM {similarity:.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, 5, height=1.1)
peak = float(image.abs().max())
show(axes[0, 0], image, "phantom", vmax=peak)
axes[1, 0].axis("off")
for column, (name, estimate) in enumerate(reconstructions.items(), start=1):
    show(axes[0, column], scaled(estimate, image), name, vmax=peak)
    show(axes[1, column], (scaled(estimate, image) - image.abs()).abs(), vmax=0.2 * peak)
axes[1, 1].set_ylabel("|error|, x5")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Both plug-and-play reconstructions remove the noise and the incoherent
# aliasing that total variation leaves at this weight. They differ in the
# step: FISTA applies the denoiser after a gradient step of fixed length on the
# data term, ADMM after solving a quadratic problem that holds the image to
# the data with weight :math:`\rho`. A fixed :math:`\sigma` makes neither
# iteration the minimization of a known objective, so the number of iterations
# and :math:`\rho` enter the result, and are parameters to be chosen like the
# weight of a specified term.
#
# The noise level
# ---------------
#
# :math:`\sigma` is the strength of the prior. A denoiser asked for less noise
# than the iterate contains leaves the residual noise and aliasing in place;
# one asked for more removes image detail with them.

levels = (0.02, 0.05, 0.1)
with torch.no_grad():
    sweep = {
        sigma: optim.admm(
            data, A, priors.ImplicitPrior(denoiser, sigma=sigma, spatial=2), maxiter=40, rho=0.2
        )
        for sigma in levels
    }

for sigma, estimate in sweep.items():
    error = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
    similarity = bt.ssim(image.abs(), scaled(estimate, image))
    print(f"sigma {sigma:.2f}  NRMSE {error:.3f}  SSIM {similarity:.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 3)
for axis, (sigma, estimate) in zip(axes[0], sweep.items()):
    show(axis, scaled(estimate, image), f"$\\sigma$ = {sigma}", vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The denoiser was not trained on MR images, nor for the residual aliasing an
# undersampled acquisition leaves, which is not white Gaussian noise. The next
# lesson, :doc:`02-modl-with-admm`, trains a network inside the iteration, on
# the acquisition it is applied to.

# %%
#
# References
# ----------
#
# .. [#venkatakrishnan] Venkatakrishnan SV, Bouman CA, Wohlberg B. Plug-and-play
#    priors for model based reconstruction. *IEEE Global Conference on Signal and
#    Information Processing*, 945-948 (2013).
#    https://doi.org/10.1109/GlobalSIP.2013.6737048
#
# .. [#ahmad] Ahmad R, Bouman CA, Buzzard GT, Chan S, Liu S, Reehorst ET,
#    Schniter P. Plug-and-play methods for magnetic resonance imaging: using
#    denoisers for image recovery. *IEEE Signal Process Mag* 37(1):105-116
#    (2020). https://doi.org/10.1109/MSP.2019.2949470
#
# .. [#zhang] Zhang K, Li Y, Zuo W, Zhang L, Van Gool L, Timofte R. Plug-and-play
#    image restoration with deep denoiser prior. *IEEE Trans Pattern Anal Mach
#    Intell* 44(10):6360-6376 (2022). https://doi.org/10.1109/TPAMI.2021.3088914
