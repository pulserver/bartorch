r"""
================================
10. Plug-and-play reconstruction
================================

In lesson 9 the prior knowledge was a signal model; in lesson 5 it was an
explicit penalty, a wavelet or total-variation term with a weight. In this
lesson the prior is a pretrained denoising network, used in place of the
penalty inside the same ADMM and FISTA iterations.

ADMM and FISTA use the penalty :math:`g` only through its proximal operator,

.. math::

   \operatorname{prox}_{\gamma g}(v) = \arg\min_x \; \tfrac12 \|x - v\|_2^2
   + \gamma\, g(x).

This is the maximum a posteriori estimate of an image observed as
:math:`v` in white Gaussian noise, under the prior :math:`\exp(-g)`: the
proximal operator is a denoiser. Plug-and-play (PnP) regularization
[#venkatakrishnan]_ [#ahmad]_ replaces it by an image denoiser
:math:`D_\sigma`, without an explicit :math:`g`. Each iteration alternates a
data-consistency step on the measured k-space and a denoising step, and the
denoiser's noise level :math:`\sigma` takes the role of the regularization
weight.

The denoiser here is DRUNet [#zhang]_, with the weights ``deepinv``
distributes. It was trained to remove Gaussian noise from grayscale
photographs, not from MR images.

**Learning objectives**

- Wrap a pretrained denoiser as :class:`bartorch.priors.ImplicitPrior` and
  hand it to :func:`bartorch.optim.admm` and :func:`bartorch.optim.fista` in
  place of a :mod:`bartorch.priors` term.
- Compare the result with total variation on the same data, in the images,
  the error maps and an enlarged region.
- Change the denoiser's noise level and recognise under- and
  over-regularization.

Previous: :doc:`../05-model-based/01-quantitative-models`. Next:
:doc:`02-modl-with-admm`, where you train the denoiser through the iteration
instead of using a pretrained one.

The pretrained weights, about 125 MB, are downloaded the first time you run
this. The network runs once per iteration, and on a CPU that is most of the
run time.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

WIDTH = 7.8  # inches, the width of the documentation column


def panels(columns, rows=1, width=WIDTH, bar=False):
    """A row (or grid) of frameless square image panels, with room for a
    colorbar if ``bar``."""
    side = (width - (0.9 if bar else 0.0)) / columns
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(width, rows * side + 0.5))
    for axis in axes.flat:
        axis.set_axis_off()
    return figure, axes


def show(axis, values, title=None, vmax=None, cmap="gray", vmin=0.0):
    """One panel, of a magnitude by default."""
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


def outline(axis, crop):
    """Mark the region ``crop`` that a later figure enlarges."""
    rows, cols = crop
    axis.add_patch(
        Rectangle(
            (cols.start - 0.5, rows.start - 0.5),
            cols.stop - cols.start,
            rows.stop - rows.start,
            fill=False,
            edgecolor="#e8a33d",
            linewidth=1.5,
        )
    )


def errors(figure, axes, estimates, reference, scale):
    """|estimate - reference| relative to the reference's peak, on one scale."""
    peak = float(reference.abs().max())
    for axis, estimate in zip(axes, estimates):
        difference = (scaled(estimate, reference) - reference.abs()).abs() / peak
        handle = show(axis, difference, cmap="magma", vmax=scale)
    return figure.colorbar(handle, ax=axes, fraction=0.046, label="|error| / peak")


def scaled(estimate, reference):
    """``estimate`` scaled to ``reference`` in the least-squares sense."""
    a, b = estimate.abs().double(), reference.abs().double()
    return (float((a * b).sum() / (a * a).sum()) * a).float()


def report(results):
    """NRMSE and SSIM of each magnitude against the phantom's."""
    for name, estimate in results.items():
        error = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
        similarity = bt.ssim(image.abs(), scaled(estimate, image))
        print(f"{name:>16}  NRMSE {error:.3f}  SSIM {similarity:.3f}")


def simulate(acceleration, calibration, seed=11):
    """A variable-density pattern of phase encodes, with a full ACS block."""
    encodes = torch.arange(SIZE) - SIZE // 2
    centre = (encodes.abs() < calibration // 2).to(torch.float32)
    drawn = torch.multinomial(
        (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0 * (1.0 - centre),
        SIZE // acceleration - calibration,
        replacement=False,
        generator=torch.Generator().manual_seed(seed),
    )
    lines = centre.clone()
    lines[drawn] = 1.0
    return lines.reshape(SIZE, 1).to(torch.complex64)


# The phantom: a BrainWeb [#brainweb]_ segmentation turned into a
# T1-weighted spin-echo image, with a smooth phase.
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

import torch

SIZE = 128
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)
MARGIN = 0.25

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
fractions = np.flipud(get_mri(sub_id=0, contrast="fuzzy")[SLICE])[..., list(TISSUES)].copy()

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
signal = (
    proton_density
    * (1 - torch.exp(-600.0 / T1.clamp(min=1e-3)))
    * torch.exp(-12.0 / T2.clamp(min=1e-3))
)
signal = torch.where(T1 > 0, signal, torch.zeros(()))
signal = signal / signal.max()
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)
peak = float(image.abs().max())
# Occipital cortex and the posterior horns of the lateral ventricles.
zoom = (slice(70, 118), slice(40, 88))
# sphinx_gallery_end_ignore

# %%
# Your scan
# ---------
#
# The object is the BrainWeb slice [#brainweb]_ of lessons 2 and 5, a
# :math:`T_1`-weighted spin-echo image, here at 128 by 128; the code that
# builds it is hidden. You acquire a quarter of the phase encodes
# (:math:`R = 4`), drawn at random around a 16-line ACS block, with eight
# channels of BART's analytical head coil and complex Gaussian noise of
# variance :math:`10^{-3}`. At that acceleration and noise level, both noise
# amplification and incoherent aliasing limit a reconstruction without a
# prior.
#
# The SENSE operator uses the sensitivities the data were simulated with, so
# that the reconstructions below differ only in the prior. Lesson 3 is about
# estimating them.
import torch

import bartorch
import bartorch.tools as bt

from bartorch import linop, optim, priors

COILS = 8
ACCELERATION = 4
CALIBRATION = 16

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities /= bartorch.rss(sensitivities, axes=(0,), keepdim=True)
pattern = simulate(ACCELERATION, CALIBRATION)  # (y, 1)

A = linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=pattern)
data = bt.noise(A(image), n=1e-3, s=42) * pattern
zero_filled = A.H(data)

# sphinx_gallery_start_ignore
figure, axes = panels(3)
show(axes[0, 0], image, "the object", vmax=peak)
outline(axes[0, 0], zoom)
show(axes[0, 1], pattern.real.expand(SIZE, SIZE), "sampling pattern", vmax=1.0)
show(axes[0, 2], scaled(zero_filled, image), "zero-filled", vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The zero-filled image, :math:`A^H y`, carries the incoherent aliasing of
# the random sampling, spread over the whole field of view, on top of the
# thermal noise. The orange box marks the region enlarged below.
#
# An explicit penalty
# -------------------
#
# The reference is total variation under ADMM, as in lesson 5. Of the
# weights 0.002, 0.005, 0.01 and 0.02, 0.005 gives the lowest NRMSE against
# the phantom and 0.02 the highest SSIM; 0.01 is within 0.001 of that NRMSE
# with an SSIM close to the highest, and is used here.
total_variation = optim.admm(
    data,
    A,
    priors.TotalVariation((-1, -2), 0.01),
    maxiter=60,
    rho=0.1,
)

# sphinx_gallery_start_ignore
figure, axes = panels(2, width=0.75 * WIDTH, bar=True)
show(axes[0, 0], scaled(total_variation, image), "total variation", vmax=peak)
errors(figure, axes[0, 1:], [total_variation], image, 0.15)
axes[0, 1].set_title("its error")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Total variation removes most of the aliasing and the noise. At this weight
# it still leaves a blotchy texture across the brain and flattens gradual
# intensity changes into patches, the piecewise-constant images that total
# variation favours.
#
# A denoiser in its place
# -----------------------
#
# The network takes real planes of shape ``(n, channels, y, x)``; your image
# is complex. :class:`~bartorch.priors.ImplicitPrior` bridges the two:
# ``spatial=2`` says the network works on 2D planes, and around each call the
# prior scales the image to unit peak modulus, denoises its real and imaginary
# parts as two grayscale planes, and scales the result back. So ``sigma`` is
# in units of the image's peak.
#
# Nothing is trained here, so you evaluate the network without gradients.
# You run the same prior under ADMM and under FISTA.
from deepinv.models import DRUNet

ITERATIONS = 30

denoiser = DRUNet(in_channels=1, out_channels=1, pretrained="download")
denoiser = denoiser.eval()
prior = priors.ImplicitPrior(denoiser, sigma=0.05, spatial=2)

with torch.no_grad():
    admm = optim.admm(data, A, prior, maxiter=ITERATIONS, rho=0.2)
    fista = optim.fista(data, A, prior, maxiter=ITERATIONS)

# sphinx_gallery_start_ignore
reconstructions = {
    "zero-filled": zero_filled,
    "total variation": total_variation,
    "DRUNet, ADMM": admm,
    "DRUNet, FISTA": fista,
}
compared = ("total variation", "DRUNet, ADMM", "DRUNet, FISTA")
figure, axes = panels(3, bar=True)
errors(figure, axes[0], [reconstructions[name] for name in compared], image, 0.15)
for axis, name in zip(axes[0], compared):
    axis.set_title(f"{name} error")
plt.show()

figure, axes = panels(4)
show(axes[0, 0], image.abs()[zoom], "truth", vmax=peak)
for axis, name in zip(axes[0, 1:], compared):
    show(axis, scaled(reconstructions[name], image)[zoom], name, vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The error maps are on one scale. Under ADMM the denoiser removes the noise
# and the aliasing and keeps the tissue boundaries, and its error map is
# darker inside the brain than the total-variation one. In the enlarged region
# the ventricles and the larger cortical folds are there, though the finest
# sulci are lost. Under FISTA, at the same :math:`\sigma`, the same denoiser
# gives a much smoother image: the cortical folds are gone, and its error runs
# along every tissue boundary.
#
# The two iterations differ in the step before the denoiser. FISTA takes a
# gradient step of fixed length on the data term; ADMM solves a least-squares
# problem that holds the image to the data with weight :math:`\rho`. The
# denoiser is therefore applied to different images, and the same
# :math:`\sigma` has a different effective strength. With a
# fixed :math:`\sigma` neither iteration minimises a known objective, so the
# iteration count and :math:`\rho` are part of the result, and you choose
# them as you would choose the weight of a penalty.
#
# The numbers:
report(reconstructions)

# %%
# SSIM ranks the over-smoothed FISTA image above the ADMM one. One figure of
# merit does not replace looking at the images.
#
# Your turn: the noise level
# --------------------------
#
# :math:`\sigma` is the strength of the prior. A :math:`\sigma` below the
# noise level of the iterate leaves noise and aliasing in place; one above it
# removes image detail as well. Run ADMM at a smaller and a larger
# :math:`\sigma` and compare with 0.05:
levels = (0.02, 0.05, 0.12)
sweep = {0.05: admm}
with torch.no_grad():
    for sigma in (0.02, 0.12):
        sweep[sigma] = optim.admm(
            data,
            A,
            priors.ImplicitPrior(denoiser, sigma=sigma, spatial=2),
            maxiter=ITERATIONS,
            rho=0.2,
        )

report({f"sigma {sigma:.2f}": sweep[sigma] for sigma in levels})

# sphinx_gallery_start_ignore
figure, axes = panels(3)
for axis, sigma, label in zip(axes[0], levels, ("too small", "", "too large")):
    title = f"$\\sigma$ = {sigma}" + (f" ({label})" if label else "")
    show(axis, scaled(sweep[sigma], image)[zoom], title, vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
# At the smallest :math:`\sigma` the noise and the aliasing remain as a
# mottled texture; at the largest the cortex is smoothed into uniform white
# matter and small structures disappear. The printed errors have their
# minimum at 0.05. You can repeat the sweep under FISTA and compare where
# its minimum lies.
#
# DRUNet was trained neither on MR images nor on the residual aliasing an
# undersampled scan leaves, which is not white Gaussian noise. In lesson 11
# you train a network for exactly that, inside the iteration.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Reconstruct a 2D Cartesian SENSE acquisition (128x128, 8 coils, R=4,
#    16-line ACS, known sensitivities) with bartorch, using a pretrained
#    denoiser as the prior. Build A with linop.CartesianSense. Load deepinv's
#    DRUNet (1 channel, pretrained weights) in eval mode and wrap it as
#    priors.ImplicitPrior(denoiser, sigma=0.05, spatial=2). Under
#    torch.no_grad, run optim.admm (30 iterations, rho=0.2) and optim.fista
#    (30 iterations) with it. Compare against optim.admm with
#    priors.TotalVariation over the last two axes (weight 0.01, 60
#    iterations, rho=0.1). Report NRMSE and SSIM of the magnitudes after a
#    global least-squares scale, and repeat ADMM at sigma 0.02 and 0.12.

# %%
#
# References
# ----------
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
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
