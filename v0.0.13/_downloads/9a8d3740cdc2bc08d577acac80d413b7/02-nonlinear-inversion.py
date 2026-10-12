"""
=====================================================
4. Joint reconstruction of image and sensitivity maps
=====================================================

In lesson 3 you saw ESPIRiT fail with eight ACS lines while nonlinear
inversion kept working, because it fits the sensitivities to every acquired
sample rather than to the ACS block alone. In this lesson you keep only six
central lines, too few for any separate calibration, and reconstruct the
image and the sensitivities together.

In the SENSE model

.. math::

   y_c = P F (S_c \\cdot x)

both :math:`S_c` and :math:`x` become unknowns, and the model becomes
bilinear: the data are a product of the two. Nonlinear inversion (NLINV)
[#nlinv]_ solves it with the iteratively regularized Gauss-Newton method
(IRGNM) [#bakushinsky]_. The factorization into image and sensitivities is
not unique, and NLINV restricts it by building smoothness of the
sensitivities into the model. You first run it as BART's ``nlinv``,
then write the same reconstruction out as an operator and a solver you can
change.

**Learning objectives**

- Reconstruct an image and its sensitivities jointly with
  :func:`bartorch.tools.nlinv` from an ACS region too small for ESPIRiT.
- Explain the ambiguity of the bilinear model and how the Sobolev weighting
  of the sensitivities resolves it.
- Write the same reconstruction as :class:`bartorch.nlop.NonlinearSense`
  under :class:`bartorch.nlop.IRGNM`.
- See how the number of Gauss-Newton steps acts as a regularization
  parameter.

Previous: :doc:`01-coil-calibration`. Next:
:doc:`../03-regularization/01-regularized-reconstruction`, where you add a
prior on the image to the reconstruction and learn how to choose it.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from gallery_style import domain, phase_bar
from scipy import ndimage

WIDTH = 7.8  # inches, the width of the documentation column at 110 dpi


def panels(columns, rows=1, width=WIDTH, bars=0):
    """A row (or grid) of frameless square image panels, leaving room for
    ``bars`` colorbars in each row."""
    side = (width - 0.9 * bars) / columns
    figure, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(width, rows * (side + 0.35) + 0.2),
    )
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


def scaled(estimate, reference):
    """``estimate`` scaled to ``reference`` in the least-squares sense."""
    a, b = estimate.abs().double(), reference.abs().double()
    return (float((a * b).sum() / (a * a).sum()) * a).float()


# The phantom: the BrainWeb [#brainweb]_ slice of lesson 2, rendered as a
# T1-weighted spin-echo image with a smooth phase.
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

import torch

SIZE = 128
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
peak = float(image.abs().max())


def simulate(acceleration, calibration, seed=11):
    """A variable-density random set of phase encodes, with a full centre."""
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
    return lines


# sphinx_gallery_end_ignore

# %%
# Simulated acquisition
# ---------------------
#
# The object is the BrainWeb slice of lesson 2 [#brainweb]_, at 128 by 128,
# and you receive it with BART's analytical eight-channel head coil, as there.
# You keep a third of the phase encodes, drawn at random and more densely near
# the centre, but only six fully sampled lines at the centre itself. How the
# phantom and the pattern are made is hidden; ``lines`` is a vector of ones on
# the acquired phase encodes.
import torch

import bartorch
import bartorch.tools as bt

from bartorch import nlop

COILS = 8
ACCELERATION = 3
CALIBRATION = 6  # lines at the centre, far fewer than ESPIRiT needs

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities /= bartorch.rss(sensitivities, axes=(0,), keepdim=True)

kspace = bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True)
kspace = bt.noise(kspace, n=1e-5, s=42)

lines = simulate(ACCELERATION, CALIBRATION)
pattern = lines.reshape(SIZE, 1).to(torch.complex64)
measured = kspace[:, None] * pattern

print(f"{float(lines.mean()):.0%} of the lines, {CALIBRATION} at the centre")

# %%
# Six central lines tell you where the centre of k-space is, but they do not
# calibrate anything on their own. ESPIRiT's kernel is six points wide by
# default, so a six-line ACS region leaves it a single kernel position along
# the phase-encoding axis: far too few rows for the calibration matrix of
# :func:`bartorch.tools.ecalib`.
#
# Joint reconstruction
# --------------------
#
# :func:`bartorch.tools.nlinv` takes the k-space and returns the image and,
# if you ask, the sensitivities it estimated on the way. ``maxiter`` counts
# Gauss-Newton steps, not linear iterations. The regularization weight is
# halved after every step, so the count is a regularization parameter rather
# than a convergence threshold: fewer steps give a smoother image, more steps
# admit more noise. BART's default is eight steps; you use
# twelve.
STEPS = 12

reconstruction, estimated = bt.nlinv(
    measured,
    maxiter=STEPS,
    return_sensitivities=True,
)

channel_images = bartorch.ifft(measured[:, 0], axes=(-2, -1), unitary=True)
zero_filled = bartorch.rss(channel_images, axes=(0,))

for name, estimate in (("zero-filled", zero_filled), ("nlinv", reconstruction)):
    nrmse = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
    print(f"NRMSE, {name:>11}: {nrmse:.3f}")

# sphinx_gallery_start_ignore
figure, axes = panels(2, rows=2, bars=1)
show(axes[0, 0], image, "reference", vmax=peak)
show(axes[0, 1], scaled(zero_filled, image), "zero-filled", vmax=peak)
show(axes[1, 0], scaled(reconstruction, image), "nlinv", vmax=peak)
handle = show(
    axes[1, 1],
    (scaled(reconstruction, image) - image.abs()).abs() / peak,
    "|error|, nlinv",
    cmap="magma",
    vmax=0.1,
)
figure.colorbar(handle, ax=axes[1, 1], fraction=0.046, label="|error| / peak")
# A colorbar of the same width keeps the upper panels aligned with the lower.
figure.colorbar(handle, ax=axes[0, 1], fraction=0.046).ax.set_visible(False)
plt.show()
# sphinx_gallery_end_ignore

# %%
# From a third of the phase encodes and six central lines, nonlinear
# inversion removes the aliasing of the zero-filled image. Its error sits at
# tissue boundaries, around the skull, and in the noise-like residue that
# random sampling leaves. As in lesson 2, ``scaled=True`` fits one global
# factor before comparing, so the NRMSE compares shapes, not the overall
# scale.
#
# The estimated sensitivities
# ----------------------------
#
# The coil unknown in NLINV is not the sensitivity map itself but its k-space
# representation :math:`\hat{s}`, and the map follows as
#
# .. math::
#
#    S = \mathcal{F}^{-1}\left[(1 + a|k|^2)^{-b/2} \, \hat{s}\right],
#
# a Sobolev-norm weighting that suppresses high spatial frequencies. Every
# Gauss-Newton step in :math:`\hat{s}` is therefore a smooth change of the
# map, and the joint problem needs no separate penalty on the coils.
#
# This matters because the pair is determined only up to a common factor:
# multiply every map by a nonzero function :math:`\gamma(r)`, divide the image
# by it, and the data do not change (:doc:`../../explanation/nonlinear`). The
# weighting restricts :math:`\gamma` to smooth functions, so your estimate can
# differ from the true maps by a smooth common magnitude and phase. To compare
# the two, you remove that factor: divide the maps by their root sum of
# squares and subtract the phase of channel 0.
maps = estimated[:, 0]
maps = maps / bartorch.rss(maps, axes=(0,), keepdim=True)
maps = maps * torch.exp(-1j * maps[0].angle())

# sphinx_gallery_start_ignore
# BART's coil model gives channel 0 zero phase, so the simulated maps are
# normalized this way already.
inside = torch.as_tensor(ndimage.binary_fill_holes((signal > 0.02).numpy()))
pairs = (("simulated", sensitivities * inside), ("nlinv", maps * inside))
channels = (2, 4, 6)
figure, axes = panels(3, rows=2, bars=1)
for row, (label, shown) in enumerate(pairs):
    for column, channel in enumerate(channels):
        domain(axes[row, column], shown[channel], f"{label}, channel {channel}", ceiling=1.0)
phase_bar(figure, axes)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inside the head, the normalized estimates reproduce the simulated maps in
# both magnitude and phase. Outside it neither factor is determined at all,
# since their product is zero there for any pair, so the maps follow from the
# initialization and the weighting; the figure masks them. ``nlinv`` reports
# its image multiplied by the root sum of squares of its maps, so the image
# it returns is the one that goes with maps normalized this way.
#
# The model and the solver
# ------------------------
#
# ``nlinv`` runs a fixed configuration. To change any part of it, you write
# the model and the solver out explicitly.
# :class:`bartorch.nlop.NonlinearSense` is the forward model above as a
# nonlinear operator with two inputs, the image and the coil coefficients, and
# :class:`bartorch.nlop.IRGNM` is the Gauss-Newton loop over it. Each step
# linearizes the model at the current estimate :math:`x_k` and solves
#
# .. math::
#
#    \min_x \, \| DF_{x_k} (x - x_k) - (y - F(x_k)) \|^2
#    + \alpha_k \| x - x_{\mathrm{ref}} \|^2,
#
# where :math:`DF_{x_k}` is the derivative of the forward model,
# :math:`x_{\mathrm{ref}}` is zero unless you give one, and :math:`\alpha_k`
# is halved after every step: the first steps are heavily regularized and the
# later ones are not.
model = nlop.NonlinearSense(
    (COILS, 1, SIZE, SIZE),
    pattern=lines.reshape(1, SIZE, 1).to(torch.complex64),
)
print(f"inputs {model.ishapes} -> output {model.oshapes}")

# %%
# To get what ``nlinv`` gets, you need three of its settings. It scales the
# data by :math:`100 / \|y\|` before it starts, which fixes what
# :math:`\alpha` means, and it runs the conjugate gradients of each step to a
# hundred iterations or a relative tolerance of a tenth:
scale = 100.0 / float(torch.linalg.vector_norm(measured))
data = model.prepare(measured * scale)

solver = nlop.IRGNM(iterations=STEPS, cg_maxiter=100, cg_tol=0.1)
fitted, coefficients = solver(data, model)

coil_maps = model.coils(coefficients)
combined = fitted.squeeze() * bartorch.rss(coil_maps[:, 0], axes=(0,))

# sphinx_gallery_start_ignore
application = bt.nlinv(measured, maxiter=STEPS, normalize=False)
difference = float((fitted.squeeze() - application).abs().max())
print(f"largest difference from nlinv: {difference / float(fitted.abs().max()):.1e}")
print(f"NRMSE {bt.nrmse(image.abs(), combined.abs(), scaled=True):.3f}")
# sphinx_gallery_end_ignore

# %%
# The loop you wrote and the application agree to about a thousandth of the
# image peak, and give the same NRMSE, rather than agreeing to the last bit:
# the data scaling is computed here and inside the application by different
# expressions.
#
# What the operator form gives you is access to everything around the step.
# You can hand the linearized problem to a solver from :mod:`bartorch.optim`
# instead of the conjugate gradients inside the library (``inner=optim.CG()``
# is the same method written out, and a regularized solver makes the step a
# regularized one), unroll the loop as :class:`bartorch.nlop.IRGNMBlock`, and
# differentiate a Gauss-Newton step with respect to the data, the iterate,
# the regularization centre and :math:`\alpha`
# (:doc:`../../explanation/learned-reconstruction`). Putting a signal model in front
# of the same encoding, to reconstruct parameter maps rather than an image, is
# lesson 9 (:doc:`../05-model-based/01-quantitative-models`).
#
# Exercise: Gauss-Newton steps
# ----------------------------
#
# The number of Gauss-Newton steps was described above as a regularization
# parameter. Run ``nlinv`` for a few more step counts and compute the NRMSE
# of each image; the plot adds the twelve-step result from above:
counts = (4, 8, 16)
scores = []
for count in counts:
    estimate = bt.nlinv(measured, maxiter=count)
    scores.append(bt.nrmse(image.abs(), estimate.abs(), scaled=True))

# sphinx_gallery_start_ignore
# The twelve-step reconstruction above completes the sweep.
counts, scores = zip(
    *sorted(
        [
            *zip(counts, scores),
            (STEPS, bt.nrmse(image.abs(), reconstruction.abs(), scaled=True)),
        ]
    )
)
for count, value in zip(counts, scores):
    print(f"{count:>3} steps: NRMSE {value:.3f}")

figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 0.42 * WIDTH))
axis.plot(counts, scores, "o-", lw=1.8, color="C0")
axis.set_xlabel("Gauss-Newton steps")
axis.set_ylabel("NRMSE")
axis.set_xticks(counts)
axis.set_ylim(0, None)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The error is lowest at twelve steps and higher on both sides. With four
# steps the regularization weight is still large and the estimate is biased
# towards the regularization centre; at sixteen the weight has been halved
# four more times and the noise amplified by the inversion dominates. The
# optimum depends on the noise level and the undersampling: try a different
# ``n`` in ``bt.noise`` and see where the minimum moves. Later steps are also
# more expensive than early ones, so a large step count costs considerably
# more time.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With bartorch, reconstruct a 2D Cartesian 8-channel k-space of shape
#    (coils, 1, y, x), undersampled along y to a third of the lines with only
#    6 fully sampled central lines, by nonlinear inversion: bt.nlinv with 12
#    Gauss-Newton steps, returning the sensitivities. Then write the same
#    reconstruction as nlop.NonlinearSense (with the sampling pattern) under
#    nlop.IRGNM(iterations=12, cg_maxiter=100, cg_tol=0.1), on data scaled
#    by 100 / ||y||, and check it against bt.nlinv(normalize=False). Compare
#    the estimated maps with the true ones after dividing by their root sum
#    of squares and removing the phase of channel 0.

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
# .. [#nlinv] Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized
#    nonlinear inversion -- joint estimation of coil sensitivities and image
#    content. *Magn Reson Med* 60(3):674-682 (2008).
#    https://doi.org/10.1002/mrm.21691
#
# .. [#bakushinsky] Bakushinsky AB, Kokurin MY. *Iterative Methods for Approximate Solution of
#    Inverse Problems.* Springer (2004).
#    https://doi.org/10.1007/978-1-4020-3122-9
