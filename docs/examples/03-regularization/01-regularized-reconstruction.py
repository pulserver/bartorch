"""
=========================
5. Choosing a regularizer
=========================

In lesson 2 you reconstructed an undersampled scan with a wavelet penalty at a
weight of 0.004, both fixed without justification. In this lesson you
choose them. You reconstruct one noisy, undersampled SENSE
acquisition with three different penalties, sweep the weight of each, and
compare under- and over-regularized images. Then you
combine two penalties in one reconstruction.

The mental model: at high acceleration or low SNR, the SENSE problem is
ill-conditioned: the data do not determine the image uniquely, and a plain
least-squares fit amplifies the noise by the g-factor. You therefore add prior
knowledge about the image as a penalty,

.. math::

   \\hat{x} = \\arg\\min_x \\tfrac12 \\| A x - y \\|_2^2 + \\lambda\\, g(G x),

where :math:`A` is the SENSE encoding, :math:`g` a convex function,
:math:`G` a linear transform and :math:`\\lambda` the regularization weight.
Tikhonov regularization favours a small image; a wavelet :math:`\\ell_1`
penalty favours an image that is sparse in a wavelet basis, as in compressed
sensing; total variation (TV) favours one that is piecewise constant. The
weight sets the balance between data consistency and the penalty.
:doc:`../../explanation/inverse-problems` introduces the formulation and the
algorithms.

**Learning objectives**

- Pass regularization terms from :mod:`bartorch.priors` to
  :func:`bartorch.apps.pics`, with a solver the term admits.
- Compare Tikhonov, wavelet :math:`\\ell_1` and TV regularization on the same
  data, by their images and error maps.
- Choose a regularization weight by the error against a reference, and
  recognise under- and over-regularization in the image.
- Combine two terms in one reconstruction.

Previous: :doc:`../02-parallel-imaging/02-nonlinear-inversion`. Next:
:doc:`02-operators-and-solvers`, where you build the operator, the term and
the solver that ``pics`` puts together, and run them yourself.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

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


def simulate(acceleration, calibration=24, seed=11):
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


# The phantom: the BrainWeb [#brainweb]_ slice of lesson 2, rendered as a
# T1-weighted spin-echo image with a smooth phase, and BART's eight-channel
# head coil.
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

import torch

import bartorch
import bartorch.tools as bt

SIZE = 192
COILS = 8
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)
MARGIN = 0.25
TR, TE = 600.0, 12.0  # ms

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
    proton_density * (1 - torch.exp(-TR / T1.clamp(min=1e-3))) * torch.exp(-TE / T2.clamp(min=1e-3))
)
signal = torch.where(T1 > 0, signal, torch.zeros(()))
signal = signal / signal.max()
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE),
    torch.linspace(-1.0, 1.0, SIZE),
    indexing="ij",
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)
peak = float(image.abs().max())

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

# %%
# Your scan
# ---------
#
# The object, the coils and the sampling are those of lesson 2: the BrainWeb
# slice [#brainweb]_, eight channels, and one phase-encoding line in three
# around a fully sampled ACS block of 24 lines. The one change is the noise,
# three times stronger: complex Gaussian noise of variance
# :math:`3 \times 10^{-4}` per sample of the unitary transform of an image
# whose peak is one. At this level the noise amplified by the inversion, and
# not only the aliasing, sets the error of an unregularized reconstruction.
import torch

import bartorch
import bartorch.tools as bt

from bartorch import apps, priors

ACCELERATION = 3
CALIBRATION = 24

clean = bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True)
kspace = bt.noise(clean, n=3e-4, s=42)
measured = kspace[:, None] * simulate(ACCELERATION, CALIBRATION)

maps = bt.ecalib(measured, maps=1, calib_size=CALIBRATION, crop=0.8)

# %%
# Three penalties
# ---------------
#
# Each penalty admits particular solvers; ``pics`` takes the term and the
# solver:
#
# - Tikhonov, :math:`g(x) = \tfrac12\|x\|_2^2`, is the ``l2`` argument. It
#   keeps the problem quadratic, so conjugate gradients solve it.
# - The wavelet term :math:`\|\Psi x\|_1` [#lustig]_ is
#   :class:`bartorch.priors.Wavelet`. Its transform is orthogonal and sits
#   inside its proximal operator, so FISTA [#beck]_ solves it.
# - Total variation, :math:`\sum_r \|(\nabla x)_r\|_2` [#block]_, is
#   :class:`bartorch.priors.TotalVariation`. The finite differences
#   :math:`\nabla` have no such closed-form proximal operator, so TV needs a
#   splitting method, here ADMM [#boyd]_.
#
# The first argument of a term is the tensor axes it acts along, here the two
# image axes. You run each penalty over four weights:
sweeps = {
    "Tikhonov": [0.01, 0.03, 0.1, 0.3],
    "wavelet": [0.003, 0.006, 0.012, 0.03],
    "total variation": [0.002, 0.006, 0.012, 0.04],
}


def reconstruct(name, weight, data=measured):
    if name == "Tikhonov":
        return apps.pics(data, maps, l2=weight, maxiter=100)
    if name == "wavelet":
        term, solver = priors.Wavelet((-1, -2), weight), "fista"
    else:
        term, solver = priors.TotalVariation((-1, -2), weight), "admm"
    return apps.pics(data, maps, regularizers=term, solver=solver, maxiter=100)


def nrmse(estimate):
    return bt.nrmse(image.abs(), estimate.abs(), scaled=True)


results = {
    name: {weight: reconstruct(name, weight) for weight in weights}
    for name, weights in sweeps.items()
}
scores = {}
for name, sweep in results.items():
    scores[name] = {weight: nrmse(x) for weight, x in sweep.items()}
best = {name: min(sweep, key=sweep.get) for name, sweep in scores.items()}
for name, weight in best.items():
    print(f"{name:>16}: best weight {weight}, NRMSE {scores[name][weight]:.3f}")

# sphinx_gallery_start_ignore
chosen = {name: results[name][weight] for name, weight in best.items()}

figure, axes = panels(2, rows=2)
show(axes[0, 0], image, "the object", vmax=peak)
for axis, (name, estimate) in zip(axes.flat[1:], chosen.items()):
    show(axis, scaled(estimate, image), f"{name}, $\\lambda$ = {best[name]}", vmax=peak)
plt.show()

figure, axes = panels(3, bars=1)
errors(figure, axes[0], chosen.values(), image, 0.1)
for axis, name in zip(axes[0], chosen):
    axis.set_title(f"{name} error")
plt.show()

# Occipital cortex, where the gyri are thinnest.
zoom = (slice(110, 175), slice(60, 130))
figure, axes = panels(2, rows=2)
show(axes[0, 0], image.abs()[zoom], "the object, enlarged", vmax=peak)
for axis, (name, estimate) in zip(axes.flat[1:], chosen.items()):
    show(axis, scaled(estimate, image)[zoom], name, vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
# You see each penalty at its best weight, then the error maps on one scale,
# then the occipital cortex enlarged. Tikhonov leaves the amplified noise and
# the incoherent aliasing of the random sampling across the whole head: a
# quadratic penalty strong enough to remove them would blur the image too,
# so its best weight leaves them in. The two sparsity penalties remove most
# of the noise and keep the tissue boundaries, which is the much darker error
# inside the brain. Their residual artefacts differ in the enlarged region:
# the wavelet penalty leaves a blotchy texture in white matter, and TV turns
# the gradual intensity changes within white matter into patches of constant
# intensity (staircasing).
#
# The regularization weight
# -------------------------
#
# Too small a weight leaves the noise in; too large a weight takes image
# detail out with it. The error against the truth has a minimum in between,
# and the sweep above spans a factor of ten or more for each penalty to find
# it. You can do that here because you know the truth; for measured data you
# choose the weight by a criterion that needs no reference, or fix it once
# for a protocol.

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 0.45 * WIDTH))
for name, sweep in scores.items():
    axis.semilogx(list(sweep), list(sweep.values()), "o-", lw=1.8, label=name)
axis.set_xlabel("regularization weight $\\lambda$")
axis.set_ylabel("NRMSE")
axis.grid(True, which="both", alpha=0.3)
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()

weights = sweeps["total variation"]
figure, axes = panels(3)
for axis, weight, label in zip(
    axes[0], (weights[0], best["total variation"], weights[-1]), ("too small", "best", "too large")
):
    estimate = results["total variation"][weight]
    show(axis, scaled(estimate, image)[zoom], f"TV, $\\lambda$ = {weight}\n{label}", vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each curve has its minimum inside the sweep. You cannot compare weights
# across penalties, because the functions differ; only where each minimum
# lies means something. At its minimum, each sparsity penalty reaches a lower
# error than Tikhonov at its own, for this image and this noise level.
#
# The three TV images show both failure modes. At the smallest weight
# the noise and the aliasing remain. At the largest, the cortex is flattened
# into patches of constant intensity and thin gyri merge.
#
# Combining penalties
# -------------------
#
# ``regularizers`` also takes a list, and the terms are added. ADMM splits
# the variable once per term with a transform of its own, so it takes any
# combination; FISTA takes only terms whose transform is the identity. Here
# is wavelet plus TV, each at half its best weight:
combined = apps.pics(
    measured,
    maps,
    regularizers=[
        priors.Wavelet((-1, -2), best["wavelet"] / 2),
        priors.TotalVariation((-1, -2), best["total variation"] / 2),
    ],
    solver="admm",
    maxiter=100,
)
print(f"wavelet + TV: NRMSE {nrmse(combined):.3f}")

# %%
# The sum reaches an error comparable to either penalty alone. Whether a
# combination beats its parts depends on the image and the weights, and you
# find out with a comparison like the one above rather than by assuming it.
#
# ``nrmse(..., scaled=True)`` fits one global factor before comparing, so it
# compares the shape of the image and not its overall scale.
#
# Your turn: change the noise
# ---------------------------
#
# Does the best weight depend on the noise? Sweep the wavelet weights again on
# the same scan with a third of the noise and with three times as much:
levels = {1e-4: {}, 3e-4: scores["wavelet"], 1e-3: {}}
for level, sweep in levels.items():
    if sweep:
        continue
    noisy = bt.noise(clean, n=level, s=42)
    data = noisy[:, None] * simulate(ACCELERATION, CALIBRATION)
    for weight in sweeps["wavelet"]:
        sweep[weight] = nrmse(reconstruct("wavelet", weight, data))

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 0.45 * WIDTH))
for level, sweep in levels.items():
    axis.semilogx(list(sweep), list(sweep.values()), "o-", lw=1.8, label=f"noise {level:g}")
axis.set_xticks(sweeps["wavelet"], [f"{w:g}" for w in sweeps["wavelet"]])
axis.minorticks_off()
axis.set_xlabel("wavelet weight $\\lambda$")
axis.set_ylabel("NRMSE")
axis.grid(True, which="both", alpha=0.3)
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
for level, sweep in levels.items():
    print(f"noise {level:g}: best wavelet weight {min(sweep, key=sweep.get)}")
# sphinx_gallery_end_ignore

# %%
# The best weight moves with the noise: in this sweep it is 0.003 at a third
# of the noise, 0.006 at the level used above, and 0.012 at three times the
# noise, and the error at every weight rises with the noise. At the lowest
# noise the minimum lies at the smallest weight of the sweep, so the best
# weight there may be smaller still; extend the sweep downwards to find it. A
# weight chosen for one SNR is therefore not the best for another.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With bartorch, reconstruct a 2D Cartesian 8-channel k-space, undersampled
#    3-fold along y around a 24-line ACS block, with ESPIRiT maps (bt.ecalib,
#    crop 0.8) and apps.pics, 100 iterations each: Tikhonov (l2) at
#    0.01-0.3 with CG, priors.Wavelet over the last two axes at 0.003-0.03
#    with FISTA, and priors.TotalVariation at 0.002-0.04 with ADMM. Pick each
#    weight by the lowest NRMSE against the reference (scaled=True), show the
#    images and error maps at the best weights, and run wavelet + TV at half
#    of each best weight with ADMM.

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
# .. [#lustig] Lustig M, Donoho D, Pauly JM. Sparse MRI: the application of compressed
#    sensing for rapid MR imaging. *Magn Reson Med* 58(6):1182-1195 (2007).
#    https://doi.org/10.1002/mrm.21391
#
# .. [#beck] Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm for
#    linear inverse problems. *SIAM J Imaging Sci* 2(1):183-202 (2009).
#    https://doi.org/10.1137/080716542
#
# .. [#block] Block KT, Uecker M, Frahm J. Undersampled radial MRI with multiple coils.
#    Iterative image reconstruction using a total variation constraint.
#    *Magn Reson Med* 57(6):1086-1098 (2007). https://doi.org/10.1002/mrm.21236
#
# .. [#boyd] Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. Distributed optimization and
#    statistical learning via the alternating direction method of multipliers.
#    *Found Trends Mach Learn* 3(1):1-122 (2011). https://doi.org/10.1561/2200000016
