"""
==========================
Regularized reconstruction
==========================

Three regularization terms applied to the same undersampled SENSE problem,
and the dependence of the reconstruction error on the regularization weight.

An undersampled, noisy acquisition does not determine the image, and a
reconstruction adds prior information as a penalty:

.. math::

   \\hat{x} = \\arg\\min_x \\tfrac12 \\| A x - y \\|_2^2 + \\lambda\\, g(G x),

with :math:`A` the SENSE encoding, :math:`g` a convex functional,
:math:`G` a linear transform and :math:`\\lambda` the regularization weight.
:mod:`bartorch.priors` provides BART's terms :math:`g(Gx)` as objects, and
:func:`bartorch.apps.pics` solves the problem with the iteration each term
admits. :doc:`../../explanation/inverse-problems` introduces the formulation
and the algorithms.

**Learning objectives**

- Pass regularization terms from :mod:`bartorch.priors` to
  :func:`bartorch.apps.pics`, and choose a solver the term admits.
- Compare Tikhonov, wavelet :math:`\\ell_1` and total-variation
  regularization on the same data.
- Select a regularization weight by the error against a reference, and
  interpret the trade-off between residual noise and loss of detail.
- Combine two terms in one reconstruction.

The previous lessons, :doc:`../01-basics/02-from-kspace-to-image` and
:doc:`../02-parallel-imaging/01-coil-calibration`, reconstructed with a fixed
weight. The next lesson, :doc:`02-operators-and-solvers`, assembles the same
reconstruction from an operator, a term and a solver.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap
from matplotlib.colors import ListedColormap

# Fuderer et al. (Magn Reson Med 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()
# Phase is cyclic, so the colormap has to be: -pi and +pi are the same colour.
# mygbm, turned so that zero phase is yellow and +/-pi is blue.
MYGBM = Colormap("colorcet:CET_C2").to_matplotlib().reversed()
PHASE = ListedColormap(MYGBM([((step + 60) % 256) / 255 for step in range(256)]))

# Colormap, window and unit per parameter.  Both relaxation windows stop short
# of cerebrospinal fluid, so that white and grey matter -- 500 against 833 ms
# in T1, 70 against 83 ms in T2 -- take up most of the scale and CSF saturates.
STYLE = {
    "T1": (LIPARI, (0.0, 1200.0), "$T_1$ [ms]"),
    "T2": (NAVIA, (0.0, 120.0), "$T_2$ [ms]"),
}

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
        axis.set_title(title)
    return handle


def parameter(axis, values, name, title=None):
    """One relaxation map in the colormap and window its parameter is read in."""
    cmap, limits, _ = STYLE[name]
    return show(axis, values, title, vmax=limits[1], cmap=cmap, vmin=limits[0])


def domain(axis, values, title=None):
    """A complex map the way a coil sensitivity is read: phase in colour,
    magnitude in brightness, so an unsupported corner reads as background
    rather than as a phase."""
    values = values.detach().cpu()
    colours = PHASE((values.angle() / (2 * np.pi) + 0.5).numpy())[..., :3]
    magnitude = values.abs().numpy()
    magnitude = magnitude / max(float(magnitude.max()), 1e-12)
    axis.imshow(colours * magnitude[..., None])
    if title is not None:
        axis.set_title(title)


def scalebar(figure, axes, handle=None, label=None, name=None):
    """One colorbar for a group of panels, so none gives up width to its own."""
    if name is not None:
        cmap, limits, label = STYLE[name]
        handle = plt.cm.ScalarMappable(plt.Normalize(*limits), cmap)
    bar = figure.colorbar(handle, ax=axes, fraction=0.046, label=label)
    return bar


def phase_bar(figure, axes):
    """The colour-to-phase key for the panels beside it."""
    bar = figure.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(-np.pi, np.pi), PHASE),
        ax=axes,
        fraction=0.046,
        ticks=[-np.pi, 0.0, np.pi],
    )
    bar.ax.set_yticklabels(["$-\\pi$", "0", "$\\pi$"])
    bar.set_label("phase [rad]")


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

import bartorch
import bartorch.tools as bt
from bartorch import apps, priors

SIZE = 192
COILS = 8
ACCELERATION = 4
CALIBRATION = 24

# %%
#
# The phantom is the BrainWeb [#brainweb]_ slice of
# :doc:`../01-basics/02-from-kspace-to-image`, with the same eight-channel
# sensitivities; the cell that builds both is hidden on this page and present in
# the script this page can be downloaded as.

# sphinx_gallery_start_ignore
# The phantom, the relaxation maps behind it and the coil sensitivities, built
# as :doc:`/auto_examples/01-basics/02-from-kspace-to-image` builds them.
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

# Where each class sits in ``memberships``, by the name the table gives it.
CLASS = {entries[label]["Tissue"]: index for index, label in enumerate(TISSUES)}

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
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
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
# A quarter of the phase encodes, drawn at random from a variable density
# around a fully sampled calibration region of 24 lines, as in
# :doc:`../01-basics/02-from-kspace-to-image`. The noise is a hundred times
# stronger than in that lesson: complex Gaussian noise of variance
# :math:`10^{-3}` per sample of the unitary transform of an image whose peak
# is one, so that noise amplification, and not only aliasing, determines the
# error of an unregularized reconstruction.

kspace = bt.noise(bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True), n=1e-3, s=42)

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

measured = kspace[:, None] * lines.reshape(SIZE, 1).to(torch.complex64)
maps = bt.ecalib(measured, maps=1, calib_size=CALIBRATION, crop=0.8)

# %%
#
# Three terms
# -----------
#
# Tikhonov regularization, :math:`g(x) = \tfrac12\|x\|_2^2`, is the ``l2``
# argument of ``pics`` and keeps the problem quadratic, so conjugate gradients
# solve it. The wavelet term :math:`\|\Psi x\|_1` promotes an image whose
# wavelet coefficients are sparse [#lustig]_; its transform is inside its
# proximal operator, so FISTA [#beck]_ applies it. Total variation,
# :math:`\sum_r \|(\nabla x)_r\|_2`, promotes a piecewise-constant image
# [#block]_; its transform :math:`\nabla` is not, so it requires a splitting
# method, here ADMM [#boyd]_. The axes of a term are the tensor axes it acts
# along, here the two spatial axes.
#
# The weights are the ones that minimize the error of each term in the sweep
# further down. They are relative to the data divided by the scaling
# :func:`bartorch.optim.data_scaling` estimates, which ``pics`` applies, so the
# same weight means the same thing for data of a different overall scale.

reconstructions = {
    "Tikhonov": apps.pics(measured, maps, l2=0.1, maxiter=60),
    "wavelet": apps.pics(
        measured, maps, regularizers=priors.Wavelet((-1, -2), 0.012), solver="fista", maxiter=60
    ),
    "total variation": apps.pics(
        measured,
        maps,
        regularizers=priors.TotalVariation((-1, -2), 0.012),
        solver="admm",
        maxiter=60,
    ),
}

for name, estimate in reconstructions.items():
    error = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
    similarity = bt.ssim(image.abs(), scaled(estimate, image))
    print(f"{name:>16}  NRMSE {error:.3f}  SSIM {similarity:.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, 4, height=1.1)
peak = float(image.abs().max())
show(axes[0, 0], image, "phantom", vmax=peak)
axes[1, 0].axis("off")
for column, (name, estimate) in enumerate(reconstructions.items(), start=1):
    show(axes[0, column], scaled(estimate, image), name, vmax=peak)
    show(axes[1, column], (scaled(estimate, image) - image.abs()).abs(), vmax=0.2 * peak)
axes[1, 1].set_ylabel("|error|, x5")
figure.suptitle(f"{ACCELERATION}x undersampled, noisy")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The Tikhonov reconstruction retains noise and the incoherent aliasing of the
# random sampling across the whole image. Both sparsity-promoting terms reduce
# them; the wavelet term leaves a faint blockiness at the scale of its coarsest
# wavelets, and total variation flattens the gradual intensity variation within
# white matter into patches of constant intensity.
#
# The regularization weight
# -------------------------
#
# A weight that is too small leaves the noise in; one that is too large
# removes image detail with it, and the error against the reference has a
# minimum between the two. The sweep below spans a factor of sixteen for each
# term. It is possible here because the phantom is known; for measured data
# the weight is chosen by a criterion that does not require the reference, or
# fixed once for a protocol.

sweeps = {
    "Tikhonov": [0.01, 0.03, 0.1, 0.3, 1.0],
    "wavelet": [0.003, 0.006, 0.012, 0.025, 0.05],
    "total variation": [0.003, 0.006, 0.012, 0.025, 0.05],
}


def reconstruct(name, weight):
    if name == "Tikhonov":
        return apps.pics(measured, maps, l2=weight, maxiter=60)
    if name == "wavelet":
        term, solver = priors.Wavelet((-1, -2), weight), "fista"
    else:
        term, solver = priors.TotalVariation((-1, -2), weight), "admm"
    return apps.pics(measured, maps, regularizers=term, solver=solver, maxiter=60)


errors = {
    name: [bt.nrmse(image.abs(), reconstruct(name, w).abs(), scaled=True) for w in weights]
    for name, weights in sweeps.items()
}

# %%

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(6.0, 3.6))
for name, weights in sweeps.items():
    axis.semilogx(weights, errors[name], "o-", label=name)
axis.set_xlabel("regularization weight $\\lambda$")
axis.set_ylabel("NRMSE")
axis.legend()
axis.grid(True, which="both", alpha=0.3)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Each curve has an interior minimum. The weights of the Tikhonov term and of
# the two :math:`\ell_1` terms are not comparable with each other, because the
# functionals differ; only the location of each minimum is meaningful. At its
# minimum each sparsity-promoting term reaches a lower error than Tikhonov
# regularization at its own minimum, for this image and this noise level.
#
# Combining terms
# ---------------
#
# ``regularizers`` accepts a list, whose terms are summed. ADMM splits the
# variable once per term with a nontrivial transform, so it accepts any
# combination; FISTA accepts only terms whose transform is the identity.

combined = apps.pics(
    measured,
    maps,
    regularizers=[priors.Wavelet((-1, -2), 0.006), priors.TotalVariation((-1, -2), 0.006)],
    solver="admm",
    maxiter=60,
)
print(f"wavelet + TV  NRMSE {bt.nrmse(image.abs(), combined.abs(), scaled=True):.3f}")

# %%
#
# With each weight halved the sum reaches an error comparable to either term
# alone. Whether a combination improves on its parts depends on the image and
# the weights, and is established by a comparison such as the one above rather
# than assumed.
#
# :func:`bartorch.apps.pics` builds three objects -- the encoding operator, the
# terms and the iteration -- and runs BART's solver on them. The next lesson,
# :doc:`02-operators-and-solvers`, builds them separately.

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
