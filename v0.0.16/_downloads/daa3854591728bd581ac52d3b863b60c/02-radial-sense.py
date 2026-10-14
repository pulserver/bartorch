"""
===============
8. Radial SENSE
===============

In lesson 7 you computed radial trajectories, the NUFFT, density
compensation and the point spread function, and saw that omitting spokes
produces streak artefacts. In this lesson you reconstruct an undersampled
golden-angle radial scan with eight receive coils: you estimate the coil
sensitivities from the radial data, then reconstruct by gridding, by
iterative SENSE, and by SENSE with a total-variation penalty.

The forward model is the SENSE model of lesson 2 with the FFT replaced by a
NUFFT [#pruessmann2001]_:

.. math::

   y = W \\, \\mathrm{NUFFT} \\, S x + \\varepsilon,

where :math:`S` multiplies by the coil sensitivities, the NUFFT samples each
coil image along the trajectory, :math:`W` is an optional weighting of the
samples and :math:`\\varepsilon` is noise. :math:`A^H A` is not diagonal in
any basis, so the model is solved iteratively, with one NUFFT and one adjoint
NUFFT (or an equivalent convolution) per iteration.

Every spoke passes through the centre of k-space, so the low spatial
frequencies remain fully sampled. The angular gaps between spokes alias into
streaks distributed over the whole field of view, not into the discrete
fold-over of Cartesian undersampling. The coil encoding and the penalty act
on this incoherent aliasing.

**Learning objectives**

- Simulate a multichannel radial acquisition with
  :class:`bartorch.linop.NoncartesianSense`.
- Estimate the sensitivities from the radial data with
  :func:`bartorch.tools.ncalib`.
- Reconstruct by gridding, by CG-SENSE and by SENSE with total variation, and
  name the residual artefact of each.
- Rebuild the same reconstruction from the operator and
  :class:`bartorch.optim.ADMM`, and compare the two forms of the normal
  operator.

Previous: :doc:`01-trajectories-and-transforms`. Next:
:doc:`../05-model-based/01-quantitative-models`, where you put a signal model
inside the forward operator and fit a :math:`T_2` map straight from k-space.
The Tour :doc:`../08-workflows/01-dynamic-golden-angle` adds a time axis to
this encoding.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from gallery_style import domain, phase_bar

WIDTH = 7.8  # inches, the width of the documentation column


def panels(columns, rows=1, width=WIDTH):
    """A row (or grid) of frameless square image panels."""
    side = width / columns
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


# The phantom: a BrainWeb [#brainweb]_ segmentation turned into a
# T1-weighted spin-echo image with a smooth phase, built as lesson 2 builds it.
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

import torch

SIZE = 192
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times
MARGIN = 0.25  # what the field of view leaves around the head

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
# sphinx_gallery_end_ignore

# %%
# Simulated acquisition
# ---------------------
#
# The object is the BrainWeb brain slice [#brainweb]_ of lesson 2, on a 192
# matrix, seen by BART's analytical eight-channel head coil. You acquire 48
# spokes of 192 samples.
# To sample the edge of k-space at the Nyquist rate a radial scan needs
# :math:`\pi/2` times as many spokes as the matrix has lines, here about 302,
# so the scan is undersampled by a factor of about 6.3. Successive spokes are
# rotated by the golden angle, 111.25 degrees, so any contiguous subset of
# them covers k-space nearly uniformly [#winkelmann]_.
#
# :class:`bartorch.linop.NoncartesianSense` implements the model above: it
# maps an image to the samples of every channel along the trajectory. You
# simulate the scan by applying it to the object and adding complex Gaussian
# noise. The data are generated with the same transform used for the
# reconstruction, so the errors below come from the undersampling, the noise
# and the estimated sensitivities, not from a model mismatch.
import time

import torch

import bartorch
import bartorch.tools as bt

from bartorch import apps, linop, optim, priors

COILS = 8
SPOKES = 48

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities /= bartorch.rss(sensitivities, axes=(0,), keepdim=True)

trajectory = bt.traj(readout=SIZE, spokes=SPOKES, radial=True, golden=True)

E = linop.NoncartesianSense(sensitivities, (SIZE, SIZE), traj=trajectory)
measured = bt.noise(E(image), n=1e-4, s=7)

print(f"{E.ishape} -> {E.oshape}")
print(E.plan)

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(0.8 * WIDTH, 0.42 * WIDTH))
k = trajectory.real[..., :2].numpy()
limit = SIZE / 2
for axis, count in zip(axes, (5, SPOKES)):
    for spoke in range(count):
        axis.plot(k[spoke, :, 0], k[spoke, :, 1], color="C0", lw=1.0)
    axis.set_title(f"first {count} spokes" if count < SPOKES else f"all {count} spokes")
    axis.set_aspect("equal")
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel("$k_x$ [grid units]")
axes[0].set_ylabel("$k_y$ [grid units]")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each golden-angle increment places the new spoke inside the largest
# angular gap left by the previous ones, so the angular coverage is nearly
# uniform after any number of spokes. Near the centre the 48 spokes sample
# k-space more densely than the Nyquist rate; towards the edge the azimuthal
# distance between spokes exceeds one grid unit, and this undersampling of the
# high spatial frequencies produces the streaks.
#
# The data are ``(coils, spokes, samples)`` along a trajectory
# ``(spokes, samples, 3)`` whose ``kz`` component is zero, which makes the
# transform two-dimensional. :func:`bartorch.apps.pics` takes that layout.
# BART's commands, :func:`bartorch.tools.ncalib` and
# :func:`bartorch.nufft_adjoint` among them, carry non-Cartesian k-space as
# ``(coils, spokes, samples, 1)``, so you hand them ``measured[..., None]``.
#
# Coil sensitivities from the spokes
# ----------------------------------
#
# ESPIRiT, used in lessons 2 and 3, builds its calibration matrix from a
# Cartesian block at the k-space centre, so on radial data that block must be
# gridded first. :func:`bartorch.tools.ncalib` operates on the non-Cartesian
# samples directly: it runs the nonlinear inversion of lesson 4 [#nlinv]_ at
# low resolution, with the densely sampled centre of the radial scan as the
# calibration region.
#
# ``N=True`` divides the maps by their root sum of squares, as was done for
# the simulated maps above. SENSE recovers the image :math:`x` for which
# :math:`Sx` explains the data, so a root sum of squares that varies across
# the field of view appears in the image as its reciprocal, a smooth intensity
# shading. ESPIRiT maps are normalized by construction; nonlinear-inversion
# maps are not.
maps = bt.ncalib(measured[..., None], t=trajectory, N=True)

# sphinx_gallery_start_ignore
figure, axes = panels(3, 2)
for column, coil in enumerate((0, 3, 6)):
    domain(axes[0, column], sensitivities[coil], f"coil {coil + 1}")
    domain(axes[1, column], maps[coil, 0])
for axis, label in zip(axes[:, 0], ("simulated", "estimated")):
    axis.text(-0.04, 0.5, label, transform=axis.transAxes, rotation=90, ha="right", va="center")
phase_bar(figure, axes)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Brightness is magnitude, colour is phase. The estimated maps reproduce the
# magnitude of the simulated ones and their phase pattern across the head.
# They are smoother, because nonlinear inversion penalizes the high spatial
# frequencies of the sensitivities, and in places their phase is offset from
# the simulated one. A phase common to all coils is not identifiable: SENSE
# cannot separate it from the phase of the image, which absorbs it. Outside
# the head there is no signal to calibrate from, and the maps there are
# extrapolated.
#
# Gridding
# --------
#
# The gridding reconstruction is the density-compensated adjoint of lesson 7,
# applied per channel: weight each sample by its distance from the k-space
# centre (the ramp filter of filtered back-projection), apply the adjoint
# NUFFT, and combine the channels by root sum of squares.
weights = torch.linalg.norm(trajectory.real[..., :2], dim=-1, keepdim=True)
weights = weights.clamp(min=0.25).to(torch.complex64)

channels = bartorch.nufft_adjoint(
    measured[..., None] * weights,
    trajectory,
    (SIZE, SIZE),
)
gridded = bartorch.rss(channels[:, 0], axes=(0,))

# sphinx_gallery_start_ignore
figure, axes = panels(3)
show(axes[0, 0], image, "reference", vmax=peak)
show(axes[0, 1], scaled(gridded, image), "gridding", vmax=peak)
errors(figure, axes[0, 2:], [gridded], image, 0.2)
axes[0, 2].set_title("gridding error")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The anatomy is sharp, since the centre of k-space is fully sampled, but
# streaks cover the whole field of view, including the region outside the
# head where the object has no signal. They are the image convolved with the
# point spread function of lesson 7. Gridding uses no model of the coil
# encoding, so it cannot separate aliased signal from true signal.
#
# CG-SENSE
# --------
#
# :func:`bartorch.apps.pics` with a trajectory and no penalty runs conjugate
# gradients on the normal equations :math:`A^H A x = A^H y`: CG-SENSE
# [#pruessmann2001]_.
cg_sense = apps.pics(measured, maps, traj=trajectory, maxiter=30)

# sphinx_gallery_start_ignore
figure, axes = panels(3)
show(axes[0, 0], scaled(gridded, image), "gridding", vmax=peak)
show(axes[0, 1], scaled(cg_sense, image), "CG-SENSE", vmax=peak)
errors(figure, axes[0, 2:], [cg_sense], image, 0.2)
axes[0, 2].set_title("CG-SENSE error")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The streaks are removed: each coil weights the aliased signal differently,
# and the coil encoding separates it from the anatomy. The residual is noise
# amplified across the head, since at sixfold undersampling the inversion is
# ill-conditioned. The largest errors are on the scalp, a bright, thin edge
# whose energy lies mostly at high spatial frequencies, where the spokes are
# farthest apart. Without a penalty, the iteration count is the only
# regularization: further iterations fit more of the noise. The last section
# of the lesson varies it.
#
# SENSE with total variation
# --------------------------
#
# A total-variation penalty [#rof]_ over the two image axes regularizes
# explicitly: noise and streaks have a large total variation, piecewise
# constant anatomy a small one. ``pics`` solves the penalized problem with
# ADMM.
TV_WEIGHT = 0.0005
ITERATIONS = 30

term = priors.TotalVariation(axes=(-1, -2), weight=TV_WEIGHT)

start = time.perf_counter()
reconstruction = apps.pics(
    measured,
    maps,
    traj=trajectory,
    regularizers=term,
    solver="admm",
    maxiter=ITERATIONS,
)
print(f"pics: {time.perf_counter() - start:.2f} s")

# sphinx_gallery_start_ignore
results = {"gridding": gridded, "CG-SENSE": cg_sense, "SENSE + TV": reconstruction}
figure, axes = panels(2, 2, width=0.8 * WIDTH)
show(axes[0, 0], image, "reference", vmax=peak)
for axis, (name, estimate) in zip(axes.flat[1:], results.items()):
    show(axis, scaled(estimate, image), name, vmax=peak)
figure.suptitle(f"{SPOKES} golden-angle spokes, {COILS} coils")
plt.show()

figure, axes = panels(3)
errors(figure, axes[0], results.values(), image, 0.2)
for axis, name in zip(axes[0], results):
    axis.set_title(f"{name} error")
plt.show()

# Posterior cortex and the occipital horns of the lateral ventricles.
zoom = (slice(105, 170), slice(60, 130))
figure, axes = panels(2, 2, width=0.8 * WIDTH)
show(axes[0, 0], image.abs()[zoom], "reference", vmax=peak)
for axis, (name, estimate) in zip(axes.flat[1:], results.items()):
    show(axis, scaled(estimate, image)[zoom], name, vmax=peak)
figure.suptitle("enlarged")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The penalty suppresses the noise as well as the streaks. Its residual
# error lies along tissue boundaries, and in the enlarged region behind the
# ventricles the thin cortical folds are flattened: a boundary between two
# tissues of similar intensity also has a small total variation and is
# smoothed with the noise. The resolution of all three is limited by the disc
# of k-space the spokes cover. The NRMSE and SSIM against the reference:
for name, estimate in results.items():
    nrmse = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
    ssim = bt.ssim(image.abs(), scaled(estimate, image))
    print(f"{name:>12}: NRMSE {nrmse:.3f}, SSIM {ssim:.3f}")

# %%
# CG-SENSE and SENSE with TV have nearly the same NRMSE but different SSIM.
# NRMSE sums the squared error over the image: CG-SENSE distributes it as
# noise over the head, TV concentrates it at edges. SSIM compares local
# means, variances and covariances, and is higher for the TV reconstruction.
# ``scaled=True`` fits one global factor before comparing, so the metric
# compares shapes, not overall scale.
#
# CG-SENSE from an operator and a solver
# --------------------------------------
#
# The same reconstruction can be assembled from its parts, as in lesson 6:
# the encoding operator over the estimated maps, and the ADMM solver with the
# same term. ``pics`` also divides the data by a scale before solving and
# multiplies the image back by it; off the grid it estimates the scale from
# the adjoint reconstruction, so :func:`bartorch.optim.data_scaling` takes the
# operator.
A = linop.NoncartesianSense(maps[:, 0], (SIZE, SIZE), traj=trajectory)
scale = optim.data_scaling(measured[..., None], A=A)
data = measured / scale

start = time.perf_counter()
assembled = optim.ADMM(term, maxiter=ITERATIONS)(data, A) * scale
print(f"operator and solver: {time.perf_counter() - start:.2f} s")

difference = (assembled.squeeze() - reconstruction.squeeze()).abs().max()
relative = float(difference / reconstruction.abs().max())
print(f"relative difference from pics: {relative:.1e}")

# %%
# The two run the same iteration over the same operator. They are not
# guaranteed to be bit-identical: the NUFFT spreads samples onto the grid over
# several threads and sums them in the order the threads finish, so two runs
# can differ at the level of floating-point round-off.
#
# The normal operator
# -------------------
#
# Every iteration applies :math:`A^H A`. For one coil that is a convolution
# with the point spread function of the trajectory, so it can be computed
# exactly by FFTs on a grid of twice the matrix size, the Toeplitz embedding
# [#fessler]_, instead of by a NUFFT followed by an adjoint NUFFT; with coils
# it is that convolution between multiplications by the sensitivities. The
# operator uses the convolution by default (``normal=kernel`` in the plan
# printed above), and ``toeplitz=False`` asks for the transform pair.
pair = linop.NoncartesianSense(
    maps[:, 0],
    (SIZE, SIZE),
    traj=trajectory,
    toeplitz=False,
)

start = time.perf_counter()
paired = optim.ADMM(term, maxiter=ITERATIONS)(data, pair) * scale
print(f"without the Toeplitz normal: {time.perf_counter() - start:.2f} s")

relative = float((paired - assembled).abs().max() / assembled.abs().max())
print(f"relative difference {relative:.1e}")

# %%
# The two normal operators differ by the tolerance of the transforms, and
# the iterations propagate that difference into the images. Their costs scale
# differently. The convolution is an FFT, a pointwise multiplication and an
# inverse FFT on the doubled grid per coil, independent of the number of
# samples. The transform pair is two non-uniform transforms, whose spreading
# and interpolation cost grows with the number of samples. The convolution
# therefore gains as the number of samples grows, with more spokes or with
# the frames of a dynamic series sharing one normal operator. The timings
# printed above are for this machine and this sample count.
#
# Exercise: early stopping of CG
# ------------------------------
#
# Check that the iteration count regularizes CG-SENSE: rerun it with a range
# of iteration counts and compute the NRMSE of each.
counts = (5, 10, 20, 40, 60)
scores = []
for count in counts:
    estimate = apps.pics(measured, maps, traj=trajectory, maxiter=count)
    scores.append(bt.nrmse(image.abs(), estimate.abs(), scaled=True))

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 0.42 * WIDTH))
axis.plot(counts, scores, "o-", lw=1.8, label="CG-SENSE")
tv_score = bt.nrmse(image.abs(), reconstruction.abs(), scaled=True)
axis.axhline(tv_score, color="C1", lw=1.2, ls="--", label="SENSE + TV")
axis.set_xlabel("CG iterations")
axis.set_ylabel("NRMSE")
axis.set_xticks(counts)
axis.set_ylim(0, None)
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
for count, score in zip(counts, scores):
    print(f"{count:3d} iterations: NRMSE {score:.3f}")
# sphinx_gallery_end_ignore

# %%
# The error falls steeply over the first iterations, as CG recovers the
# image, reaches its minimum between 20 and 40 iterations, and rises again at
# 60, as CG fits more of the noise: semi-convergence. The dashed line is the
# TV reconstruction. Repeat the experiment with fewer spokes, or a higher
# noise level in ``bt.noise``, and see where the minimum moves.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Reconstruct a 2D golden-angle radial multichannel acquisition (8 coils,
#    48 spokes of 192 samples, 192x192 matrix) with bartorch. Simulate it with
#    linop.NoncartesianSense and bt.noise. Estimate normalized sensitivities
#    with bt.ncalib(kspace[..., None], t=traj, N=True). Reconstruct three ways:
#    gridding (ramp-weighted bartorch.nufft_adjoint, root sum of squares),
#    CG-SENSE (apps.pics, 30 iterations) and SENSE with total variation
#    (priors.TotalVariation over the last two axes, weight 0.0005, ADMM,
#    30 iterations). Repeat the TV reconstruction with optim.data_scaling and
#    optim.ADMM over the operator, with and without the Toeplitz normal, and
#    report the differences. Report NRMSE and SSIM of the magnitudes against
#    the reference after a global least-squares scale.

# %%
#
# References
# ----------
#
# .. [#pruessmann2001] Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in sensitivity
#    encoding with arbitrary k-space trajectories. *Magn Reson Med*
#    46(4):638-651 (2001). https://doi.org/10.1002/mrm.1241
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
#
# .. [#winkelmann] Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An optimal
#    radial profile order based on the golden ratio for time-resolved MRI.
#    *IEEE Trans Med Imaging* 26(1):68-76 (2007).
#    https://doi.org/10.1109/TMI.2006.885337
#
# .. [#nlinv] Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized
#    nonlinear inversion -- joint estimation of coil sensitivities and image
#    content. *Magn Reson Med* 60(3):674-682 (2008).
#    https://doi.org/10.1002/mrm.21691
#
# .. [#rof] Rudin LI, Osher S, Fatemi E. Nonlinear total variation based noise removal
#    algorithms. *Physica D* 60(1-4):259-268 (1992).
#    https://doi.org/10.1016/0167-2789(92)90242-F
#
# .. [#fessler] Fessler JA, Lee S, Olafsson VT, Shi HR, Noll DC. Toeplitz-based iterative
#    image reconstruction for MRI with correction for magnetic field
#    inhomogeneity. *IEEE Trans Signal Process* 53(9):3393-3402 (2005).
#    https://doi.org/10.1109/TSP.2005.853152
