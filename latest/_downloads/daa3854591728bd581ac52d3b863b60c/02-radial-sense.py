"""
===========================
Radial SENSE reconstruction
===========================

This lesson reconstructs an undersampled golden-angle radial acquisition with
eight receive coils: the density-compensated gridding reconstruction first,
then an iterative SENSE reconstruction with coil sensitivities estimated from
the radial data themselves, with and without a total-variation penalty. The
aim is to see which of the streak artefacts of radial undersampling the coil
encoding removes, which the regularization removes, and what each costs.

A radial acquisition that satisfies the Nyquist criterion at the edge of
k-space needs :math:`\\pi/2` times as many spokes as the matrix has lines;
with fewer, the azimuthal gaps between spokes alias into streaks that run
across the whole field of view. Radial undersampling is nevertheless
benign compared with Cartesian undersampling: every spoke passes through the
k-space centre, so the low spatial frequencies stay fully sampled and the
aliasing is incoherent rather than a discrete fold-over. Parallel imaging
removes it by fitting the image to the non-Cartesian SENSE model
[#pruessmann2001]_

.. math::

   A = W \\, \\mathrm{NUFFT} \\, S,

with :math:`S` the coil sensitivities, the NUFFT evaluated along the
trajectory and :math:`W` an optional weighting of the samples. The fit is
solved iteratively, since :math:`A^H A` is not diagonal in any basis.

The measured data are simulated with the same transform the reconstruction
uses, so the comparison isolates the undersampling, the noise and the error
of the estimated sensitivities from any mismatch between the forward model
and the measurement. The phantom and the coil sensitivities are built as in
:doc:`../01-basics/02-from-kspace-to-image`; the cell that does it is hidden on
this page and present in the script this page can be downloaded as.

**Learning objectives**

- Simulate a multichannel radial acquisition with
  :class:`bartorch.linop.NoncartesianSense`.
- Estimate sensitivities from the radial data with
  :func:`bartorch.tools.ncalib`.
- Compare gridding, unregularized CG-SENSE and total-variation-regularized
  SENSE, and identify the residual artefact of each.
- Reconstruct with :func:`bartorch.apps.pics` and with the operator under
  :class:`bartorch.optim.ADMM`, and compare the two forms of the normal
  operator.

It follows :doc:`01-trajectories-and-transforms`. The next lesson,
:doc:`../05-model-based/01-quantitative-models`, fits a signal model to the
data. The Tour :doc:`../08-workflows/01-dynamic-golden-angle` adds a time axis
to this encoding.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap
from gallery_style import domain, phase_bar

WIDTH = 7.8  # inches, the width of the documentation column

# Fuderer et al. (Magn Reson Med 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()

# Colormap, window and unit per parameter.  Both relaxation windows stop short
# of cerebrospinal fluid, so that white and grey matter -- 500 against 833 ms
# in T1, 70 against 83 ms in T2 -- take up most of the scale and CSF saturates.
STYLE = {
    "T1": (LIPARI, (0.0, 1200.0), "$T_1$ [ms]"),
    "T2": (NAVIA, (0.0, 120.0), "$T_2$ [ms]"),
}


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


def parameter(axis, values, name, title=None):
    """One relaxation map in the colormap and window its parameter is read in."""
    cmap, limits, _ = STYLE[name]
    return show(axis, values, title, vmax=limits[1], cmap=cmap, vmin=limits[0])


def scalebar(figure, axes, handle=None, label=None, name=None):
    """One colorbar for a group of panels, so none gives up width to its own."""
    if name is not None:
        cmap, limits, label = STYLE[name]
        handle = plt.cm.ScalarMappable(plt.Normalize(*limits), cmap)
    return figure.colorbar(handle, ax=axes, fraction=0.046, label=label)


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


# sphinx_gallery_end_ignore
import csv
import time
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import apps, linop, optim, priors

SIZE = 192
COILS = 8
SPOKES = 48  # against pi/2 * SIZE = 302 for a trajectory that is not undersampled
TV_WEIGHT = 0.0005
ITERATIONS = 30

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
# Forty-eight golden-angle spokes of 192 samples each, across a 192 matrix:
# :math:`\pi/2 \times 192 \approx 302` spokes would sample the edge of k-space
# at the Nyquist rate, so the acquisition is undersampled by a factor of about
# 6.3. Successive spokes are rotated by the golden angle, 111.25 degrees, so
# any contiguous subset of them covers k-space nearly uniformly
# [#winkelmann]_; :doc:`../08-workflows/01-dynamic-golden-angle` relies on that property.
#
# :class:`bartorch.linop.NoncartesianSense` maps an image to the samples of
# every channel along the trajectory. The measurement is that operator applied
# to the phantom, with complex Gaussian noise of variance :math:`10^{-4}` per
# sample added, as in the Cartesian lessons.

trajectory = bt.traj(readout=SIZE, spokes=SPOKES, radial=True, golden=True)

E = linop.NoncartesianSense(sensitivities, (SIZE, SIZE), traj=trajectory)
measured = bt.noise(E(image), n=1e-4, s=7)

print(f"{E.ishape} -> {E.oshape}")
print(E.plan)

# %%
#
# The operator's samples are ``(coils, spokes, samples)`` along a trajectory
# ``(spokes, samples, 3)`` whose ``kz`` component is zero, which makes the
# transform two-dimensional. :func:`bartorch.apps.pics` takes that layout.
# BART's commands, :func:`bartorch.tools.ncalib` and
# :func:`bartorch.nufft_adjoint` among them, carry non-Cartesian k-space as
# ``(coils, spokes, samples, 1)``, whose trailing axis is the readout dimension
# of a Cartesian acquisition, so they are given ``measured[..., None]``.
#
# Sensitivity calibration
# -----------------------
#
# ESPIRiT reads its calibration matrix from a Cartesian neighbourhood of the
# k-space centre, so on radial data it needs that region gridded first.
# :func:`bartorch.tools.ncalib` estimates the sensitivities from the samples as
# they were measured, by nonlinear inversion [#nlinv]_ at low resolution; the
# densely sampled centre of a radial acquisition acts as its own
# autocalibration region.
#
# ``N=True`` divides the estimated maps by their root sum of squares. A SENSE
# fit recovers the image :math:`x` for which :math:`Sx` explains the data, so
# maps whose root sum of squares varies across the field of view leave its
# reciprocal in the image as a smooth intensity shading. ESPIRiT maps are
# normalized by construction; nonlinear inversion maps are not.

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
#
# The estimated maps reproduce the magnitude and phase of the simulated ones
# over the head. They are smoother, because nonlinear inversion penalizes the
# high spatial frequencies of the sensitivities, and outside the head, where
# there is no signal to calibrate from, they are extrapolated.
#
# Gridding
# --------
#
# The gridding reconstruction is the density-compensated adjoint: each sample
# is weighted by its distance from the k-space centre (the ramp filter of
# filtered back-projection), the samples are interpolated onto the grid by the
# adjoint NUFFT, and the channels are combined by root sum of squares. It uses
# no model of the coil encoding, so the missing spokes appear in it as the
# streaks the point spread function of the trajectory predicts.

weights = torch.linalg.norm(trajectory.real[..., :2], dim=-1, keepdim=True)
weights = weights.clamp(min=0.25).to(torch.complex64)

channels = bartorch.nufft_adjoint(measured[..., None] * weights, trajectory, (SIZE, SIZE))
gridded = bartorch.rss(channels[:, 0], axes=(0,))

# %%
#
# Iterative SENSE
# ---------------
#
# Conjugate gradients on the normal equations :math:`A^H A x = A^H y`, with no
# penalty, is CG-SENSE [#pruessmann2001]_. The coil encoding separates the
# aliased signal the streaks consist of, but at this undersampling the problem
# is ill-conditioned, and each further iteration fits more of the noise; the
# number of iterations acts as the regularization. A total-variation penalty
# [#rof]_ adds prior knowledge instead: streaks and noise have a large total
# variation, the anatomy a small one. ADMM is the algorithm ``pics`` selects
# for this penalty.

cg_sense = apps.pics(measured, maps, traj=trajectory, maxiter=30)

term = priors.TotalVariation(axes=(-1, -2), weight=TV_WEIGHT)

start = time.perf_counter()
reconstruction = apps.pics(
    measured, maps, traj=trajectory, regularizers=term, solver="admm", maxiter=ITERATIONS
)
print(f"pics: {time.perf_counter() - start:.2f} s")

results = {"gridding": gridded, "CG-SENSE": cg_sense, "SENSE + TV": reconstruction}
for name, estimate in results.items():
    error = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
    similarity = bt.ssim(image.abs(), scaled(estimate, image))
    print(f"{name:>12}  NRMSE {error:.3f}  SSIM {similarity:.3f}")

# %%

# sphinx_gallery_start_ignore
peak = float(image.abs().max())
figure, axes = panels(2, 2, width=0.8 * WIDTH)
show(axes[0, 0], image, "reference", vmax=peak)
for axis, (name, estimate) in zip(axes.flat[1:], results.items()):
    show(axis, scaled(estimate, image), name, vmax=peak)
figure.suptitle(f"{SPOKES} golden-angle spokes, 8 coils")
plt.show()

figure, axes = panels(3)
errors(figure, axes[0], results.values(), image, 0.2)
for axis, name in zip(axes[0], results):
    axis.set_title(name)
figure.suptitle("error magnitude")
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
#
# Gridding shows the streaks of radial undersampling over the whole field of
# view, superimposed on an image that is otherwise sharp: the low spatial
# frequencies are fully sampled, and the streaks come from the high ones. The
# error map shows them extending outside the head, where the object has no
# signal. CG-SENSE removes the streaks, whose aliased signal the coil
# encoding separates, and leaves amplified noise across the head; its largest
# errors are at the scalp, whose bright, thin edge has most of its energy at
# spatial frequencies the sparse outer k-space samples poorly. The
# total-variation penalty suppresses the noise as well. Its residual error
# lies along the tissue boundaries, and in the enlarged region the thin
# cortical folds are flattened, since a boundary between two tissues of
# similar intensity also has a small total variation. The disc of k-space the
# trajectory samples limits the resolution of all three.
#
# The same solve through the operator
# -----------------------------------
#
# Besides the iteration, ``pics`` divides the data by a scale and multiplies
# the image back by it. Off the Cartesian grid it estimates the scale from the
# adjoint reconstruction and therefore needs the operator, which
# :func:`bartorch.optim.data_scaling` takes. The encoding is the operator
# built above, now over the estimated sensitivities.

A = linop.NoncartesianSense(maps[:, 0], (SIZE, SIZE), traj=trajectory)
scale = optim.data_scaling(measured[..., None], A=A)
data = measured / scale

start = time.perf_counter()
assembled = optim.ADMM(term, maxiter=ITERATIONS)(data, A) * scale
print(f"operator and solver: {time.perf_counter() - start:.2f} s")

difference = (assembled.squeeze() - reconstruction.squeeze()).abs().max()
print(f"relative difference from pics: {float(difference / reconstruction.abs().max()):.1e}")

# %%
#
# The two run the same iteration over the same operator. The NUFFT spreads
# samples onto the grid over several threads and sums in the order they
# finish in, so the two can differ at the level of floating-point round-off.
#
# The normal operator
# -------------------
#
# Each iteration applies :math:`A^H A`. For a single coil this is a
# convolution with the point spread function of the trajectory, so it can be
# computed exactly by FFTs on a grid of twice the matrix size (the Toeplitz
# embedding [#fessler]_) instead of by a NUFFT and an adjoint NUFFT; with coils
# it is that convolution between multiplications by the sensitivities. The
# operator uses the convolution by default, and ``toeplitz=False`` requests the
# transform pair. The two differ by the tolerance of the transforms, and the
# iterations carry that difference into the reconstructions.

start = time.perf_counter()
transforms = linop.NoncartesianSense(maps[:, 0], (SIZE, SIZE), traj=trajectory, toeplitz=False)
pair = optim.ADMM(term, maxiter=ITERATIONS)(data, transforms) * scale
print(f"without the Toeplitz normal: {time.perf_counter() - start:.2f} s")
print(f"relative difference {float((pair - assembled).abs().max() / assembled.abs().max()):.1e}")

# %%
#
# The convolution costs an FFT, a pointwise multiplication and an inverse FFT
# on the doubled grid per coil, independent of the number of samples; the pair
# costs two non-uniform transforms, whose spreading and interpolation grow with
# the number of samples. With 48 spokes there are fewer samples than grid
# points, and the pair is not the slower of the two; as the number of samples
# grows, with more spokes or with the frames of a dynamic series sharing one
# normal operator, the convolution becomes the cheaper.

# %%
#
# References
# ----------
#
# .. [#pruessmann2001] Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in sensitivity
#    encoding with arbitrary k-space trajectories. *Magn Reson Med*
#    46(4):638-651 (2001). https://doi.org/10.1002/mrm.1241
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
