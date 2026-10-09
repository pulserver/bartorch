"""
=====================
From k-space to image
=====================

This lesson reconstructs an undersampled Cartesian brain acquisition from its
multichannel k-space to a coil-combined image, and shows what each step of a
parallel-imaging and compressed-sensing pipeline contributes. Scan time in
Cartesian MRI is proportional to the number of phase-encoding lines; skipping
lines shortens the scan by the acceleration factor :math:`R`, but violates the
Nyquist criterion and folds the image onto itself. Recovering an unaliased
image from such data is what the receive coil array, and prior knowledge of
the image, are used for.

The acquisition is simulated from a BrainWeb tissue segmentation and the eight
channels of BART's head-coil model, with one line in three acquired along the
phase-encoding direction. The pipeline consists of coil compression,
sensitivity calibration by ESPIRiT, and a regularized least-squares fit of the
SENSE forward model

.. math::

   y = P F S x + \\varepsilon,

with :math:`S` the coil sensitivities, :math:`F` the Fourier transform,
:math:`P` the sampling operator that keeps the acquired phase encodes, and
:math:`\\varepsilon` complex Gaussian noise. :doc:`../../explanation/encoding`
states the model and :doc:`../../explanation/inverse-problems` the estimator.

Shapes are C order, so a Cartesian k-space is ``(coils, z, y, x)`` with the
readout along ``x`` and the phase encoding along ``y``; see
:doc:`/explanation/data-layout`.

**Learning objectives**

- Simulate a multichannel Cartesian acquisition from a tissue segmentation.
- Undersample the phase-encoding direction with a variable-density pattern
  around a fully sampled autocalibration (ACS) region.
- Compress the channels with :func:`bartorch.tools.cc` and estimate their
  sensitivities with :func:`bartorch.tools.ecalib`.
- Reconstruct with :func:`bartorch.apps.pics`, with a Tikhonov and with a
  wavelet sparsity penalty, and compare the results by error maps, NRMSE and
  SSIM.

It builds on the conventions of :doc:`01-tensors-and-commands`. The sections
after it examine calibration, regularization and the operator form of each
step in turn; the next lesson, :doc:`../02-parallel-imaging/01-coil-calibration`,
compares sensitivity estimators.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap
from gallery_style import domain, phase_bar
from scipy import ndimage

WIDTH = 7.8  # inches, the width of the documentation column at 110 dpi

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


def panels(columns, rows=1, width=WIDTH, bars=0):
    """A row (or grid) of frameless square image panels, leaving room for
    ``bars`` colorbars in each row."""
    side = (width - 0.9 * bars) / columns
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(width, rows * (side + 0.35) + 0.2)
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
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import apps, priors

# %%
#
# Phantom
# -------
#
# BrainWeb [#brainweb]_ publishes a segmentation rather than an image: one membership map
# per tissue class, from which a table of relaxation times and proton
# densities gives the signal of a chosen acquisition. The volume
# ``brainweb-dl`` returns is indexed ``(inferior-superior, posterior-anterior,
# left-right)``, so its first axis selects an axial slice, and an image is
# drawn from its first row down, so flipping it puts anterior at the top.

SIZE = 192
COILS = 8
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]

fractions = np.flipud(get_mri(sub_id=0, contrast="fuzzy")[SLICE])[..., list(TISSUES)].copy()

# %%
#
# The slice is cropped to a square field of view around the head and resampled
# to the matrix reconstructed here. The crop leaves a margin, as a real field
# of view does: the aliased copies of an undersampled acquisition then fall
# partly outside the head.

MARGIN = 0.25

# sphinx_gallery_start_ignore
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
# sphinx_gallery_end_ignore

# %%
#
# A membership-weighted average of the table gives :math:`T_1`, :math:`T_2` and
# the proton density at every voxel, and the spin-echo signal
#
# .. math::
#
#    S = \rho \, \left(1 - e^{-T_R/T_1}\right) e^{-T_E/T_2}
#
# turns those into the image the experiment measures. At a short repetition
# time and a short echo time the contrast is :math:`T_1`-weighted: white matter
# bright, cerebrospinal fluid dark, subcutaneous fat brightest of all.

TR, TE = 600.0, 12.0  # ms

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

# A smooth quadratic phase stands in for the transmit and off-resonance phase
# of a real object, so that nothing below depends on the image being real.
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)

# %%
#
# Coils
# -----
#
# Each receive channel measures the object weighted by its complex sensitivity
# profile, :math:`x_c = S_c x`. The sensitivities here are BART's analytical
# head coil, evaluated on the image grid that :func:`bartorch.tools.grid`
# describes. Dividing them by their root sum of squares over the channels
# normalizes :math:`\sum_c |S_c|^2` to one, so that the optimal coil
# combination of the coil images is the image itself and a reconstruction can
# be compared against it directly. Complex Gaussian noise is then added to
# every k-space sample, as thermal noise is in the receiver chain.

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)

coil_images = sensitivities * image
kspace = bartorch.fft(coil_images, axes=(-2, -1), unitary=True)
kspace = bt.noise(kspace, n=1e-4, s=42)

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, rows=2, bars=2)
peak = float(image.abs().max())
head = proton_density > 0.05
parameter(axes[0, 0], torch.where(head, T1, 0.0), "T1", "$T_1$")
scalebar(figure, axes[0, 0], name="T1")
parameter(axes[0, 1], torch.where(head, T2, 0.0), "T2", "$T_2$")
scalebar(figure, axes[0, 1], name="T2")
handle = show(axes[1, 0], proton_density, "proton density", vmax=1.0)
scalebar(figure, axes[1, 0], handle, "relative")
handle = show(axes[1, 1], image, "$T_1$-weighted image", vmax=peak)
scalebar(figure, axes[1, 1], handle, "magnitude [a.u.]")
plt.show()

# Three of the eight channels, with the outline of the head drawn over each.
outline = ndimage.binary_fill_holes(head.numpy()).astype(float)
figure, axes = panels(3, bars=1)
for column, channel in enumerate((2, 4, 6)):
    domain(axes[0, column], sensitivities[channel], f"channel {channel}", ceiling=1.0)
    axes[0, column].contour(outline, levels=[0.5], colors="white", linewidths=0.8)
phase_bar(figure, axes)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The first figure is the ground truth: the relaxation maps, drawn with the
# perceptually uniform colormaps recommended for relaxometry [#fuderer]_ --
# lipari for :math:`T_1`, navia for :math:`T_2` -- and with a window that stops
# short of cerebrospinal fluid, the proton density, and the
# :math:`T_1`-weighted image they give. The second shows three of the eight
# sensitivities as complex images, with the phase in colour, the magnitude in
# brightness and the outline of the head. Each magnitude is highest near its
# coil element and falls off across the head; the phase varies smoothly. These
# spatial variations are the extra encoding that parallel imaging uses to
# separate aliased voxels.
# :doc:`../02-parallel-imaging/01-coil-calibration` compares how they are
# estimated.
#
# Sampling
# --------
#
# The readout is fully sampled, since it costs no scan time, and a subset of
# the phase encodes is acquired. The lines are drawn at random from a
# variable density that is highest at the k-space centre, where most of the
# signal energy is, with a block of 24 central lines, the autocalibration
# signal (ACS) region, acquired in full. ESPIRiT reads its calibration matrix
# from the ACS region, so an acquisition without one would need a separate
# calibration scan. The pattern is a column vector along the phase-encoding
# direction: it broadcasts over the readout and over the channels.

ACCELERATION = 3
CALIBRATION = 24

encodes = torch.arange(SIZE) - SIZE // 2
profile = (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0
centre = (encodes.abs() < CALIBRATION // 2).to(torch.float32)
drawn = torch.multinomial(
    profile * (1.0 - centre),
    SIZE // ACCELERATION - CALIBRATION,
    replacement=False,
    generator=torch.Generator().manual_seed(11),
)
lines = centre.clone()
lines[drawn] = 1.0

pattern = lines.reshape(SIZE, 1).to(torch.complex64)
measured = kspace[:, None] * pattern

print(f"{int(lines.sum())} of {SIZE} phase encodes acquired, R = {SIZE / lines.sum():.1f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, bars=1)
show(axes[0, 0], pattern.real.expand(SIZE, SIZE), "sampling pattern", vmax=1.0)
axes[0, 0].annotate(
    "ACS",
    xy=(SIZE * 0.98, SIZE / 2),
    xytext=(SIZE * 1.02, SIZE / 2),
    va="center",
    ha="left",
    color="C1",
    annotation_clip=False,
)
log_k = torch.log10(measured[0, 0].abs() / kspace.abs().max()).clamp(min=-5)
handle = show(
    axes[0, 1], log_k.numpy(), "acquired k-space, channel 0", cmap="magma", vmin=-5, vmax=0
)
figure.colorbar(handle, ax=axes[0, 1], fraction=0.046, label="$\\log_{10}$ (|signal| / peak)")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# In the pattern (readout horizontal, phase encoding vertical) every acquired
# phase encode is a full line; the lines cluster towards the centre and the
# ACS band is dense.
#
# Channel compression
# -------------------
#
# Eight channels carry less independent information than eight images: the
# sensitivities overlap, and the singular value spectrum of the calibration
# matrix falls off. :func:`bartorch.tools.cc` returns the matrix that projects
# the channels onto their leading singular vectors [#huangcc]_, the virtual
# coils, and :func:`bartorch.tools.ccapply` applies it. Calibration, the
# encoding operator and every iteration then cost six channels rather than
# eight, at a negligible loss of the encoding capacity of the array.

VIRTUAL = 6

matrix = bt.cc(measured, p=VIRTUAL, M=True, r=CALIBRATION)
compressed = bt.ccapply(measured, matrix, p=VIRTUAL)

# %%
#
# Sensitivity calibration
# -----------------------
#
# ESPIRiT [#espirit]_ estimates the sensitivities from the ACS region alone:
# it builds a calibration matrix from all k-space neighbourhoods (kernels) in
# the region, and obtains the sensitivities at each voxel as the eigenvector
# of an operator derived from that matrix whose eigenvalue is one. Outside the
# object no eigenvalue is close to one; ``crop`` sets the maps to zero where
# the eigenvalue falls below it, which keeps the background out of the
# reconstruction.

maps = bt.ecalib(compressed, maps=1, calib_size=CALIBRATION, crop=0.8)

# %%
#
# Reconstruction
# --------------
#
# Three reconstructions of the same data are compared.
#
# - The **zero-filled** reconstruction sets the missing phase encodes to zero,
#   inverse-transforms each channel and combines them by root sum of squares.
#   It uses no model of the encoding, so every missing line leaves aliasing.
# - **SENSE** [#sense]_ solves :math:`\min_x \|PFSx - y\|_2^2 +
#   \lambda\|x\|_2^2` by conjugate gradients. The sensitivities unfold the
#   aliasing, but the inversion amplifies the noise by the g-factor, which is
#   highest where the coils cannot distinguish aliased voxels.
# - **Compressed sensing** [#lustig]_ replaces the Tikhonov term by an
#   :math:`\ell_1` penalty on the wavelet coefficients, solved by FISTA
#   [#beck]_. The random undersampling makes the aliasing incoherent, i.e.
#   noise-like in the wavelet domain, and the sparsity penalty removes it
#   together with the amplified noise.

channel_images = bartorch.ifft(compressed[:, 0], axes=(-2, -1), unitary=True)
zero_filled = bartorch.rss(channel_images, axes=(0,))

sense = apps.pics(compressed, maps, l2=0.001, maxiter=60)
wavelet = apps.pics(
    compressed,
    maps,
    regularizers=priors.Wavelet((-1, -2), 0.004),
    solver="fista",
    maxiter=100,
)

# %%
#
# The sensitivities ESPIRiT estimates and the ones the acquisition was
# simulated with differ by a phase that varies from voxel to voxel, so the
# reconstructed image does too, and the comparison is between magnitudes.
# ``pics`` returns the image in the units of the k-space rather than those of
# the simulated object, so :func:`bartorch.tools.nrmse` is called with
# ``scaled=True``, which fits a global factor before comparing.

results = {"zero-filled": zero_filled, "SENSE": sense, "wavelet CS": wavelet}
for name, estimate in results.items():
    error = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
    similarity = bt.ssim(image.abs(), scaled(estimate, image))
    print(f"{name:>12}  NRMSE {error:.3f}  SSIM {similarity:.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, rows=2)
show(axes[0, 0], image, "reference", vmax=peak)
for axis, (name, estimate) in zip(axes.flat[1:], results.items()):
    show(axis, scaled(estimate, image), name, vmax=peak)
plt.show()

figure, axes = panels(3, bars=1)
errors(figure, axes[0], results.values(), image, 0.1)
for axis, name in zip(axes[0], results):
    axis.set_title(f"{name} error")
plt.show()

# The posterior horn of the lateral ventricles and the cortex behind it.
zoom = (slice(95, 165), slice(55, 125))
figure, axes = panels(2, rows=2)
show(axes[0, 0], image.abs()[zoom], "reference, enlarged", vmax=peak)
for axis, (name, estimate) in zip(axes.flat[1:], results.items()):
    show(axis, scaled(estimate, image)[zoom], name, vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The zero-filled image carries the aliasing of the missing phase encodes as
# blurring and ghosting along the vertical, phase-encoding direction. SENSE
# removes the coherent aliasing, but its error map shows noise amplified in
# the centre of the head, where the coil sensitivities are least distinct, and
# incoherent residual artefacts of the random sampling. The wavelet penalty
# suppresses both; in the enlarged region the cortical folding and the
# ventricle boundaries are sharper and the background of the brain is smooth.
# The NRMSE and SSIM printed above quantify the same ordering.
#
# How much the penalty removes depends on its weight, which is chosen here and
# not estimated: a larger weight removes more noise and more fine texture with
# it. :doc:`../02-parallel-imaging/01-coil-calibration` compares sensitivity
# estimators, and :doc:`../03-regularization/01-regularized-reconstruction`
# varies the weight.

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
# .. [#fuderer] Fuderer M, Wichtmann B, Crameri F, de Souza NM, Baeßler B, Gulani V,
#    et al. Color-map recommendation for MR relaxometry maps. *Magn Reson Med*
#    93(2):490-506 (2025). https://doi.org/10.1002/mrm.30290
#
# .. [#huangcc] Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR. A software channel
#    compression technique for faster reconstruction with many channels.
#    *Magn Reson Imaging* 26(1):133-141 (2008).
#    https://doi.org/10.1016/j.mri.2007.04.010
#
# .. [#espirit] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS,
#    Lustig M. ESPIRiT -- an eigenvalue approach to autocalibrating parallel
#    MRI: where SENSE meets GRAPPA. *Magn Reson Med* 71(3):990-1001 (2014).
#    https://doi.org/10.1002/mrm.24751
#
# .. [#sense] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: sensitivity
#    encoding for fast MRI. *Magn Reson Med* 42(5):952-962 (1999).
#    https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S
#
# .. [#lustig] Lustig M, Donoho D, Pauly JM. Sparse MRI: the application of compressed
#    sensing for rapid MR imaging. *Magn Reson Med* 58(6):1182-1195 (2007).
#    https://doi.org/10.1002/mrm.21391
#
# .. [#beck] Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm for
#    linear inverse problems. *SIAM J Imaging Sci* 2(1):183-202 (2009).
#    https://doi.org/10.1137/080716542
