"""
===========================
2. From k-space to an image
===========================

In lesson 1 you saw how BART's arrays look as tensors and how its FFT takes a
fully sampled k-space to an image. A real scan is rarely fully sampled: you
skip phase-encoding lines to shorten it, and the image folds onto itself. In
this lesson you reconstruct such a scan, one step at a time, and after each
step you look at what it did to the image.

The pipeline is the one most of the course builds on: compress the channels,
estimate the coil sensitivities, and solve the SENSE model

.. math::

   y = P F S x + \\varepsilon,

where :math:`S` multiplies by the coil sensitivities, :math:`F` is the Fourier
transform, :math:`P` keeps the phase encodes you acquired, and
:math:`\\varepsilon` is noise. :doc:`../../explanation/encoding` derives the
model.

**Learning objectives**

- Recognise the aliasing of an undersampled Cartesian scan.
- Compress the channels with :func:`bartorch.tools.cc` and estimate their
  sensitivities with :func:`bartorch.tools.ecalib` (ESPIRiT).
- Reconstruct with :func:`bartorch.apps.pics`, first as SENSE and then with a
  wavelet penalty (compressed sensing), and compare the two to the truth.
- See how far each one goes as you skip more lines.

Previous: :doc:`01-tensors-and-commands`. Next:
:doc:`../02-parallel-imaging/01-coil-calibration`, where you look more closely
at where the coil sensitivities come from.
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
    profile = (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0
    centre = (encodes.abs() < calibration // 2).to(torch.float32)
    drawn = torch.multinomial(
        profile * (1.0 - centre),
        round(SIZE / acceleration) - calibration,
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

SIZE = 192
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
head = proton_density > 0.05
# sphinx_gallery_end_ignore

# %%
# Your scan
# ---------
#
# The object is a slice of the BrainWeb brain [#brainweb]_, rendered as a
# :math:`T_1`-weighted spin-echo image; how it is built does not matter here,
# so that code is hidden. You acquire it with BART's analytical eight-channel
# head coil: every channel sees the image through its own sensitivity, and
# the scanner records the Fourier transform of each channel image, plus
# thermal noise.
import torch

import bartorch
import bartorch.tools as bt

from bartorch import apps, priors

COILS = 8

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities /= bartorch.rss(sensitivities, axes=(0,), keepdim=True)

kspace = bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True)
kspace = bt.noise(kspace, n=1e-4, s=42)
print(f"k-space {tuple(kspace.shape)}: (coils, y, x)")

# sphinx_gallery_start_ignore
figure, axes = panels(2, bars=1)
show(axes[0, 0], image, "the object", vmax=peak)
outline = ndimage.binary_fill_holes(head.numpy()).astype(float)
domain(axes[0, 1], sensitivities[2], "channel 2 sensitivity", ceiling=1.0)
axes[0, 1].contour(outline, levels=[0.5], colors="white", linewidths=0.8)
phase_bar(figure, axes[0, 1])
plt.show()
# sphinx_gallery_end_ignore

# %%
# On the right is one channel's sensitivity: brightness is its magnitude,
# colour its phase. It is strong near its coil element and weak across the
# head. Because every channel sees a different part of the head, the eight of
# them together carry spatial information that a single coil does not, and
# that is what lets you skip lines below.
#
# Dividing the sensitivities by their root sum of squares makes the best
# combination of the channel images equal to the image itself, so you can
# compare every reconstruction below directly with ``image``.
#
# Skipping lines
# --------------
#
# You keep every readout sample, since they cost no scan time, and acquire one
# phase-encoding line in three. The lines are drawn at random, more densely
# near the centre of k-space where most of the signal is, around a fully
# sampled block of 24 central lines: the autocalibration (ACS) region, from
# which you will estimate the coil sensitivities. The pattern is a column, so
# it broadcasts over the readout and the channels.
ACCELERATION = 3
CALIBRATION = 24

pattern = simulate(ACCELERATION, CALIBRATION)  # (y, 1), ones on acquired lines
measured = kspace[:, None] * pattern  # (coils, z, y, x), as BART expects
print(f"{int(pattern.real.sum())} of {SIZE} lines acquired")

# %%
# The simplest thing you can do with these data is to transform each channel
# back as it is, with zeros in the missing lines, and combine the channels by
# root sum of squares:
channel_images = bartorch.ifft(measured[:, 0], axes=(-2, -1), unitary=True)
zero_filled = bartorch.rss(channel_images, axes=(0,))

# sphinx_gallery_start_ignore
figure, axes = panels(3, bars=1)
show(axes[0, 0], pattern.real.expand(SIZE, SIZE), "sampling pattern", vmax=1.0)
log_k = torch.log10(measured[0, 0].abs() / kspace.abs().max()).clamp(min=-5)
show(axes[0, 1], log_k.numpy(), "acquired k-space", cmap="magma", vmin=-5, vmax=0)
show(axes[0, 2], scaled(zero_filled, image), "zero-filled", vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Readout runs horizontally, phase encoding vertically. The zero-filled image
# is the aliasing you have to remove: every missing line leaves a faint copy
# of the head, smeared along the phase-encoding direction.
#
# Fewer channels
# --------------
#
# Eight channels carry less than eight images' worth of information, because
# their sensitivities overlap. :func:`bartorch.tools.cc` finds the
# combinations of channels that carry most of it [#huangcc]_ and
# :func:`bartorch.tools.ccapply` projects the data onto them. Six virtual
# channels lose almost nothing, and everything after this runs a quarter
# faster.
VIRTUAL = 6

matrix = bt.cc(measured, p=VIRTUAL, M=True, r=CALIBRATION)
compressed = bt.ccapply(measured, matrix, p=VIRTUAL)
print(f"{measured.shape[0]} channels -> {compressed.shape[0]}")

# %%
# Coil sensitivities
# ------------------
#
# ESPIRiT [#espirit]_ estimates the sensitivities from the ACS region alone.
# ``crop`` sets them to zero where ESPIRiT finds no signal, which keeps the
# background out of the reconstruction. Lesson 3 looks inside this step.
maps = bt.ecalib(compressed, maps=1, calib_size=CALIBRATION, crop=0.8)
print(f"maps {tuple(maps.shape)}")

# sphinx_gallery_start_ignore
figure, axes = panels(3, bars=1)
for column, channel in enumerate((0, 2, 4)):
    domain(
        axes[0, column],
        maps[channel, 0],
        f"virtual channel {channel}",
        ceiling=1.0,
    )
phase_bar(figure, axes)
plt.show()
# sphinx_gallery_end_ignore

# %%
# SENSE
# -----
#
# With the sensitivities you can solve the model at the top of the page.
# :func:`bartorch.apps.pics` does it the way BART's ``pics`` command does;
# with only ``l2``, a small Tikhonov weight, it is SENSE [#sense]_, solved by
# conjugate gradients:
sense = apps.pics(compressed, maps, l2=0.001, maxiter=60)

# sphinx_gallery_start_ignore
figure, axes = panels(3, bars=1)
show(axes[0, 0], scaled(zero_filled, image), "zero-filled", vmax=peak)
show(axes[0, 1], scaled(sense, image), "SENSE", vmax=peak)
errors(figure, axes[0, 2:], [sense], image, 0.1)
axes[0, 2].set_title("SENSE error")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The folded copies are gone. What is left is noise, strongest in the middle
# of the head where the coils are least distinct and the inversion amplifies
# it most (the g-factor), and a speckle of residual aliasing from the random
# sampling.
#
# Compressed sensing
# ------------------
#
# Random sampling makes the leftover aliasing look like noise, and a brain
# image is sparse in a wavelet basis while noise is not. So you can ask
# ``pics`` for the image that fits the data *and* has few wavelet
# coefficients [#lustig]_: give it a :class:`bartorch.priors.Wavelet` term
# over the two image axes, and let FISTA [#beck]_ solve it.
wavelet = apps.pics(
    compressed,
    maps,
    regularizers=priors.Wavelet((-1, -2), 0.004),
    solver="fista",
    maxiter=100,
)

# sphinx_gallery_start_ignore
results = {"zero-filled": zero_filled, "SENSE": sense, "wavelet CS": wavelet}
figure, axes = panels(3, bars=1)
errors(figure, axes[0], results.values(), image, 0.1)
for axis, name in zip(axes[0], results):
    axis.set_title(f"{name} error")
plt.show()

zoom = (slice(95, 165), slice(55, 125))
figure, axes = panels(4)
show(axes[0, 0], image.abs()[zoom], "truth", vmax=peak)
for axis, (name, estimate) in zip(axes[0, 1:], results.items()):
    show(axis, scaled(estimate, image)[zoom], name, vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The error maps are on one scale. The wavelet penalty removes most of the
# noise and the speckle; in the enlarged region behind the ventricles the
# cortex is sharp and the white matter smooth. The NRMSE and SSIM say the
# same:
for name, estimate in results.items():
    nrmse = bt.nrmse(image.abs(), estimate.abs(), scaled=True)
    ssim = bt.ssim(image.abs(), scaled(estimate, image))
    print(f"{name:>12}: NRMSE {nrmse:.3f}, SSIM {ssim:.3f}")

# %%
# ``scaled=True`` fits one global factor before comparing, so that only the
# shape of the error counts and not the overall scale of the image. You
# compare magnitudes because ESPIRiT's maps can carry a different phase from
# the true ones, and the image then carries it too.
#
# The wavelet weight, 0.004, is a choice: a larger one removes more noise and
# more fine detail with it. Lesson 5 shows how to choose it.
#
# Your turn: skip more lines
# --------------------------
#
# How far can you push the acceleration? Rerun the same two reconstructions
# with every third, fourth, fifth and sixth line, keeping the ACS block:
accelerations = (2, 3, 4, 5, 6)
scores = {"SENSE": [], "wavelet CS": []}
for R in accelerations:
    acquired = kspace[:, None] * simulate(R, CALIBRATION)
    data = bt.ccapply(acquired, matrix, p=VIRTUAL)
    maps_R = bt.ecalib(data, maps=1, calib_size=CALIBRATION, crop=0.8)
    sense_R = apps.pics(data, maps_R, l2=0.001, maxiter=60)
    wavelet_R = apps.pics(
        data,
        maps_R,
        regularizers=priors.Wavelet((-1, -2), 0.004),
        solver="fista",
    )
    for name, estimate in (("SENSE", sense_R), ("wavelet CS", wavelet_R)):
        scores[name].append(bt.nrmse(image.abs(), estimate.abs(), scaled=True))

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 0.42 * WIDTH))
for name, values in scores.items():
    axis.plot(accelerations, values, "o-", lw=1.8, label=name)
axis.set_xlabel("acceleration R")
axis.set_ylabel("NRMSE")
axis.set_xticks(accelerations)
axis.set_ylim(0, None)
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# Both errors grow with the acceleration. The wavelet penalty stays ahead
# everywhere, but its lead shrinks: with fewer lines there is less incoherence
# for it to exploit and more of the image itself is missing. Try another
# weight, or a smaller ACS block, and see which one breaks first.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Reconstruct a 2D Cartesian multichannel k-space of shape (coils, 1, y, x),
#    undersampled along y with a fully sampled 24-line ACS block, with bartorch.
#    Compress to 6 virtual channels with bt.cc and bt.ccapply, estimate one set
#    of ESPIRiT maps with bt.ecalib (crop 0.8), then reconstruct with
#    apps.pics twice: SENSE (l2=0.001, 60 CG iterations) and wavelet
#    compressed sensing (priors.Wavelet over the last two axes, weight 0.004,
#    FISTA, 100 iterations). Report NRMSE and SSIM of both magnitudes against
#    the reference after a global least-squares scale.

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
