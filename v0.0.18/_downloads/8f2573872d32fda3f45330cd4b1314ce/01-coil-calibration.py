"""
=====================
3. Coil sensitivities
=====================

In lesson 2 you estimated the coil sensitivities with ESPIRiT from a block of
24 central lines and passed them to ``pics`` without examining them. SENSE
[#sense]_ solves

.. math::

   y_c = P F (S_c \\, x)

for the image :math:`x` with the sensitivities :math:`S_c` taken as known.
Whatever is wrong in :math:`S_c` ends up in the image, as residual aliasing or
as shading, however good the solver.

On a scanner the sensitivities come either from a separate low-resolution
prescan or, as here, from the scan itself: a fully sampled block of lines at
the centre of k-space, the autocalibration (ACS) region. In this lesson you
estimate them in three ways, reconstruct with each, and then shrink the ACS
region to see which of the three still works.

**Learning objectives**

- Estimate coil sensitivities with :func:`bartorch.tools.caldir`,
  :func:`bartorch.tools.ecalib` (ESPIRiT) and :func:`bartorch.tools.nlinv`,
  and use each in :func:`bartorch.apps.pics`.
- Recognise what a poor set of maps does to a SENSE reconstruction.
- Explain why ESPIRiT needs an ACS region larger than its kernel, and why
  nonlinear inversion does not.

Previous: :doc:`../01-basics/02-from-kspace-to-image`. Next:
:doc:`02-nonlinear-inversion`, where you reconstruct the image and the
sensitivities together and write that reconstruction out as an operator and a
solver.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from gallery_style import domain, phase_bar

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


# sphinx_gallery_end_ignore

# %%
# Simulated acquisition
# ---------------------
#
# The object is BART's Shepp-Logan phantom, seen through BART's analytical
# eight-channel head coil. As in lesson 1, you ask for its k-space directly
# (``kspace=True``) rather than transforming a sampled image, so the data do
# not share the discretization of the reconstructions below. Then you add
# complex Gaussian noise of variance 2 to every sample.
import torch

import bartorch
import bartorch.tools as bt

from bartorch import apps

SIZE = 128
COILS = 8
ACCELERATION = 3

clean = bt.phantom(SIZE, coils=COILS, kspace=True)
kspace = bt.noise(clean, n=2.0, s=3)

# sphinx_gallery_start_ignore
# The reference: the root sum of squares of the fully sampled, noise-free coil
# images, compared inside the phantom's support.
reference = bartorch.rss(bartorch.ifft(clean, axes=(-2, -1)), axes=(0,))[0].abs()
support = (bt.phantom(SIZE).abs() > 0).to(torch.float32)


def error(estimate):
    """NRMSE of a magnitude image within the phantom's support."""
    return bt.nrmse(reference * support, estimate.abs().squeeze() * support, scaled=True)


def within(estimate):
    """A magnitude image scaled to the reference, inside the support."""
    return scaled(estimate.squeeze(), reference) * support


# The skull is the brightest structure; a window at half its intensity shows
# the aliasing inside the phantom.
peak = 0.5 * float(reference.max())
# sphinx_gallery_end_ignore

# %%
# You acquire every third phase encode, plus a central block of
# ``calibration`` lines in full. Regular undersampling by :math:`R` replicates
# the point spread function :math:`R` times across the field of view, so each
# voxel of the zero-filled image is the sum of three voxels a third of the
# field of view apart. Only the sensitivities, which differ between those
# three voxels, can separate them.
encodes = torch.arange(SIZE) - SIZE // 2


def acquire(calibration):
    """Every third line, and a fully sampled block at the centre."""
    acs = encodes.abs() < calibration // 2
    lines = (encodes % ACCELERATION == 0) | acs
    return kspace * lines.to(torch.complex64).reshape(SIZE, 1)


measured = acquire(24)
print(f"{float(bt.pattern(measured).real.mean()):.0%} of k-space acquired")

channel_images = bartorch.ifft(measured, axes=(-2, -1))
zero_filled = bartorch.rss(channel_images, axes=(0,))[0]

# sphinx_gallery_start_ignore
figure, axes = panels(2, width=0.8 * WIDTH)
show(axes[0, 0], reference, "reference", vmax=peak)
show(axes[0, 1], within(zero_filled), "zero-filled", vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The images are windowed at half the intensity of the skull, which
# saturates. In the zero-filled image you see the three overlapping copies of
# the phantom that regular undersampling makes. The reference on the left is
# the root sum of squares of the fully sampled, noise-free data; every error
# below is the NRMSE of a magnitude image against it, inside the phantom,
# after fitting one global scale, because the three estimators normalize the
# maps, and so the image, differently.
#
# Sensitivity estimation
# ----------------------
#
# **Directly from the ACS lines.** :func:`bartorch.tools.caldir` reconstructs
# a low-resolution image per channel from the ACS region alone and divides
# each by their root sum of squares. It is the simplest estimate, and its
# resolution is the resolution of the ACS block:
direct = bt.caldir(measured, 24)

# %%
# **ESPIRiT** [#espirit]_ slides a small kernel over the ACS region, builds a
# calibration matrix from every position of it, and takes the sensitivities as
# the eigenvectors of an operator derived from that matrix. ``crop`` sets the
# maps to zero where the eigenvalue falls below it, which keeps the background
# out of the reconstruction:
espirit = bt.ecalib(measured, maps=1, calib_size=24, crop=0.8)

# %%
# **Nonlinear inversion** [#nlinv]_ estimates the sensitivities and the image
# together, from every acquired sample; the ACS region is just part of the
# data. ``maxiter`` counts Gauss-Newton steps, and because the regularization
# shrinks with every step, the count acts as a regularization parameter;
# twelve suit this noise level. You divide its maps by their root sum of
# squares, which is the normalization the other two use. Lesson 4 is about
# this method.
joint, estimated = bt.nlinv(measured, maxiter=12, return_sensitivities=True)

rss_maps = bartorch.rss(estimated, axes=(0,), keepdim=True)
nonlinear = estimated / rss_maps.abs().clamp(min=1e-6)

print(f"caldir {tuple(direct.shape)}")
print(f"ecalib {tuple(espirit.shape)}")
print(f"nlinv  {tuple(nonlinear.shape)}")

# sphinx_gallery_start_ignore
channel = 2
figure, axes = panels(3, bars=1)
for column, (name, maps) in enumerate(
    (("caldir", direct), ("ESPIRiT", espirit), ("nlinv", nonlinear))
):
    domain(axes[0, column], maps[channel, 0], f"{name}, channel {channel}", ceiling=1.0)
phase_bar(figure, axes)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Here is channel 2 from each: brightness is the magnitude and colour the
# phase. The phase of a set of maps is only defined up to a phase common to
# all channels, which each estimator fixes its own way; that common phase
# passes into the phase of the image and leaves its magnitude alone. Up to
# it, the three agree inside the object. Outside it the data say nothing
# about a sensitivity, and the estimators return different values there:
# ``caldir`` divides noise by noise and returns an arbitrary value of unit
# magnitude, ESPIRiT sets the maps to zero, and ``nlinv`` continues the
# smooth function its regularization favours.
#
# SENSE and map quality
# ---------------------
#
# Now you give each set to :func:`bartorch.apps.pics` with the same Tikhonov
# weight and the same number of conjugate-gradient iterations, so the only
# thing that differs is the maps. The image ``nlinv`` returned with its maps
# is a fourth estimate.
reconstructions = {
    name: apps.pics(measured, maps, l2=0.001, maxiter=40)
    for name, maps in (
        ("caldir", direct),
        ("ESPIRiT", espirit),
        ("nlinv", nonlinear),
    )
}

for name, estimate in reconstructions.items():
    print(f"{name:>12}  NRMSE {error(estimate):.3f}")
print(f"{'nlinv image':>12}  NRMSE {error(joint):.3f}")

# sphinx_gallery_start_ignore
figure, axes = panels(3, rows=2)
show(axes[0, 0], reference, "reference", vmax=peak)
show(axes[0, 1], within(zero_filled), "zero-filled", vmax=peak)
show(axes[0, 2], within(joint), "nlinv image", vmax=peak)
for axis, name in zip(axes[1], reconstructions):
    show(axis, within(reconstructions[name]), f"SENSE, {name}", vmax=peak)
plt.show()

figure, axes = panels(3, bars=1)
errors(
    figure,
    axes[0],
    [within(reconstructions[name]) for name in reconstructions],
    reference * support,
    0.03,
)
for axis, name in zip(axes[0], reconstructions):
    axis.set_title(f"SENSE, {name}")
plt.show()
# sphinx_gallery_end_ignore

# %%
# All three sets of maps unfold the three copies, and so does ``nlinv`` on its
# own. The error maps, at 3 % of the image peak, show what is left. The
# direct estimate leaves a faint residual fold at the edges of the phantom,
# where its low-resolution maps are least accurate; ESPIRiT and nonlinear
# inversion leave mostly noise.
#
# ACS size
# --------
#
# Now you keep only eight central lines. ESPIRiT's kernel is six samples wide
# by default, so eight lines leave it three positions along the
# phase-encoding axis, and its calibration matrix has too few rows. ``caldir``
# still runs, on an image of eight lines' resolution. ``nlinv`` uses every
# acquired sample, so the ACS region affects it only through its first
# Gauss-Newton steps.
scarce = acquire(8)

espirit = bt.ecalib(scarce, maps=1, calib_size=8, crop=0.8)
covered = float((espirit.abs() > 0).float().mean())
print(f"ESPIRiT: {covered:.0%} of voxels with a sensitivity")

direct = bt.caldir(scarce, 8)
joint, estimated = bt.nlinv(scarce, maxiter=12, return_sensitivities=True)
rss_maps = bartorch.rss(estimated, axes=(0,), keepdim=True)
nonlinear = estimated / rss_maps.abs().clamp(min=1e-6)

scarce_reconstructions = {
    "caldir": apps.pics(scarce, direct, l2=0.001, maxiter=40),
    "nlinv": apps.pics(scarce, nonlinear, l2=0.001, maxiter=40),
    "nlinv image": joint,
}
for name, estimate in scarce_reconstructions.items():
    print(f"{name:>12}  NRMSE {error(estimate):.3f}")

# sphinx_gallery_start_ignore
figure, axes = panels(3)
show(axes[0, 0], reference, "reference", vmax=peak)
show(axes[0, 1], within(scarce_reconstructions["caldir"]), "SENSE, caldir", vmax=peak)
show(axes[0, 2], within(scarce_reconstructions["nlinv"]), "SENSE, nlinv", vmax=peak)
plt.show()

figure, axes = panels(2, width=0.8 * WIDTH, bars=1)
errors(
    figure,
    axes[0],
    [within(scarce_reconstructions[name]) for name in ("caldir", "nlinv")],
    reference * support,
    0.1,
)
axes[0, 0].set_title("SENSE, caldir")
axes[0, 1].set_title("SENSE, nlinv")
plt.show()
# sphinx_gallery_end_ignore

# %%
# ESPIRiT returns maps that are zero in every voxel, which no reconstruction
# can use. With the direct estimate you see residual aliasing: maps at the
# resolution of eight lines do not follow the coil profiles closely enough to
# unfold three copies, and the folded edges of the skull reappear inside the
# phantom (the error maps are now at 10 % of the peak). The maps from
# nonlinear inversion still unfold the image, because they are fitted to all
# the acquired samples. The same estimator works on non-Cartesian data, where
# there is no Cartesian ACS region at all, as :func:`bartorch.tools.ncalib`,
# which you use in lesson 8 (:doc:`../04-non-cartesian/02-radial-sense`).
#
# These errors are for one phantom, one noise level and one sampling pattern,
# and they depend on how each reconstruction is regularized; they do not rank
# the estimators in general.
#
# Exercise: ACS size
# ------------------
#
# Between eight and 24 lines the outcome changes. Sweep the size of the ACS
# region for the two estimators that use only the ACS block, and compare how
# their errors depend on it:
sizes = (12, 16, 24, 32)
scores = {"caldir": [], "ESPIRiT": []}
for lines in sizes:
    data = acquire(lines)
    maps = {
        "caldir": bt.caldir(data, lines),
        "ESPIRiT": bt.ecalib(data, maps=1, calib_size=lines, crop=0.8),
    }
    for name, estimate in maps.items():
        image = apps.pics(data, estimate, l2=0.001, maxiter=40)
        scores[name].append(error(image))

# sphinx_gallery_start_ignore
for lines, *values in zip(sizes, *scores.values()):
    print(f"{lines:>3} lines: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(scores, values)))

figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 0.42 * WIDTH))
for name, values in scores.items():
    axis.plot(sizes, values, "o-", lw=1.8, label=name)
axis.set_xlabel("ACS lines")
axis.set_ylabel("NRMSE")
axis.set_xticks(sizes)
axis.set_ylim(0, None)
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The ESPIRiT error is close to its 24-line value from twelve lines on: once
# the kernel has enough positions in the ACS region, more lines add little.
# The ``caldir`` error keeps falling as the region grows, because its maps
# are only as sharp as the ACS image they are computed from, and it stays
# above ESPIRiT's at every size here. Try ``crop`` values between 0.5 and 0.95
# for ESPIRiT, or another ``maxiter`` for ``nlinv`` at eight lines, and see
# how the errors move.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With bartorch, simulate 8-channel analytical k-space of a 128x128
#    Shepp-Logan phantom (bt.phantom(coils=8, kspace=True)) with noise of
#    variance 2. Keep every third phase encode plus a fully sampled central
#    ACS block. Estimate sensitivities three ways: bt.caldir, bt.ecalib
#    (one map, crop 0.8) and bt.nlinv (12 Gauss-Newton steps, maps divided
#    by their root sum of squares). Reconstruct with apps.pics (l2=0.001,
#    40 CG iterations) using each set, and report the NRMSE of the magnitude
#    inside the phantom against the noise-free RSS image, after a global
#    scale. Repeat with 24 and with 8 ACS lines.

# %%
#
# References
# ----------
#
# .. [#sense] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: sensitivity
#    encoding for fast MRI. *Magn Reson Med* 42(5):952-962 (1999).
#    https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S
#
# .. [#espirit] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS,
#    Lustig M. ESPIRiT -- an eigenvalue approach to autocalibrating parallel
#    MRI: where SENSE meets GRAPPA. *Magn Reson Med* 71(3):990-1001 (2014).
#    https://doi.org/10.1002/mrm.24751
#
# .. [#nlinv] Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized
#    nonlinear inversion -- joint estimation of coil sensitivities and image
#    content. *Magn Reson Med* 60(3):674-682 (2008).
#    https://doi.org/10.1002/mrm.21691
