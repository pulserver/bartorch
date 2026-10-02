"""
============================
Coil sensitivity calibration
============================

This lesson compares three ways of estimating the receive sensitivities of a
coil array from the undersampled acquisition itself, and shows how the size
of the fully sampled calibration region decides which of them can be used.
A SENSE reconstruction [#sense]_ inverts

.. math::

   y_c = P F (S_c \\, x),

and is only as accurate as the sensitivities :math:`S_c` it is given: an
error in :math:`S_c` appears in the image as residual aliasing or as shading,
however good the solver. In clinical practice the sensitivities are estimated
either from a separate low-resolution prescan or, as here, by
autocalibration, from a fully sampled block of lines at the centre of
k-space, the autocalibration signal (ACS) region. The three estimators
differ in what they take from the data:

- :func:`bartorch.tools.caldir` divides each low-resolution coil image,
  reconstructed from the ACS region alone, by their root sum of squares;
- :func:`bartorch.tools.ecalib` (ESPIRiT [#espirit]_) takes the sensitivities
  as the eigenvectors of an operator built from the k-space neighbourhoods of
  the ACS region;
- :func:`bartorch.tools.nlinv` (nonlinear inversion [#nlinv]_) estimates the
  sensitivities and the image jointly from all acquired samples, with the ACS
  region only as part of the data.

The acquisition is regularly undersampled by :math:`R = 3`, which folds the
object into three overlapping copies along the phase-encoding direction. With
a generous ACS region all three estimators unfold it; with a small one, the
direct estimate is too coarse to separate the copies and ESPIRiT has too few
kernel positions to calibrate at all.

**Learning objectives**

- Estimate coil sensitivities with ``caldir``, ``ecalib`` and ``nlinv``, and
  use each in :func:`bartorch.apps.pics`.
- Evaluate a reconstruction against a noise-free reference within the object
  support, by an error map and the NRMSE.
- State why ESPIRiT requires an ACS region larger than its kernel, and why
  nonlinear inversion does not.

The previous lesson, :doc:`../01-basics/02-from-kspace-to-image`, used ESPIRiT
with a 24-line ACS region. The next lesson, :doc:`02-nonlinear-inversion`,
writes nonlinear inversion out as a nonlinear operator and a Gauss-Newton
solver.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap
from matplotlib.colors import ListedColormap

WIDTH = 7.8  # inches, the width of the documentation column at 110 dpi

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
import numpy as np
import torch

import bartorch
import bartorch.tools as bt
from bartorch import apps

SIZE = 128
COILS = 8
ACCELERATION = 3

# %%
#
# Data
# ----
#
# The k-space is BART's analytical Shepp-Logan phantom seen through its
# analytical head coil, evaluated in k-space rather than transformed from a
# sampled image (:doc:`../01-basics/01-tensors-and-commands`), so the
# reconstructions below do not share the discretization of the simulation.
# Complex Gaussian noise of variance 2 is added to every sample.
#
# The reference is the root sum of squares of the fully sampled, noise-free
# coil images. Every error is the NRMSE of a magnitude image within the
# phantom's support, after a least-squares fit of a global scale, since the
# three estimators normalize the sensitivities, and hence the image, in
# different ways.

clean = bt.phantom(SIZE, coils=COILS, kspace=True)
kspace = bt.noise(clean, n=2.0, s=3)

reference = bartorch.rss(bartorch.ifft(clean, axes=(-2, -1)), axes=(0,))[0].abs()
support = (bt.phantom(SIZE).abs() > 0).to(torch.float32)


def error(estimate):
    """NRMSE of a magnitude image within the phantom's support."""
    return bt.nrmse(reference * support, estimate.abs().squeeze() * support, scaled=True)


# %%
#
# Every third phase encode is acquired, and a central block of ``calibration``
# lines is acquired in full. Regular undersampling by :math:`R` replicates the
# point spread function :math:`R` times across the field of view, so each
# voxel of the zero-filled image is the sum of three voxels a third of the
# field of view apart. Only the sensitivities, which differ between those
# voxels, can separate them.

encodes = torch.arange(SIZE) - SIZE // 2


def acquire(calibration):
    """The regularly undersampled k-space with a fully sampled central block."""
    lines = (encodes % ACCELERATION == 0) | (encodes.abs() < calibration // 2)
    return kspace * lines.to(torch.complex64).reshape(SIZE, 1)


measured = acquire(24)
print(f"{float(bt.pattern(measured).real.mean()):.0%} of k-space acquired")

zero_filled = bartorch.rss(bartorch.ifft(measured, axes=(-2, -1)), axes=(0,))[0]

# %%
#
# Three calibrations
# ------------------
#
# ESPIRiT's ``crop`` sets the sensitivities to zero where the eigenvalue of the
# calibration operator falls below it, which removes the background from the
# reconstruction. ``nlinv`` counts Gauss-Newton steps, and its regularization
# decreases with every step, so the count acts as a regularization parameter;
# twelve steps suit this noise level. Its sensitivities are normalized here to
# unit root sum of squares, the normalization the other two estimators use.

direct = bt.caldir(measured, 24)
espirit = bt.ecalib(measured, maps=1, calib_size=24, crop=0.8)
joint, estimated = bt.nlinv(measured, maxiter=12, return_sensitivities=True)
nonlinear = estimated / bartorch.rss(estimated, axes=(0,), keepdim=True).abs().clamp(min=1e-6)

print(
    f"caldir {tuple(direct.shape)}, ecalib {tuple(espirit.shape)}, nlinv {tuple(nonlinear.shape)}"
)

# %%

# sphinx_gallery_start_ignore
channel = 2
figure, axes = panels(3, rows=2, bars=1)
for column, (name, maps) in enumerate(
    (("caldir", direct), ("ESPIRiT", espirit), ("nlinv", nonlinear))
):
    handle = show(axes[0, column], maps[channel, 0], f"{name}, channel {channel}", vmax=1.0)
    show(axes[1, column], maps[channel, 0].angle().numpy(), cmap=PHASE, vmin=-np.pi, vmax=np.pi)
scalebar(figure, axes[0, :], handle, "|sensitivity|")
phase_bar(figure, axes[1, :])
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The figure shows the three estimates of one channel, magnitude above and
# phase below. The phase of a sensitivity map is determined only up to a phase
# common to all channels, which each estimator fixes differently; that common phase
# passes into the phase of the reconstructed image and leaves its magnitude
# unchanged. Up to it, the three estimates agree inside the object. They
# differ outside it, where the data do not determine a sensitivity:
# ``caldir`` divides noise by noise there and returns an arbitrary unit-modulus
# value, ESPIRiT sets the maps to zero, and ``nlinv`` extrapolates the smooth
# function its regularization favours.
#
# Each set of sensitivities is given to :func:`bartorch.apps.pics` with the
# same Tikhonov weight and number of conjugate-gradient iterations, so that
# the reconstructions differ only in the sensitivities. The image ``nlinv``
# returns jointly with its sensitivities is a fourth estimate.

reconstructions = {
    name: apps.pics(measured, maps, l2=0.001, maxiter=40)
    for name, maps in (("caldir", direct), ("ESPIRiT", espirit), ("nlinv", nonlinear))
}

for name, estimate in reconstructions.items():
    print(f"{name:>12}  NRMSE {error(estimate):.3f}")
print(f"{'nlinv image':>12}  NRMSE {error(joint):.3f}")

# %%

# sphinx_gallery_start_ignore
# The skull is the brightest structure; a window at half its intensity shows
# the aliasing inside the phantom.
peak = 0.5 * float(reference.max())


def within(estimate):
    """A magnitude image scaled to the reference, inside the support."""
    return scaled(estimate.squeeze(), reference) * support


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
#
# The images are windowed at half the intensity of the skull, which
# saturates. The zero-filled image shows the three overlapping copies of the
# phantom that regular undersampling produces. All three calibrations unfold
# them, and so does the image ``nlinv`` returns with its sensitivities.
# The error maps, at 3 % of the image peak, show the remaining
# differences: the direct estimate leaves a faint residual fold at the edges
# of the phantom, where its low-resolution sensitivities are least accurate,
# and ESPIRiT and nonlinear inversion leave mostly noise.
#
# A smaller calibration region
# ----------------------------
#
# ESPIRiT builds its calibration matrix from every position of a kernel, six
# samples wide by default, inside the ACS region. Eight lines leave three
# kernel positions along the phase-encoding axis, and from this data
# ``ecalib`` returns sensitivities that are zero in every voxel, which no
# reconstruction can use. ``caldir`` still runs, on an image of eight lines'
# resolution. ``nlinv`` uses every acquired sample, so the calibration region
# affects it only through the first Gauss-Newton steps.

scarce = acquire(8)

espirit = bt.ecalib(scarce, maps=1, calib_size=8, crop=0.8)
print(f"ESPIRiT: {float((espirit.abs() > 0).float().mean()):.0%} of voxels with a sensitivity")

direct = bt.caldir(scarce, 8)
joint, estimated = bt.nlinv(scarce, maxiter=12, return_sensitivities=True)
nonlinear = estimated / bartorch.rss(estimated, axes=(0,), keepdim=True).abs().clamp(min=1e-6)

scarce_reconstructions = {
    "caldir": apps.pics(scarce, direct, l2=0.001, maxiter=40),
    "nlinv": apps.pics(scarce, nonlinear, l2=0.001, maxiter=40),
    "nlinv image": joint,
}
for name, estimate in scarce_reconstructions.items():
    print(f"{name:>12}  NRMSE {error(estimate):.3f}")

# %%

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
#
# With eight ACS lines the direct estimate leaves visible residual aliasing:
# sensitivities estimated at a resolution of eight lines do not represent the
# coil profiles closely enough to unfold three copies, and the fold-over
# edges of the skull reappear inside the phantom. The sensitivities from
# nonlinear inversion still unfold the image, because they are fitted to all
# acquired samples. The same estimator applies to non-Cartesian data, where
# no Cartesian ACS region exists, as :func:`bartorch.tools.ncalib`, which
# :doc:`../04-non-cartesian/02-radial-sense` uses.
#
# The errors above are for one phantom, one noise level and one sampling
# pattern, and they depend on the regularization of each reconstruction; they
# do not rank the estimators in general.

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
