"""
===============================
Dynamic golden-angle radial MRI
===============================

This example reconstructs a dynamic contrast-enhanced series from one
continuous golden-angle radial acquisition, cut into frames of thirteen spokes
each. Each frame on its own is undersampled fifteenfold and cannot be
reconstructed; the series can, because consecutive frames are strongly
correlated, and a total-variation penalty along the time axis states that
correlation. The example compares frame-by-frame gridding with this joint
reconstruction on the images and on the time-intensity curve a perfusion
analysis would use.

In a golden-angle acquisition [#winkelmann]_ each spoke is rotated from the
previous one by :math:`180^\\circ / \\phi \\approx 111.25^\\circ`, with
:math:`\\phi` the golden ratio, so that any block of consecutive spokes covers
k-space approximately uniformly, whatever its length and wherever it starts.
The acquisition therefore runs without interruption, and the temporal
resolution is chosen at reconstruction: fewer spokes per frame give a finer
temporal resolution and stronger streak artefacts. Combined with parallel
imaging and a sparsity penalty along time, this is GRASP [#feng]_.

The encoding is that of :doc:`../04-non-cartesian/02-radial-sense` with a frame axis added: the
image is ``(frames, y, x)``, the trajectory indexes frames as well as spokes,
and the coil sensitivities are shared by all frames. The phantom and the coil
sensitivities are built as in :doc:`../01-basics/02-from-kspace-to-image`; the
cell that does it is hidden on this page and present in the script this page
can be downloaded as.

**Prerequisites.** :doc:`../04-non-cartesian/01-trajectories-and-transforms` and
:doc:`../04-non-cartesian/02-radial-sense`.

**Learning objectives**

- Divide a continuous golden-angle acquisition into frames after the fact.
- Build an encoding whose image and trajectory carry a frame axis.
- Regularize along time with a total-variation term over the frame axis.
- Compare frame-by-frame gridding with the joint reconstruction in the
  images, in an x-t profile and in the time-intensity curve of a region.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap

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
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import linop, optim, priors

SIZE = 128
COILS = 8
FRAMES = 16
SPOKES = 13  # per frame

# %%
#
# A contrast-enhanced series
# --------------------------
#
# The phantom is the BrainWeb slice of the course lessons with a contrast
# agent bolus passing through it. A gamma-variate curve describes the
# first-pass concentration over time, and each tissue enhances in proportion
# to its blood volume: strongly in grey matter, weakly in white matter, and not
# at all in cerebrospinal fluid. The series is therefore smooth in time, with
# the same anatomy in every frame, which is the structure the temporal penalty
# exploits.

CLASSES = {"grey matter": ("GM", 0.8), "white matter": ("WM", 0.25)}

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
# A gamma-variate bolus, and the static image enhanced by it where each class
# is vascular.
moment = torch.linspace(0.0, 1.0, FRAMES)
bolus = (moment / 0.25) ** 3 * torch.exp(3.0 - 3.0 * moment / 0.25)
bolus = bolus / bolus.max()

series = image[None].repeat(FRAMES, 1, 1)
for label, weight in CLASSES.values():
    enhancing = (memberships[CLASS[label]] * image).to(torch.complex64)
    series = series + weight * enhancing[None] * bolus[:, None, None]

top = float(series.abs().max())
peak_frame = int(bolus.argmax())
chosen = (0, peak_frame, FRAMES - 1)
figure, axes = panels(3)
for axis, frame in zip(axes[0], chosen):
    show(axis, series[frame], f"frame {frame}", vmax=top)
figure.suptitle("phantom: before, during and after the first pass")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Acquisition
# -----------
#
# ``FRAMES * SPOKES`` spokes are generated as one golden-angle trajectory and
# reshaped, so that the first axis indexes frames and the second the spokes
# within a frame. With :math:`\pi/2 \times 128 \approx 201` spokes needed for
# a fully sampled frame, thirteen spokes undersample each frame by a factor of
# about fifteen. The image varies along the frame axis, so the operator holds
# one NUFFT per frame, all applied inside the same loop over the coils. The
# noise is that of the Cartesian course lessons, of variance :math:`10^{-4}` per
# sample.

trajectory = bt.traj(readout=SIZE, spokes=FRAMES * SPOKES, radial=True, golden=True)
trajectory = trajectory.reshape(FRAMES, SPOKES, SIZE, 3)

# sphinx_gallery_start_ignore
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

A = linop.NoncartesianSense(sensitivities, (FRAMES, SIZE, SIZE), traj=trajectory)
measured = bt.noise(A(series), n=1e-4, s=3)

print(f"{A.ishape} -> {A.oshape}")
print(A.plan)

# %%
#
# ``plan.items`` is the number of frames the encoding carries, each with its
# own transform.
#
# Reconstruction
# --------------
#
# Two reconstructions of the same data. The first treats the frames as
# independent: the adjoint of the encoding applied to density-compensated
# samples, which is the gridding reconstruction of thirteen spokes per frame,
# with the coils combined by the sensitivities. The second solves for the
# whole series at once, with a total-variation penalty along the frame axis:
# the solution is the series that explains all the data and changes least from
# frame to frame. The streak pattern of each frame is different, because each
# frame has different spokes, so it has a large temporal total variation and
# is suppressed, while the anatomy, which is the same in every frame, is not.

weights = torch.linalg.norm(trajectory.real[..., :2], dim=-1).clamp(min=0.25)
gridded = A.H(measured * weights.to(torch.complex64))

data = measured / optim.data_scaling(measured[..., None], A=A)
temporal = optim.ADMM(priors.TotalVariation(axes=(-3,), weight=0.02), maxiter=30)(data, A)

for name, volume in (("gridding", gridded), ("temporal TV", temporal)):
    print(f"{name:>12}  NRMSE over all frames {bt.nrmse(series.abs(), volume, scaled=True):.3f}")

# %%
#
# The frame axis is ``-3``, the axis in front of the two spatial ones. A term
# given ``(-1, -2)`` would penalize the spatial gradient instead, and one given
# all three would penalize both; the axes a term acts on are the whole
# difference between a spatial and a temporal regularizer.

# sphinx_gallery_start_ignore
results = {"gridding": gridded, "temporal TV": temporal}
figure, axes = panels(3)
show(axes[0, 0], series[peak_frame], "reference", vmax=top)
for axis, (name, volume) in zip(axes[0, 1:], results.items()):
    show(axis, scaled(volume, series)[peak_frame], name, vmax=top)
figure.suptitle(f"frame {peak_frame}, peak of the first pass, {SPOKES} spokes")
plt.show()

figure, axes = panels(2, width=0.85 * WIDTH)
errors(
    figure, axes[0], [volume[peak_frame] for volume in results.values()], series[peak_frame], 0.3
)
for axis, name in zip(axes[0], results):
    axis.set_title(name)
figure.suptitle(f"error magnitude, frame {peak_frame}")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# In the gridding reconstruction the streaks of thirteen spokes dominate the
# frame, and only the ventricles and the outline of the head are recognizable.
# The joint reconstruction recovers the anatomy and the enhanced cortex; its
# residual error is small and concentrated at the tissue boundaries, which
# carry the high spatial frequencies each frame samples most sparsely.
#
# An x-t profile, one line of the image plotted against time, shows the time
# axis directly. The line below runs left to right through the ventricles and
# the grey matter on either side.

# sphinx_gallery_start_ignore
ROW = 58
figure, axes = panels(3)
for axis, (name, volume) in zip(
    axes[0],
    (("reference", series.abs()),) + tuple((n, scaled(v, series)) for n, v in results.items()),
):
    show(axis, volume[:, ROW, :], name, vmax=top)
    axis.set_aspect("auto")
axes[0, 0].set_axis_on()
axes[0, 0].set_xticks([])
axes[0, 0].set_ylabel("frame")
axes[0, 0].set_yticks(range(0, FRAMES, 5))
for spine in axes[0, 0].spines.values():
    spine.set_visible(False)
figure.suptitle(f"x-t profile, row {ROW}")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# In the reference profile the grey matter brightens and fades over a few
# frames, while cerebrospinal fluid and the scalp stay constant. Gridding
# shows the same enhancement under a different streak pattern in every frame,
# so the profile changes from one row to the next even where the object does
# not; temporal total variation removes that variation and keeps the time
# course.
#
# Time-intensity curve
# --------------------
#
# A perfusion study reports the signal in a region as a function of time, so
# the reconstructions are compared on that curve too. The region is the grey
# matter, where the enhancement is strongest.

region = memberships[CLASS["GM"]] > 0.6

curves = {
    "reference": series.abs(),
    "gridding": scaled(gridded, series),
    "temporal TV": scaled(temporal, series),
}
truth = curves["reference"][:, region].mean(-1)
for name, volume in curves.items():
    if name == "reference":
        continue
    enhancement = volume[:, region].mean(-1)
    print(f"{name:>12}  curve NRMSE {float((enhancement - truth).norm() / truth.norm()):.3f}")

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 3.6))
for (name, volume), style in zip(curves.items(), ("-", "o-", "s-")):
    axis.plot(range(FRAMES), volume[:, region].mean(-1).cpu().numpy(), style, ms=4, label=name)
axis.set_xlabel("frame")
axis.set_ylabel("mean signal [a.u.]")
axis.set_title("grey matter")
axis.legend()
axis.grid(True, alpha=0.3)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Averaged over the grey matter, the streaks largely cancel, and gridding
# recovers the shape of the curve but not its level: part of the signal of
# each frame is spread into streaks across the field of view, outside the
# region. The joint reconstruction follows the reference curve closely, with
# the peak slightly attenuated and the baseline slightly raised: a temporal
# total-variation penalty flattens a signal change that lasts only a few
# frames more than any other feature, and a larger weight trades more of the
# peak for less noise.

# %%
#
# References
# ----------
#
# .. [#winkelmann] Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An optimal
#    radial profile order based on the Golden Ratio for time-resolved MRI.
#    *IEEE Trans Med Imaging* 26(1):68-76 (2007).
#    https://doi.org/10.1109/TMI.2006.885337
#
# .. [#feng] Feng L, Grimm R, Block KT, Chandarana H, Kim S, Xu J, Axel L, Sodickson DK,
#    Otazo R. Golden-angle radial sparse parallel MRI: combination of
#    compressed sensing, parallel imaging, and golden-angle radial sampling for
#    fast and flexible dynamic volumetric MRI. *Magn Reson Med* 72(3):707-717
#    (2014). https://doi.org/10.1002/mrm.24980
