"""
===============================
Dynamic golden-angle radial MRI
===============================

In this tour you reconstruct a dynamic contrast-enhanced series from one
continuous golden-angle radial acquisition, which you cut into frames of
thirteen spokes each after the scan. On its own, each frame is undersampled
about fifteenfold, and its gridding reconstruction is dominated by streaks.
Consecutive frames, however, show the same anatomy and differ by a slow
change of contrast, so you reconstruct all frames jointly with a penalty on
the temporal variation of the series. You compare frame-by-frame gridding
with that joint reconstruction on the images, on an x-t profile, and on the
time-intensity curve a perfusion analysis would read.

The acquisition uses the golden angle [#winkelmann]_: each spoke is rotated
from the previous one by :math:`180^\\circ / \\phi \\approx 111.25^\\circ`,
with :math:`\\phi` the golden ratio, so any block of consecutive spokes covers
k-space approximately uniformly, independent of its length and starting
point. You therefore acquire without interruption and choose the temporal
resolution at reconstruction: fewer spokes per frame give finer temporal
resolution and stronger streaks. Combined with parallel imaging and a
sparsity penalty along time, this is GRASP [#feng]_.

The encoding is the radial SENSE of lesson 8 with a frame axis added: the
image is ``(frames, y, x)``, the trajectory indexes frames as well as spokes,
and all frames share the coil sensitivities.

Prerequisites: lessons 7 and 8,
:doc:`../04-non-cartesian/01-trajectories-and-transforms` and
:doc:`../04-non-cartesian/02-radial-sense`.

**Learning objectives**

- Cut a continuous golden-angle acquisition into frames after the fact.
- Build an encoding whose image and trajectory carry a frame axis.
- Regularize along time with a total-variation term over the frame axis.
- Compare frame-by-frame gridding with the joint reconstruction on the
  images, on an x-t profile and on the time-intensity curve of a region.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

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


# The phantom: the BrainWeb [#brainweb]_ slice of the course lessons, as a
# T1-weighted spin-echo image with a smooth phase.
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

import torch

SIZE = 128
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)
MARGIN = 0.25

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
CLASS = {entries[label]["Tissue"]: index for index, label in enumerate(TISSUES)}

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
# sphinx_gallery_end_ignore

# %%
# Contrast-enhanced series
# ------------------------
#
# The object is the BrainWeb slice [#brainweb]_ of the course lessons with a
# bolus of contrast agent passing through it. A gamma-variate curve gives the
# first-pass concentration over time, and each tissue enhances in proportion
# to its blood volume: strongly in grey matter, weakly in white matter, not at
# all in cerebrospinal fluid. The series is smooth in time and has the same
# anatomy in every frame; the temporal penalty below relies on both
# properties. The code that builds it is hidden.
FRAMES = 16
SPOKES = 13  # per frame

# Relative enhancement at the peak of the bolus, per tissue class.
CLASSES = {"grey matter": ("GM", 0.8), "white matter": ("WM", 0.25)}

# sphinx_gallery_start_ignore
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
figure.suptitle("the object before, during and after the first pass")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The cortex enhances most at the peak of the first pass, in frame 4, and
# is close to its baseline again in the last frame.
#
# Acquisition
# -----------
#
# You generate ``FRAMES * SPOKES`` spokes as one golden-angle trajectory and
# reshape it, so that the first axis indexes frames and the second the spokes
# within a frame. A fully sampled frame of 128 readout points needs about
# :math:`\pi/2 \times 128 \approx 201` spokes, so thirteen undersample each
# frame about fifteenfold. The coils are BART's analytical eight-channel head
# coil, normalized as in lesson 2, and the noise has the variance of the
# Cartesian lessons, :math:`10^{-4}` per sample.
import torch

import bartorch
import bartorch.tools as bt

from bartorch import linop, optim, priors

COILS = 8

trajectory = bt.traj(
    readout=SIZE,
    spokes=FRAMES * SPOKES,
    radial=True,
    golden=True,
)
trajectory = trajectory.reshape(FRAMES, SPOKES, SIZE, 3)

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities /= bartorch.rss(sensitivities, axes=(0,), keepdim=True)

A = linop.NoncartesianSense(
    sensitivities,
    (FRAMES, SIZE, SIZE),
    traj=trajectory,
)
measured = bt.noise(A(series), n=1e-4, s=3)

print(f"{A.ishape} -> {A.oshape}")
print(A.plan)

# %%
# The image varies along the frame axis, so the operator holds one NUFFT per
# frame and applies them all inside the same loop over the coils;
# ``plan.items`` is the number of frames it carries.
#
# Gridding and regularized reconstruction
# ---------------------------------------
#
# The first treats the frames as independent: the adjoint of the encoding
# applied to density-compensated samples, which is gridding of thirteen
# spokes per frame, with the coils combined by their sensitivities. The
# density compensation is the ramp :math:`|k|`, floored at a quarter of its
# maximum.
#
# The second solves for the whole series at once, with a total-variation
# penalty along the frame axis: the solution is the series consistent with
# all the data that has the least total variation from frame to frame. Each
# frame has different spokes and therefore a different streak pattern, which
# has a large temporal total variation and is suppressed; the anatomy is the
# same in every frame and is retained.
density = torch.linalg.norm(trajectory.real[..., :2], dim=-1).clamp(min=0.25)
gridded = A.H(measured * density.to(torch.complex64))

data = measured / optim.data_scaling(measured[..., None], A=A)
temporal_tv = priors.TotalVariation(axes=(-3,), weight=0.02)
temporal = optim.ADMM(temporal_tv, maxiter=30)(data, A)

for name, volume in (("gridding", gridded), ("temporal TV", temporal)):
    nrmse = bt.nrmse(series.abs(), volume, scaled=True)
    print(f"{name:>12}: NRMSE over all frames {nrmse:.3f}")

# %%
# The frame axis is ``-3``, the one in front of the two spatial axes. Given
# ``(-1, -2)`` the same term would penalize the spatial gradient instead, and
# given all three it would penalize both: the axes a term acts on are the
# whole difference between a spatial and a temporal regularizer.

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
# In the gridded frame the streaks of thirteen spokes dominate, and only the
# ventricles and the outline of the head are recognizable. The joint
# reconstruction recovers the anatomy and the enhanced cortex. Its residual
# error is small and concentrated at the tissue boundaries, which carry
# the high spatial frequencies each frame samples most sparsely.
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
# In the reference profile the grey matter brightens and fades over a few
# frames, while cerebrospinal fluid and the scalp stay constant. Gridding
# shows the same enhancement under a different streak pattern in every frame,
# so the profile varies from frame to frame where the object is constant.
# Temporal total variation removes that variation and retains the time
# course.
#
# Time-intensity curves
# ---------------------
#
# A perfusion study reports the mean signal in a region as a function of
# time, so you compare the two reconstructions on that curve too, in the grey
# matter where the enhancement is strongest.
region = memberships[CLASS["GM"]] > 0.6

curves = {
    "reference": series.abs(),
    "gridding": scaled(gridded, series),
    "temporal TV": scaled(temporal, series),
}
truth = curves["reference"][:, region].mean(-1)
for name in ("gridding", "temporal TV"):
    enhancement = curves[name][:, region].mean(-1)
    error = float((enhancement - truth).norm() / truth.norm())
    print(f"{name:>12}: curve NRMSE {error:.3f}")

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 3.6))
for (name, volume), style in zip(curves.items(), ("-", "o-", "s-")):
    axis.plot(range(FRAMES), volume[:, region].mean(-1).cpu().numpy(), style, ms=4, label=name)
axis.set_xlabel("frame")
axis.set_ylabel("mean signal [a.u.]")
axis.set_title("grey matter")
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
axis.grid(True, alpha=0.3)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Averaged over the grey matter the streaks largely cancel, and gridding
# reproduces the shape of the curve but not its level: part of each frame's
# signal is spread into streaks across the field of view, outside the region.
# The joint reconstruction follows the reference closely, with the peak
# attenuated and the late frames slightly raised. A temporal total-variation
# penalty attenuates a change that lasts only a few frames, and a larger
# weight trades more of the peak for less noise.
#
# Exercise: retrospective frame length
# ------------------------------------
#
# Because the spokes follow the golden angle, the same data can be regrouped
# into longer frames. Join two or four frames into one, grid each, and compare
# with the reference averaged over the same frames:
for join in (1, 2, 4):
    frames = FRAMES // join
    traj_j = trajectory.reshape(frames, join * SPOKES, SIZE, 3)
    data_j = measured.reshape(COILS, frames, join * SPOKES, SIZE)
    A_j = linop.NoncartesianSense(
        sensitivities,
        (frames, SIZE, SIZE),
        traj=traj_j,
    )
    k = torch.linalg.norm(traj_j.real[..., :2], dim=-1).clamp(min=0.25)
    gridded_j = A_j.H(data_j * k.to(torch.complex64))
    reference_j = series.abs().reshape(frames, join, SIZE, SIZE).mean(1)
    nrmse = bt.nrmse(reference_j, gridded_j, scaled=True)
    print(f"{join * SPOKES} spokes, {frames:2d} frames: NRMSE {nrmse:.3f}")

# %%
# The NRMSE falls with the frame length, at the cost of temporal resolution:
# with 52 spokes per frame the rise of the bolus, frames 0 to 4, spans two
# frames. Repeat the comparison with the joint reconstruction, or vary the
# weight of the temporal term and observe the attenuation of the peak.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Reconstruct a dynamic 2D series of 16 frames, 128 x 128, from one
#    golden-angle radial acquisition of 16 * 13 spokes (bt.traj with
#    golden=True, reshaped to (frames, spokes, readout, 3)), 8 coils, with
#    bartorch. Encode with linop.NoncartesianSense over (frames, y, x).
#    Reconstruct (a) frame by frame, as the adjoint of density-compensated
#    data (|k| floored at 0.25), and (b) jointly, with optim.ADMM (30
#    iterations) and priors.TotalVariation over the frame axis (-3), weight
#    0.02, on data divided by optim.data_scaling. Report NRMSE over all
#    frames and of the mean grey-matter time-intensity curve.

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
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
