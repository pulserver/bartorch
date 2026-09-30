"""
===================================
Susceptibility distortion in EPI
===================================

In an echo-planar image, the phase-encoding direction is sampled at the echo
spacing rather than the dwell time, so its bandwidth per pixel is a few tens of
hertz. A spin off resonance by :math:`\\Delta f` is displaced along the
phase-encoding axis by :math:`\\Delta f` divided by that bandwidth, which near
the frontal sinus and the petrous bone amounts to several millimetres at 3 T:
the orbitofrontal cortex and the temporal poles are compressed or stretched,
and signal piles up where neighbouring voxels are displaced onto the same
location.

The displacement changes sign with the direction in which k-space is
traversed. Two acquisitions with opposite phase-encoding polarity,
*blip-up* and *blip-down*, are distorted in opposite directions, and the
displacement field that brings them into register is the correction
[#andersson]_. This example corrects such a pair, measured at 3 T and
published on OpenNeuro [#ds001600]_, with
:func:`bartorch.tools.correct_susceptibility`, which runs PyHySCO [#pyhysco]_,
and compares the estimated displacement with the one predicted by a
gradient-echo field map of the same subject.

**Learning objectives**

* Read a BIDS EPI series and its sidecar, and compute the bandwidth per pixel
  along phase encoding and the displacement per hertz of off-resonance.
* State the direction of the displacement in anatomical terms from the
  phase-encoding direction and the image orientation.
* Recognise the compression, stretching and signal pile-up of susceptibility
  distortion, and their reversal between the two phase-encoding polarities.
* Estimate the displacement field from a reversed phase-encoding pair and
  correct both images, including their intensity.
* Assess the estimated displacement against an independent field map, and the
  correction by the agreement of the two corrected images.
"""

# %%

# sphinx_gallery_start_ignore
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage

SOURCE = "https://s3.amazonaws.com/openneuro.org/ds001600/"
CACHE = Path.home() / ".cache" / "bartorch-examples" / "ds001600"
FILES = (
    "sub-1/func/sub-1_task-rest_acq-AP_bold",
    "sub-1/fmap/sub-1_dir-PA_epi",
    "sub-1/fmap/sub-1_acq-v4_phasediff",
    "sub-1/fmap/sub-1_acq-v4_magnitude1",
)
for name in FILES:
    for suffix in (".nii.gz", ".json"):
        path = CACHE / (name + suffix)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(SOURCE + name + suffix, path.with_suffix(".part"))
            path.with_suffix(".part").rename(path)
# sphinx_gallery_end_ignore
import json

import torch
from nibabel.orientations import aff2axcodes

import bartorch.tools as bt
from bartorch.io import read_nifti

# %%
#
# The data
# --------
#
# The dataset holds gradient-echo EPI series of one subject acquired on a
# Siemens Prisma at 3 T, 64 x 64 matrix, 44 axial slices, 3.75 x 3.75 x 4 mm,
# with the same protocol and shim and opposite phase-encoding polarity, and a
# dual-echo gradient-echo field map on the same grid. Each EPI series has five
# volumes; their mean is taken to raise the signal-to-noise ratio.
# :func:`~bartorch.io.read_nifti` returns the volumes as ``(volumes, z, y,
# x)`` and the affine from voxel indices ``(x, y, z)`` to RAS millimetres;
# the sidecars hold the timings of the readout.

blip_up_file = CACHE / "sub-1/fmap/sub-1_dir-PA_epi"
blip_down_file = CACHE / "sub-1/func/sub-1_task-rest_acq-AP_bold"
blip_up = read_nifti(f"{blip_up_file}.nii.gz")
blip_down = read_nifti(f"{blip_down_file}.nii.gz")
sidecar_up = json.loads(Path(f"{blip_up_file}.json").read_text())
sidecar_down = json.loads(Path(f"{blip_down_file}.json").read_text())

affine = blip_up.affine
voxel_mm = tuple(float(v) for v in affine[:3, :3].norm(dim=0).flip(0))  # (z, y, x)
print(f"{tuple(blip_up.image.shape)} volumes, voxel (z, y, x) {voxel_mm} mm")
print("voxel axes x, y, z point to", "".join(aff2axcodes(affine.numpy())))
for label, sidecar in (("blip-up", sidecar_up), ("blip-down", sidecar_down)):
    print(f"{label:9s} {sidecar['SeriesDescription']}: {sidecar['PhaseEncodingDirection']}")

up = blip_up.image.mean(0)
down = blip_down.image.mean(0)

# %%
#
# The voxel axis ``x`` points to the subject's left and ``y`` to anterior, so
# ``PhaseEncodingDirection`` ``j`` is posterior to anterior and ``j-`` anterior
# to posterior: phase encoding is along the anterior-posterior axis of the
# head and the readout left-right. The figures show the slices with anterior
# at the top and the subject's left on the right.
#
# Displacement along the phase-encoding axis
# ------------------------------------------
#
# With :math:`N` phase-encoding lines acquired at the effective echo spacing
# :math:`\Delta t_{esp}` (the echo spacing divided by any parallel-imaging
# acceleration), the phase accrued off resonance is linear in :math:`k_y`,
# which is a displacement by
#
# .. math::
#
#    d = \frac{\Delta f}{\mathrm{BW}_{PE}}\ \text{voxels}, \qquad
#    \mathrm{BW}_{PE} = \frac{1}{N\, \Delta t_{esp}},
#
# with :math:`\mathrm{BW}_{PE}` the bandwidth per pixel along the
# phase-encoding axis; BIDS writes :math:`(N - 1)\,\Delta t_{esp}` as
# ``TotalReadoutTime``. The displacement is along ``PhaseEncodingDirection``
# for a positive :math:`\Delta f`, and reversing the blips reverses it. The
# readout, at a bandwidth per pixel of ``PixelBandwidth``, is displaced by a
# fraction of a voxel, which is neglected.

lines = sidecar_up["AcquisitionMatrixPE"]
echo_spacing = sidecar_up["EffectiveEchoSpacing"]
bandwidth_pe = 1 / (lines * echo_spacing)
print(f"{lines} lines at an effective echo spacing of {1e3 * echo_spacing:.2f} ms")
print(
    f"bandwidth per pixel: {bandwidth_pe:.1f} Hz along phase encoding "
    f"(sidecar {sidecar_up['BandwidthPerPixelPhaseEncode']} Hz), "
    f"{sidecar_up['PixelBandwidth']} Hz along the readout"
)
print(f"total readout time {1e3 * sidecar_up['TotalReadoutTime']:.2f} ms")
print(
    f"100 Hz off resonance: {100 / bandwidth_pe:.1f} voxels = "
    f"{100 / bandwidth_pe * voxel_mm[1]:.1f} mm along phase encoding"
)

# %%
#
# Reference field map
# -------------------
#
# The gradient-echo field map is the phase difference :math:`\Delta\phi`
# between two echoes, stored by the scanner as integers from 0 to 4095 for
# :math:`-\pi` to :math:`\pi`, and the off-resonance is
# :math:`\Delta f = \Delta\phi / (2\pi\, \Delta\mathrm{TE})`. The phase is
# smoothed as a complex exponential over one voxel. The displacement it
# predicts for the blip-up image is
# :math:`\Delta f / \mathrm{BW}_{PE}` voxels towards anterior.

phase_file = CACHE / "sub-1/fmap/sub-1_acq-v4_phasediff"
sidecar_phase = json.loads(Path(f"{phase_file}.json").read_text())
phase = (read_nifti(f"{phase_file}.nii.gz").image[0] - 2048) / 2048 * torch.pi
magnitude = read_nifti(str(CACHE / "sub-1/fmap/sub-1_acq-v4_magnitude1.nii.gz")).image[0]
delta_te = sidecar_phase["EchoTime2"] - sidecar_phase["EchoTime1"]
# sphinx_gallery_start_ignore
phasor = magnitude * torch.polar(torch.ones_like(phase), phase)
phasor = torch.complex(
    *(
        torch.as_tensor(ndimage.gaussian_filter(part.numpy(), 1.0))
        for part in (phasor.real, phasor.imag)
    )
)
phase = phasor.angle()
# sphinx_gallery_end_ignore
off_resonance = phase / (2 * torch.pi * delta_te)
predicted_mm = off_resonance / bandwidth_pe * voxel_mm[1]
print(
    f"echo times {1e3 * sidecar_phase['EchoTime1']:.2f} and "
    f"{1e3 * sidecar_phase['EchoTime2']:.2f} ms: unambiguous within "
    f"+/-{1 / (2 * delta_te):.0f} Hz"
)

# %%
#
# Correction from the reversed pair
# ---------------------------------
#
# PyHySCO estimates the displacement field :math:`b` for which the blip-up
# image sampled at :math:`y + b` and the blip-down image sampled at
# :math:`y - b`, each multiplied by the Jacobian determinant of its
# transformation, :math:`1 \pm \partial b / \partial y`, agree. The Jacobian
# factor restores the intensity of voxels compressed into a pile-up or
# stretched over several voxels. A smoothness penalty on :math:`b` and a
# constraint that keeps both transformations invertible regularize the
# problem. The estimation is three-dimensional: the volume is passed with its
# voxel size and the phase-encoding axis ``y``, and :math:`b` is returned in
# millimetres along the voxel axis ``y``, that is towards anterior, on the
# faces between voxels.

result = bt.correct_susceptibility(up, down, voxel_size=voxel_mm, phase_encoding_axis=1)
estimated_mm = 0.5 * (result.field_map[:, 1:] + result.field_map[:, :-1]).float()  # voxel centres
corrected_up, corrected_down = result.blip_up.float(), result.blip_down.float()
# sphinx_gallery_start_ignore
corrected = 0.5 * (corrected_up + corrected_down)
signal = corrected > 0.15 * corrected.quantile(0.99)
signal = ndimage.binary_fill_holes(ndimage.binary_opening(signal.numpy(), iterations=1))
labels, count = ndimage.label(signal)
signal = labels == 1 + np.argmax(ndimage.sum(signal, labels, range(1, count + 1)))
brain = torch.as_tensor(
    ndimage.binary_erosion(signal, iterations=1) & (magnitude > 0.1 * magnitude.max()).numpy()
)
# sphinx_gallery_end_ignore

# %%
#
# The corrected pair
# ------------------
#
# Where the correction is right, the two corrected images are the same image.
# Their agreement is measured over the brain, where the corrected EPI and the
# field map's magnitude both have signal, as the correlation coefficient and
# as the root-mean-square difference relative to the mean image.


def similarity(first, second, region=brain):
    x, y = first[region].double(), second[region].double()
    correlation = torch.corrcoef(torch.stack((x, y)))[0, 1].item()
    return correlation, ((x - y).norm() / (0.5 * (x + y)).norm()).item()


for label, pair in (("acquired", (up, down)), ("corrected", (corrected_up, corrected_down))):
    correlation, difference = similarity(*pair)
    print(
        f"blip-up vs blip-down, {label:9s}: correlation {correlation:.3f}, "
        f"RMS difference {100 * difference:.0f} % of the mean image"
    )

# %%

# sphinx_gallery_start_ignore
ORBITOFRONTAL, TEMPORAL = 21, 14
ANTERIOR = (slice(4, 30), slice(12, 52))


def view(volume, index):
    """Axial slice with anterior at the top."""
    return volume[index].flip(0)


def show(axis, values, title, vmin=0.0, vmax=1.0, cmap="gray"):
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_axis_off()
    return handle


peak = float(corrected[brain].quantile(0.95))
nan = torch.tensor(float("nan"))


def pair_figure(index):
    rows, cols = ANTERIOR
    panels = ((up, "blip-up, P to A"), (down, "blip-down, A to P"), (corrected, "corrected"))
    figure, axes = plt.subplots(2, 3, figsize=(7.8, 5.2), height_ratios=(64, 26 * 64 / 40))
    crop = float(view(corrected, index)[rows, cols].quantile(0.99))
    for column, (volume, title) in enumerate(panels):
        image = view(volume, index)
        show(axes[0, column], image / peak, title)
        axes[0, column].add_patch(
            Rectangle(
                (cols.start - 0.5, rows.start - 0.5),
                cols.stop - cols.start,
                rows.stop - rows.start,
                fill=False,
                edgecolor="#e8a33d",
                linewidth=1.5,
            )
        )
        show(axes[1, column], image[rows, cols] / crop, "")
    return figure


def overlay(first, second, index):
    """Blip-up in magenta and blip-down in green: grey where they agree."""
    a = (view(first, index) / peak).clamp(0, 1)
    b = (view(second, index) / peak).clamp(0, 1)
    return torch.stack((a, b, a), -1)


pair_figure(ORBITOFRONTAL)
plt.show()
pair_figure(TEMPORAL)
plt.show()

figure, axes = plt.subplots(2, 2, figsize=(7.2, 7.4))
for row, index in enumerate((ORBITOFRONTAL, TEMPORAL)):
    for column, (pair, title) in enumerate(
        (((up, down), "acquired"), ((corrected_up, corrected_down), "corrected"))
    ):
        axes[row, column].imshow(overlay(*pair, index))
        axes[row, column].set_title(f"{title}, slice {index}")
        axes[row, column].set_axis_off()
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Slice 21 passes through the orbitofrontal cortex and slice 14 through the
# temporal poles and the cerebellum; the lower row of each figure enlarges the
# anterior part of the slice, with its own grey scale. Above the frontal sinus the off-resonance is
# positive: the blip-up image, phase-encoded from posterior to anterior,
# displaces the orbitofrontal cortex anteriorly and compresses its anterior
# edge into a bright rim, and the blip-down image displaces it posteriorly
# and flattens it. At the temporal poles, above the petrous bone and the
# mastoid air cells, the off-resonance is negative and the directions are
# exchanged: the blip-down image stretches the poles anteriorly into streaks,
# and the blip-up image compresses them into a few dark voxels. The corrected
# image, the mean of the two corrected images of the pair, places the cortex
# between the two and restores its intensity.
#
# In the overlays the blip-up image is shown in magenta and the blip-down
# image in green, so that tissue where the two agree is grey. Before
# correction the frontal, temporal and occipital edges carry a magenta fringe
# on one side and a green fringe on the other; after correction they are
# grey. What remains is green at the temporal poles, where the blip-up image
# has lost the signal that the blip-down image has: signal lost within a
# voxel in one acquisition is not restored by a displacement.
#
# Comparison with the field map
# -----------------------------
#
# The estimate is compared with the displacement predicted by the
# gradient-echo field map over the brain, and over the brain voxels where the
# field map predicts a displacement of more than one voxel,
# :math:`|\Delta f| > \mathrm{BW}_{PE}`: where there is a distortion to
# measure. Over the rest of the brain the field is close to zero, and the
# correlation there measures noise. The field map was acquired with its own
# shim, so the two fields may also differ by a smooth field of first and
# second order.


def correlation(first, second, region):
    return torch.corrcoef(torch.stack((first[region], second[region])).double())[0, 1].item()


distorted = brain & (off_resonance.abs() > bandwidth_pe)
print("shim, EPI      ", sidecar_up["ShimSetting"])
print("shim, field map", sidecar_phase["ShimSetting"])
print(
    f"field over the brain: {off_resonance[brain].quantile(0.01):+.0f} to "
    f"{off_resonance[brain].quantile(0.99):+.0f} Hz (1st to 99th percentile)"
)
print(
    f"displaced by more than one voxel: {100 * distorted.sum() / brain.sum():.0f} % "
    "of the brain voxels"
)
for label, region in (("brain", brain), ("displaced voxels", distorted)):
    slope = torch.linalg.lstsq(
        torch.stack((predicted_mm[region], torch.ones(int(region.sum()))), 1),
        estimated_mm[region, None],
    ).solution[0, 0]
    print(
        f"estimate vs field map over the {label}: correlation "
        f"{correlation(estimated_mm, predicted_mm, region):.2f}, slope {slope:.2f}"
    )

# %%

# sphinx_gallery_start_ignore
LIMIT = 8.0
figure, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
for row, index in enumerate((ORBITOFRONTAL, TEMPORAL)):
    mask = view(brain, index)
    for column, (volume, title) in enumerate(
        ((estimated_mm, "PyHySCO estimate"), (predicted_mm, "GRE field map"))
    ):
        handle = show(
            axes[row, column],
            torch.where(mask, view(volume, index), nan),
            f"{title}, slice {index}",
            -LIMIT,
            LIMIT,
            "PuOr_r",
        )
figure.colorbar(handle, ax=axes, fraction=0.05, label="displacement towards anterior [mm]")
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(7.8, 4.2), sharex=True, sharey=True)
for axis, (region, title) in zip(axes, ((brain, "brain"), (distorted, "displaced > 1 voxel"))):
    axis.hist2d(
        predicted_mm[region].numpy(),
        estimated_mm[region].numpy(),
        bins=48,
        range=((-12, 12), (-12, 12)),
        cmap="magma_r",
        norm="log",
    )
    axis.plot([-12, 12], [-12, 12], color="#3dbde8", linewidth=1.0)
    axis.set_xlabel("field map [mm]")
    axis.set_aspect("equal")
    axis.set_title(title)
axes[0].set_ylabel("PyHySCO estimate [mm]")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The estimate follows the pattern of the field-map prediction: anterior
# displacement above the frontal sinus and in the cerebellum, posterior
# displacement at the temporal poles. It is smaller in amplitude, with a
# slope below one. Two likely causes, neither established here, are the
# smoothness penalty on the displacement field, which at 3.75 mm voxels
# flattens a displacement that changes within a few voxels, and the difference
# between the shims of the two acquisitions. Neither EPI image has signal in
# the voids above the sinus, where the field map predicts the largest
# displacement, so the estimate there is an extrapolation by the smoothness
# penalty. In practice the pair is acquired as two short series
# of a few volumes, the displacement field is estimated once, and it is
# applied to every volume of the functional or diffusion series acquired with
# one of the two polarities.
#
# References
# ----------
#
# .. [#andersson] Andersson JLR, Skare S, Ashburner J. How to correct
#    susceptibility distortions in spin-echo echo-planar images: application
#    to diffusion tensor imaging. *NeuroImage* 20(2):870-888 (2003).
#    https://doi.org/10.1016/S1053-8119(03)00336-7
#
# .. [#pyhysco] Julian A, Ruthotto L. PyHySCO: GPU-enabled susceptibility
#    artifact distortion correction in seconds. *Front Neurosci* (2024).
#
# .. [#ds001600] Cieslak M, Elliott M, Satterthwaite T. Example Fieldmaps.
#    OpenNeuro, accession ds001600. https://openneuro.org/datasets/ds001600.
#    Licensed under CC-BY-SA.
