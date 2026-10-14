"""
==================================
Parameter maps from scanner images
==================================

In this tour you estimate a :math:`T_2` map from the magnitude images a
scanner exports, without the raw data: you read a multi-echo spin-echo
series from DICOM, fit the decay voxel by voxel, and write the map back as a
DICOM series of the same study and as a NIfTI volume. The geometry and the
echo times pass from the scanner's files through the fit to the output
unchanged, so the map overlays the images it was computed from.

Besides the pixel values, the fit needs the contrast timings and the geometry,
and both are stored in the DICOM headers. Many protocols export each echo of a
multi-echo acquisition, or each inversion time of an inversion-recovery
experiment, as a DICOM series of its own. :func:`bartorch.io.read_dicom` reads
several series as one, sorts the images into contrasts by their echo,
inversion and repetition times, and returns those times with the images: they
are the sampling points of the signal model. The fit is that of lesson 9
without the Fourier encoding: :func:`bartorch.apps.mobafit` runs
Levenberg-Marquardt on the signal equation alone, one voxel at a time.

The noise of a magnitude image is Rician where the signal is small and has a
positive floor, which a decay model fitted to it reads as a longer
:math:`T_2`. With ``magnitude=True``, :func:`~bartorch.apps.mobafit` fits the
magnitude of the model to the magnitude of the data, which takes the phase out
of the problem but not the floor; here the echoes stay above it.

Prerequisites: lesson 9, :doc:`../05-model-based/01-quantitative-models`.

**Learning objectives**

- Read a multi-echo series stored as one DICOM series per echo, with its
  echo times and its voxel-to-world affine.
- Fit :class:`bartorch.nlop.MultiEcho` to magnitude images with
  :func:`bartorch.apps.mobafit`.
- Write the map as a DICOM series of the same study, and as NIfTI, in the
  geometry of the input.
"""

# %%

# sphinx_gallery_start_ignore
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cmap import Colormap

from bartorch import io

WIDTH = 7.8  # inches, the width of the documentation column
# One perceptually uniform colormap per relaxation parameter (Fuderer et al.,
# Magn Reson Med 2025), the one the quantitative-models lesson reads T2 in.
NAVIA = Colormap("crameri:navia").to_matplotlib()

# A transverse slice of concentric tissue compartments -- scalp fat, grey and
# white matter, and ventricles of cerebrospinal fluid -- with the T2 and the
# proton density of each at 3 T, imaged by a CPMG train of eight echoes 10 ms
# apart.  The slice is tilted by 12 degrees about the left-right axis, as a
# transverse slice aligned to the AC-PC line is.
SIZE, ECHO_TIMES = 128, 10.0 * np.arange(1, 9)
y, x = np.mgrid[-1 : 1 : SIZE * 1j, -1 : 1 : SIZE * 1j]
radius = np.hypot(x / 0.78, y / 0.92)
ventricle = np.hypot((np.abs(x) - 0.12) / 0.08, (y + 0.05) / 0.25) < 1
compartments = {  # name: (region, T2 [ms], proton density)
    "fat": ((radius > 0.9) & (radius < 1.0), 45.0, 0.9),
    "grey matter": ((radius > 0.62) & (radius <= 0.9), 85.0, 0.8),
    "white matter": ((radius <= 0.62) & ~ventricle, 70.0, 0.7),
    "CSF": (ventricle & (radius <= 0.62), 250.0, 1.0),
}
T2 = np.zeros((SIZE, SIZE))
M0 = np.zeros((SIZE, SIZE))
for region, t2, density in compartments.values():
    T2[region], M0[region] = t2, density
support = M0 > 0

rng = np.random.default_rng(3)
sigma = 0.01
decay = M0 * np.exp(-ECHO_TIMES[:, None, None] / np.where(support, T2, 1.0))
noisy = decay + sigma * (rng.standard_normal(decay.shape) + 1j * rng.standard_normal(decay.shape))
magnitude = np.abs(noisy) * 1000.0

tilt = np.deg2rad(12.0)
affine = torch.eye(4, dtype=torch.float64)
affine[:3, :3] = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]]
) @ torch.diag(torch.tensor([1.7, 1.7, 4.0], dtype=torch.float64))
affine[:3, 3] = affine[:3, :3] @ torch.tensor(
    [-(SIZE - 1) / 2, -(SIZE - 1) / 2, 0.0], dtype=torch.float64
)

scanner = Path(tempfile.mkdtemp()) / "export"
for number, (te, image) in enumerate(zip(ECHO_TIMES, magnitude), start=101):
    io.write_dicom(
        scanner,
        torch.from_numpy(image.astype(np.float32)),
        affine,
        SeriesNumber=number,
        SeriesDescription=f"cpmg_te{te:.0f}",
        EchoTime=float(te),
        RepetitionTime=3000.0,
        PatientName="Phantom^Brain",
        PatientID="P0001",
    )
# sphinx_gallery_end_ignore

# %%
# DICOM export
# ------------
#
# The input is a scanner export: a CPMG multi-echo spin-echo acquisition of
# eight echoes, 10 ms apart, stored as one DICOM series per echo, numbered 101
# to 108. The slice is a simple head phantom of scalp fat, grey and white
# matter and two ventricles of cerebrospinal fluid, tilted by 12 degrees about
# the left-right axis, as a slice aligned to the AC-PC line is. The code that
# writes the export is hidden; ``scanner`` is its folder.
#
# Series import
# -------------
#
# :func:`~bartorch.io.read_dicom` reads all eight series at once. It returns
# the images as ``(contrasts, slices, rows, columns)`` in order of echo time,
# the echo times themselves in milliseconds, and the affine from voxel
# indices to RAS millimetres. Every echo has to cover the same slices.
import torch

from bartorch import apps, io, nlop

echoes = io.read_dicom(scanner, series=range(101, 109))
te = echoes.timings["echo_time"]

print(f"images {tuple(echoes.image.shape)}, TE {te.tolist()} ms")
voxel = echoes.affine[:3, :3].norm(dim=0)
print("voxel size", " x ".join(f"{v:.1f}" for v in voxel), "mm")

# sphinx_gallery_start_ignore
shown = (0, 3, 7)
figure, axes = plt.subplots(1, len(shown), figsize=(WIDTH, WIDTH / len(shown) + 0.4))
top = float(echoes.image.max())
for axis, echo in zip(axes, shown):
    axis.imshow(echoes.image[echo, 0], cmap="gray", vmin=0.0, vmax=top)
    axis.set_title(f"TE {te[echo]:.0f} ms")
    axis.set_axis_off()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Three of the echoes, on one window. The cerebrospinal fluid in the
# ventricles, with the longest :math:`T_2`, keeps its signal across the train,
# while the scalp fat loses most of it.
#
# Decay fitting
# -------------
#
# Build :class:`bartorch.nlop.MultiEcho` on the echo times the files state
# and on the in-plane shape of the slice. The fit is voxel by voxel, so you
# can drop the slice axis; for a volume, build the model on
# ``(slices, rows, columns)`` instead. DICOM intensities are in arbitrary
# units: the fit scales the echoes to unit peak and returns the amplitude in
# the units of the files. The map is set to zero where the first echo is
# below 5 % of the peak, outside the head.
images = echoes.image[:, 0].to(torch.complex64)
M = nlop.MultiEcho(te.tolist(), tuple(images.shape[1:]))
maps = apps.mobafit(images, M, magnitude=True, T2=80.0)

head = images[0].abs() > 0.05 * images.abs().max()
t2 = torch.where(head, maps["T2"], torch.zeros(()))

for name, (region, truth, _) in compartments.items():
    median = float(t2[torch.from_numpy(region)].median())
    print(f"{name:>13}: T2 {median:6.1f} ms, simulated {truth:5.1f} ms")

# %%
# DICOM output
# ------------
#
# You write the map back to DICOM as a series of the same study: passing the
# first echo's dataset as ``header`` lends the new series its patient, study
# and frame of reference, and you give it a number and a description of its
# own. ``ImageType`` marks it as derived. NIfTI stores the same affine, and
# that is what a registration or segmentation package reads.
import tempfile

from pathlib import Path

output = Path(tempfile.mkdtemp())
io.write_dicom(
    output / "t2-map",
    t2,
    echoes.affine,
    header=echoes.header,
    SeriesNumber=201,
    SeriesDescription="T2 map",
    ImageType=["DERIVED", "PRIMARY", "T2 MAP"],
)
io.write_nifti(output / "t2-map.nii.gz", t2, echoes.affine)

# %%
# Reading the series back confirms the study and the geometry:
written = io.read_dicom(output / "t2-map")
study = written.header.StudyInstanceUID == echoes.header.StudyInstanceUID
shift = float((written.affine - echoes.affine).abs().max())
print(f"patient {written.header.PatientID}, same study {study}")
print(f"largest affine difference {shift:.1e} mm")

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(0.85 * WIDTH, 0.85 * WIDTH / 2 + 0.4))
truth = np.where(support, T2, 0.0)
for axis, values, title in zip(axes, (truth, written.image[0, 0].numpy()), ("simulated", "fitted")):
    handle = axis.imshow(values, cmap=NAVIA, vmin=0.0, vmax=150.0)
    axis.set_title(title)
    axis.set_axis_off()
figure.colorbar(handle, ax=axes, fraction=0.046, label="$T_2$ [ms]")
plt.show()

figure, axis = plt.subplots(figsize=(0.55 * WIDTH, 0.5 * WIDTH))
error = axis.imshow(np.abs(written.image[0, 0].numpy() - truth), cmap="magma", vmin=0.0, vmax=20.0)
axis.set_title("|fitted - simulated|")
axis.set_axis_off()
figure.colorbar(error, ax=axis, fraction=0.046, label="$|\\Delta T_2|$ [ms]")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The map you read back from DICOM, beside the :math:`T_2` the images were
# simulated with, and the difference between the two. The error is largest
# in the cerebrospinal fluid, where eight echoes over 80 ms sample only the
# start of a 250 ms decay.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Estimate a T2 map from a multi-echo spin-echo exported as one DICOM
#    series per echo (series 101-108), with bartorch. Read all series with
#    io.read_dicom, taking the echo times from echoes.timings["echo_time"]
#    and the geometry from echoes.affine. Fit nlop.MultiEcho on those echo
#    times with apps.mobafit(magnitude=True, T2=80.0), and zero the map
#    where the first echo is below 5 % of the peak. Write the map with
#    io.write_dicom as series 201 of the same study (header=echoes.header,
#    ImageType DERIVED) and with io.write_nifti, both in the input affine.
#    Read the DICOM back and check the study UID and the affine.
