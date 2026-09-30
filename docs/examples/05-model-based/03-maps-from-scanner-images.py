"""
==================================
Parameter maps from scanner images
==================================

This lesson estimates a :math:`T_2` map from the magnitude images a scanner
exports, without access to the raw data: a multi-echo spin-echo series is read
from DICOM, the decay is fitted voxel by voxel, and the map is written back as
a DICOM series of the same study and as a NIfTI volume. The aim is to show the
geometry and the acquisition timings passing from the scanner's files to the
fit and on to the output unchanged, so that the map overlays the images it was
computed from.

Many protocols export each echo of a multi-echo acquisition, or each
inversion time of an inversion-recovery experiment, as a DICOM series of its
own. :func:`bartorch.io.read_dicom` reads several series as one, sorts the
images into contrasts by their echo, inversion and repetition times, and
returns those times with the images: they are the sampling points of the
signal model. The images and the model are then those of
:doc:`02-quantitative-models`, minus the Fourier encoding: the fit of
:func:`bartorch.apps.mobafit` is Gauss-Newton on the signal equation alone.

A magnitude image is not Gaussian where the signal is small: its noise is
Rician and has a positive floor, which a decay model fitted to it reads as a
longer :math:`T_2`. :func:`~bartorch.apps.mobafit` with ``magnitude=True``
fits the magnitude of the model to the magnitude of the data, which removes
the phase from the problem but not the floor; the echoes here stay above it.

The input series is simulated in a hidden cell and written to a temporary
directory, standing in for an export from the scanner.

**Learning objectives**

- Read a multi-echo series stored as one DICOM series per echo, with its echo
  times and its voxel-to-world affine.
- Fit :class:`bartorch.nlop.MultiEcho` to magnitude images with
  :func:`bartorch.apps.mobafit`.
- Write the map as a DICOM series of the same study, and as NIfTI, in the
  geometry of the input.

It follows :doc:`02-quantitative-models`. The next section,
:doc:`../06-learning/01-plug-and-play`, replaces a specified regularizer with a
learned denoiser.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap

WIDTH = 8.0  # inches, the width of the documentation column
# One perceptually uniform colormap per relaxation parameter (Fuderer et al.,
# Magn Reson Med 2025), the one the previous lesson reads T2 in.
NAVIA = Colormap("crameri:navia").to_matplotlib()
# sphinx_gallery_end_ignore

import tempfile
from pathlib import Path

import numpy as np
import torch

from bartorch import apps, io, nlop

# sphinx_gallery_start_ignore
# A transverse slice of concentric tissue compartments -- scalp fat, grey and
# white matter, and ventricles of cerebrospinal fluid -- with the T2 and the
# proton density of each at 3 T, imaged by a CPMG train of eight echoes 10 ms
# apart.  The slice is tilted by 12 degrees about the left-right axis, as a
# transverse slice aligned to the AC-PC line is.
SIZE, ECHO_TIMES = 128, 10.0 * np.arange(1, 9)
y, x = np.mgrid[-1 : 1 : SIZE * 1j, -1 : 1 : SIZE * 1j]
radius = np.hypot(x / 0.78, y / 0.92)
ventricle = np.hypot((np.abs(x) - 0.12) / 0.08, (y + 0.05) / 0.25) < 1
compartments = [  # (region, T2 [ms], proton density)
    ((radius > 0.9) & (radius < 1.0), 45.0, 0.9),  # fat
    ((radius > 0.62) & (radius <= 0.9), 85.0, 0.8),  # grey matter
    ((radius <= 0.62) & ~ventricle, 70.0, 0.7),  # white matter
    (ventricle & (radius <= 0.62), 250.0, 1.0),  # cerebrospinal fluid
]
T2 = np.zeros((SIZE, SIZE))
M0 = np.zeros((SIZE, SIZE))
for region, t2, density in compartments:
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
#
# Reading the series
# ------------------
#
# The export holds one series per echo, numbered 101 to 108. Asked for all of
# them, :func:`~bartorch.io.read_dicom` returns the images as
# ``(contrasts, slices, rows, columns)`` in order of echo time, the echo times
# themselves in milliseconds, and the affine from voxel indices to RAS
# millimetres; every echo has to cover the same slices.

echoes = io.read_dicom(scanner, series=range(101, 109))
te = echoes.timings["echo_time"]

print(f"images {tuple(echoes.image.shape)}, TE {te.tolist()} ms")
print("voxel size", " x ".join(f"{v:.1f}" for v in echoes.affine[:3, :3].norm(dim=0)), "mm")

# %%
#
# The images of four of the echoes, on one window: the cerebrospinal fluid in
# the ventricles, with the longest :math:`T_2`, keeps its signal across the
# train while the scalp fat loses most of it.

# sphinx_gallery_start_ignore
shown = (0, 2, 4, 7)
figure, axes = plt.subplots(1, len(shown), figsize=(WIDTH, WIDTH / len(shown) + 0.4))
top = float(echoes.image.max())
for axis, echo in zip(axes, shown):
    axis.imshow(echoes.image[echo, 0], cmap="gray", vmin=0.0, vmax=top)
    axis.set_title(f"TE {te[echo]:.0f} ms")
    axis.set_axis_off()
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Fitting the decay
# -----------------
#
# :class:`bartorch.nlop.MultiEcho` is built on the echo times the files state
# and on the in-plane shape of the slice. The fit is voxel by voxel, so the
# slice axis is dropped here; a volume is fitted the same way with the model
# built on ``(slices, rows, columns)``. DICOM stores intensities in arbitrary
# units; the fit scales the echoes to unit peak and returns the amplitude in
# the units of the files.

images = echoes.image[:, 0].to(torch.complex64)
M = nlop.MultiEcho(te.tolist(), tuple(images.shape[1:]))
maps = apps.mobafit(images, M, magnitude=True, T2=80.0)
t2 = torch.where(images[0].abs() > 0.05 * images.abs().max(), maps["T2"], torch.zeros(()))

for (region, truth, _), name in zip(compartments, ("fat", "grey matter", "white matter", "CSF")):
    inner = torch.from_numpy(region)
    print(f"{name:>13}: T2 {float(t2[inner].median()):6.1f} ms against {truth:5.1f} ms")

# %%
#
# Writing the map
# ---------------
#
# The map goes back to DICOM as a series of the same study: the first
# echo's dataset lends its patient, study and frame of reference, and the
# series gets a number and a description of its own. ``ImageType`` marks it as
# derived. NIfTI stores the same affine, which is what a registration or a
# segmentation package reads.

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

written = io.read_dicom(output / "t2-map")
same_study = written.header.StudyInstanceUID == echoes.header.StudyInstanceUID
print(f"patient {written.header.PatientID}, same study {same_study}")
print(f"largest affine difference {float((written.affine - echoes.affine).abs().max()):.1e} mm")

# %%
#
# The map read back from the DICOM series beside the :math:`T_2` the images
# were simulated with, and the difference between them. The error is largest
# in the cerebrospinal fluid, where eight echoes over 80 ms sample only the
# start of a 250 ms decay.

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH / 3 + 0.6))
truth = np.where(support, T2, 0.0)
for axis, values, title in zip(
    axes[:2], (truth, written.image[0, 0].numpy()), ("simulated", "fitted")
):
    handle = axis.imshow(values, cmap=NAVIA, vmin=0.0, vmax=150.0)
    axis.set_title(title)
figure.colorbar(handle, ax=axes[:2], fraction=0.046, label="$T_2$ [ms]")
error = axes[2].imshow(
    np.abs(written.image[0, 0].numpy() - truth), cmap="magma", vmin=0.0, vmax=20.0
)
axes[2].set_title("|fitted - simulated|")
figure.colorbar(error, ax=axes[2], fraction=0.046, label="[ms]")
for axis in axes:
    axis.set_axis_off()
plt.show()
# sphinx_gallery_end_ignore
