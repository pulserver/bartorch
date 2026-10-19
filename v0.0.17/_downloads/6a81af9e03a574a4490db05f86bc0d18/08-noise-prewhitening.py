"""
==================
Noise prewhitening
==================

In this tour you measure how much signal-to-noise ratio (SNR) a SENSE
reconstruction loses when the noise of the receive channels is correlated,
and how much of it you get back by prewhitening with a noise-only
acquisition.

The thermal noise of a receive array is correlated
between channels, through mutual inductance between the coil elements and
noise sources shared in the sample, and its level differs from channel to
channel, with the elements' size, loading and preamplifier gain. Least
squares is the maximum-likelihood estimate only when the noise is white,
that is when the channel noise covariance :math:`\\Psi` is proportional to
the identity. A reconstruction that ignores :math:`\\Psi` weights every
channel equally, and does not combine them with the best SNR [#roemer]_.
Prewhitening multiplies the data by a matrix :math:`W` with
:math:`W \\Psi W^H = I`, which makes the channel noise white; you then
calibrate the sensitivities on the whitened channels and run SENSE on them
unchanged [#pruessmann]_. You estimate :math:`\\Psi` from a noise scan, an
acquisition with the RF transmitter off that most vendors run before every
protocol.

You measure the SNR by the pseudo-replica method [#robson]_: repeat the
same reconstruction on independent noise realisations added to one
noise-free acquisition, and take the standard deviation across the
repetitions as the noise of each voxel. That is how you get SNR and
g-factor maps for an iterative reconstruction, for which there is no closed
form of the noise propagation.

Prerequisites: lessons 2 and 3,
:doc:`../01-basics/02-from-kspace-to-image` and
:doc:`../02-parallel-imaging/01-coil-calibration`.

**Learning objectives**

- Estimate the channel noise covariance from a noise scan and whiten the
  data with :func:`bartorch.tools.whiten`.
- Measure an SNR map by the pseudo-replica method.
- Quantify the SNR gain that prewhitening gives a SENSE reconstruction.
"""

# %%

# sphinx_gallery_start_ignore
import csv
from pathlib import Path

import brainweb_dl
import matplotlib.pyplot as plt
import numpy as np
from brainweb_dl import get_mri

import torch

WIDTH = 7.8  # inches, the width of the documentation column at 110 dpi
SIZE = 128


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


def channel_covariance(samples):
    """Sample covariance of the channels, over every other axis."""
    flat = samples.reshape(COILS, -1)
    return (flat @ flat.conj().T) / flat.shape[1]


def off_diagonal(matrix):
    """Mean off-diagonal magnitude relative to the mean diagonal entry."""
    diagonal = torch.diagonal(matrix).abs()
    return float((matrix.abs().sum() - diagonal.sum()) / (COILS * (COILS - 1)) / diagonal.mean())


# The phantom, as lesson 2 builds it: a BrainWeb slice rendered as a
# T1-weighted spin echo (TR 600 ms, TE 12 ms) with a smooth phase.
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
white_matter = memberships[CLASS["WM"]] > 0.8
# sphinx_gallery_end_ignore

# %%
# Correlated channel noise
# ------------------------
#
# Your object is the BrainWeb slice of lesson 2, seen by BART's analytical
# eight-channel head coil; how it is built is hidden on this page. The noise
# of your simulated array couples the channels by a correlation coefficient
# that falls as :math:`0.5^{|i-j|}` with the distance between their indices,
# and the channels' noise standard deviations range from 0.6 to 1.6 times a
# common level.
import torch

import bartorch
import bartorch.tools as bt

from bartorch import apps

COILS = 8
SIGMA = 0.01  # noise level, relative to the image's peak

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities /= bartorch.rss(sensitivities, axes=(0,), keepdim=True)

channels = torch.arange(COILS)
levels = torch.linspace(0.6, 1.6, COILS)
correlation = 0.5 ** (channels[:, None] - channels[None, :]).abs().float()
covariance = levels[:, None] * correlation * levels[None, :]

# sphinx_gallery_start_ignore
mixing = torch.linalg.cholesky(covariance).to(torch.complex64)
generator = torch.Generator().manual_seed(2)


def channel_noise(shape):
    """Complex Gaussian noise with the channel covariance above."""
    white = torch.randn(COILS, *shape, generator=generator)
    white = (white + 1j * torch.randn(COILS, *shape, generator=generator)) / 2**0.5
    return SIGMA * torch.einsum("ij,j...->i...", mixing, white.to(torch.complex64))


# sphinx_gallery_end_ignore

# %%
# The noise scan records the same channels with no signal.
# :func:`bartorch.tools.whiten` takes the data and the noise scan, estimates
# :math:`\Psi` as the sample covariance of the noise scan, and whitens the
# data with it. Applied to the noise scan itself, it returns white noise:
noise_scan = channel_noise((SIZE, SIZE))[:, None]  # (coils, 1, y, x)
whitened_scan = bt.whiten(noise_scan, noise_scan)

# sphinx_gallery_start_ignore
measured = channel_covariance(noise_scan)
whitened = channel_covariance(whitened_scan)
print(f"off-diagonal covariance: {off_diagonal(measured):.3f} measured")
print(f"                         {off_diagonal(whitened):.3f} after whitening")

figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.9))
for axis, values, title in (
    (axes[0], measured, "measured"),
    (axes[1], whitened, "whitened"),
):
    handle = axis.imshow(
        (values.abs() / values.abs().max()).numpy(), cmap="magma", vmin=0.0, vmax=1.0
    )
    axis.set_title(title)
    axis.set_xlabel("channel")
    axis.set_xticks(range(0, COILS, 2))
    axis.set_yticks(range(0, COILS, 2))
axes[0].set_ylabel("channel")
figure.colorbar(handle, ax=axes, fraction=0.046, label="|covariance| / max")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The printed number is the mean magnitude of the off-diagonal covariance
# entries relative to the mean diagonal entry, which is zero for
# uncorrelated channels. In the figure, the measured covariance has a strong
# diagonal whose entries grow with the channel index, the unequal noise
# levels, and off-diagonal bands, the correlation. After whitening it is the
# identity.
#
# Acquisition and calibration
# ---------------------------
#
# You acquire every second phase encode (:math:`R = 2`), with 24 central
# lines kept as the ACS region. Each pseudo-replica adds a new noise
# realisation to the same noise-free k-space. You calibrate the
# sensitivities once per pipeline, from the first replica: from the channels
# as measured, and from the whitened channels.
CALIBRATION = 24
REPLICAS = 32

lines = torch.zeros(SIZE)
lines[::2] = 1.0
centre = slice(SIZE // 2 - CALIBRATION // 2, SIZE // 2 + CALIBRATION // 2)
lines[centre] = 1.0
pattern = lines.reshape(SIZE, 1).to(torch.complex64)

noiseless = bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True)


def acquire():
    """One replica: the noise-free k-space plus new noise, sampled."""
    noisy = noiseless + channel_noise((SIZE, SIZE))
    return (noisy * pattern)[:, None]


first = acquire()
maps_measured = bt.ecalib(first, maps=1, calib_size=CALIBRATION, crop=0.8)
maps_whitened = bt.ecalib(
    bt.whiten(first, noise_scan),
    maps=1,
    calib_size=CALIBRATION,
    crop=0.8,
)

# %%
# Pseudo-replicas
# ---------------
#
# Both pipelines run the same reconstruction, conjugate-gradient SENSE with
# a small Tikhonov weight and a fixed number of iterations. The only
# difference is whether you whiten the data first.
plain, prewhitened = [], []
for _ in range(REPLICAS):
    data = acquire()
    plain.append(apps.pics(data, maps_measured, l2=1e-3, maxiter=30))
    data = bt.whiten(data, noise_scan)
    prewhitened.append(apps.pics(data, maps_whitened, l2=1e-3, maxiter=30))

plain, prewhitened = torch.stack(plain), torch.stack(prewhitened)

# %%
# The SNR of a voxel is the magnitude of the mean reconstruction over the
# standard deviation across replicas. You report both pipelines over the
# white matter, where the phantom is homogeneous.


def snr_map(replicas):
    return replicas.mean(0).abs() / replicas.std(0)


gain = snr_map(prewhitened) / snr_map(plain)

# sphinx_gallery_start_ignore
for name, replicas in (("as measured", plain), ("prewhitened", prewhitened)):
    values = snr_map(replicas)[white_matter]
    mean, median = float(values.mean()), float(values.median())
    print(f"{name:>12}  white-matter SNR {mean:6.1f} (median {median:.1f})")
print(f"SNR ratio, prewhitened over as measured: {float(gain[white_matter].median()):.2f} (median)")

support = (image.abs() > 0.05 * float(image.abs().max())).numpy()
shared = float(snr_map(prewhitened)[white_matter].quantile(0.99))

figure, axes = panels(2, bars=1)
for axis, values, title in (
    (axes[0, 0], snr_map(plain), "SNR, as measured"),
    (axes[0, 1], snr_map(prewhitened), "SNR, prewhitened"),
):
    handle = axis.imshow(
        np.where(support, values.numpy(), np.nan), cmap="viridis", vmin=0.0, vmax=shared
    )
    axis.set_title(title)
figure.colorbar(handle, ax=axes[0], fraction=0.046, label="SNR")
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), width_ratios=(1, 1.1), squeeze=False)
axes[0, 0].set_axis_off()
ratio = axes[0, 0].imshow(
    np.where(support, gain.numpy(), np.nan), cmap="RdBu_r", vmin=0.5, vmax=1.5
)
axes[0, 0].set_title("SNR ratio")
figure.colorbar(ratio, ax=axes[0, 0], fraction=0.046, label="prewhitened / as measured")
bins = np.linspace(0.0, shared, 40)
for replicas, label in ((plain, "as measured"), (prewhitened, "prewhitened")):
    axes[0, 1].hist(snr_map(replicas)[white_matter].numpy(), bins=bins, alpha=0.6, label=label)
axes[0, 1].set_xlabel("SNR in white matter")
axes[0, 1].set_ylabel("voxels")
axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# Prewhitening raises the SNR throughout the head, and the white-matter
# histogram shifts by the ratio printed above. The gain varies in space: it
# is largest in the posterior half of the head (the bottom of the image).
# Both the quietest channels, 0 and 1, and the noisiest, 6 and 7, are most
# sensitive there. Without whitening, the least-squares fit gives them all
# the same weight; after whitening, each channel enters with the weight its
# noise level warrants, and that matters most where channels of very
# different noise levels overlap. How large the gain is depends on the
# array's noise correlation and on the spread of its channel noise levels,
# and what you measured here is for one simulated covariance. With
# uncorrelated channels of equal noise level it is one.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Measure the SNR gain of noise prewhitening for a Cartesian SENSE
#    reconstruction with bartorch. Simulate an 8-channel acquisition with a
#    known channel noise covariance and a noise-only scan; undersample by 2
#    along y with a 24-line ACS block. Calibrate one set of ESPIRiT maps
#    (bt.ecalib, crop 0.8) from the data as measured and from the data
#    whitened by bt.whiten(data, noise_scan). Reconstruct 32 pseudo-replicas
#    with apps.pics (l2=1e-3, 30 iterations) in both pipelines, and report
#    the SNR map (|mean| / std across replicas), its white-matter mean and
#    median, and the median prewhitened / as-measured SNR ratio.

# %%
#
# References
# ----------
#
# .. [#roemer] Roemer PB, Edelstein WA, Hayes CE, Souza SP, Mueller OM. The NMR
#    phased array. *Magn Reson Med* 16(2):192-225 (1990).
#    https://doi.org/10.1002/mrm.1910160203
#
# .. [#pruessmann] Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in
#    sensitivity encoding with arbitrary k-space trajectories. *Magn Reson Med*
#    46(4):638-651 (2001). https://doi.org/10.1002/mrm.1241
#
# .. [#robson] Robson PM, Grant AK, Madhuranthakam AJ, Lattanzi R, Sodickson DK,
#    McKenzie CA. Comprehensive quantification of signal-to-noise ratio and
#    g-factor for image-based and k-space-based parallel imaging
#    reconstructions. *Magn Reson Med* 60(4):895-907 (2008).
#    https://doi.org/10.1002/mrm.21728
