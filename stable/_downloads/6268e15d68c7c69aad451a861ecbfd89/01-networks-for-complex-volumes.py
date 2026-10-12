r"""
============================
Networks for complex volumes
============================

In lessons 10 and 11 your network denoised one complex 2D slice. The data you
most need a learned reconstruction for are larger: a 3D volume of several
contrasts, the subspace coefficients of MR fingerprinting, the frames of a
cine. In this tour you train a 3D denoiser on patches of a complex,
multi-contrast brain volume, apply it to the whole volume of another subject
patch by patch, the way it would run on a scanner GPU too small for the
volume, and check that the patch boundaries leave no seams.

Three things change when you go from a slice to such a volume:

- You denoise the contrasts jointly, as channels of one image, so that the
  network can use the anatomy they share. Their signal levels differ, so you
  balance them before the network sees them.
- The network's activations for the whole volume do not fit in GPU memory, so
  you train on patches and apply the network patch by patch.
- A network applied on a fixed grid of patches leaves seams at the patch
  boundaries. Offsetting the grid at random and averaging a few calls spreads
  them out.

Prerequisites: lesson 11, :doc:`../06-learning/02-modl-with-admm`.

**Learning objectives**

- Build a 3D :class:`bartorch.learning.UNet` for complex multi-contrast images
  with :class:`bartorch.learning.ComplexNet`, and balance the contrasts by
  whitening.
- Train it on patches drawn by ``torchio``, with augmentations that preserve
  the complex MR signal.
- Apply it to a whole volume with :class:`bartorch.learning.Patchwise`, and
  average the seams out with :func:`bartorch.learning.moments`.
- Compare the size of spatial and spatiotemporal networks.
"""

# %%

# sphinx_gallery_start_ignore
import csv
import logging
from pathlib import Path

import brainweb_dl
import matplotlib.pyplot as plt
import numpy as np
import torch
from brainweb_dl import get_mri
from matplotlib.patches import Rectangle

PAGE_WIDTH = 7.8  # inches, the width of the documentation column


def panels(rows, columns, width=PAGE_WIDTH, bar=False):
    """A grid of square image panels, with room for one colorbar on the right if ``bar``."""
    side = (width - (0.9 if bar else 0.0)) / columns
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(width, rows * side + 0.45))
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
    return figure, axes


def show(axis, values, title=None, vmax=None, cmap="gray"):
    """One panel, of a magnitude."""
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


def nrmse(made, truth):
    return float((made.abs() - truth.abs()).norm() / truth.abs().norm())


def outline(axis, crop):
    rows, cols = crop
    axis.add_patch(
        Rectangle(
            (cols.start - 0.5, rows.start - 0.5),
            cols.stop - cols.start,
            rows.stop - rows.start,
            fill=False,
            edgecolor="#e8a33d",
            linewidth=1.5,
        )
    )


def compare(truth, results, crop, scale=None):
    """The reference and each result, whole and magnified on ``crop``, then the errors.

    The whole images and the magnified regions are the two rows of one figure.
    A second figure holds the magnitude error of each result relative to the
    peak of the reference, on one window ``scale`` shared by all of them (by
    default the 99th percentile of the largest error), and the NRMSE.
    """
    top = float(truth.abs().max())
    names = ["reference", *results]
    images = [truth, *results.values()]
    figure, axes = panels(2, len(images))
    for axis, name, image in zip(axes[0], names, images):
        show(axis, image, name, vmax=top)
    outline(axes[0, 0], crop)
    for axis, image in zip(axes[1], images):
        show(axis, image[crop], vmax=top)
    axes[1, 0].set_ylabel("enlarged")

    difference = [(made.abs() - truth.abs()).abs() / top for made in results.values()]
    if scale is None:
        scale = max(float(torch.quantile(d.flatten(), 0.99)) for d in difference)
    figure, axes = panels(1, len(results), bar=True)
    for axis, name, made, error in zip(axes[0], results, results.values(), difference):
        handle = show(
            axis, error, f"{name}\nNRMSE {nrmse(made, truth):.3f}", vmax=scale, cmap="magma"
        )
    figure.colorbar(handle, ax=axes[0], fraction=0.046, label="|error| / peak")


# The phantom: three spin-echo contrasts of a BrainWeb [#brainweb]_ subject,
# each voxel the sum of the signals of the tissues it contains, with a smooth
# background phase per contrast.
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
SIZE = 64
TISSUES = (1, 2, 3, 4, 5, 6, 8)
CONTRASTS = {"T1w": (600.0, 12.0), "T2w": (4000.0, 100.0), "PDw": (4000.0, 12.0)}  # TR, TE in ms

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue = {
    name: np.array([float(row[f"{name} (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
    for name in ("T1", "T2", "PD")
}
axis = torch.linspace(-1.0, 1.0, SIZE)
grid_z, grid_y, grid_x = torch.meshgrid(axis, axis, axis, indexing="ij")


def brain_volume(subject):
    """Three complex spin-echo contrasts of a BrainWeb subject, ``(3, SIZE, SIZE, SIZE)``."""
    volume = get_mri(sub_id=subject, contrast="fuzzy")[..., list(TISSUES)]
    occupied = np.nonzero(volume.sum(-1) > 0.5)
    middle = [int((a.min() + a.max()) / 2) for a in occupied]
    half = int(round(1.1 * max(a.max() - a.min() for a in occupied) / 2))
    box = np.zeros((2 * half,) * 3 + volume.shape[-1:], dtype=np.float32)
    source = tuple(slice(max(0, c - half), min(n, c + half)) for c, n in zip(middle, volume.shape))
    box[tuple(slice(s.start - (c - half), s.stop - (c - half)) for s, c in zip(source, middle))] = (
        volume[source]
    )
    memberships = torch.nn.functional.interpolate(
        torch.as_tensor(box).permute(3, 0, 1, 2)[None], size=(SIZE,) * 3, mode="area"
    )[0].flip(1)
    made = []
    for index, (tr, te) in enumerate(CONTRASTS.values()):
        signal = tissue["PD"] * (1 - np.exp(-tr / tissue["T1"])) * np.exp(-te / tissue["T2"])
        image = (memberships * torch.as_tensor(signal)[:, None, None, None]).sum(0)
        phase = torch.exp(1j * (0.8 * grid_x**2 - 0.4 * grid_y * grid_z + 0.5 * index * grid_z))
        made.append(image / image.max() * phase)
    return torch.stack(made).to(torch.complex64)


# sphinx_gallery_end_ignore

# %%
# Multi-contrast complex volume
# -----------------------------
#
# Your data are three spin-echo contrasts of a BrainWeb subject [#brainweb]_
# on a :math:`64^3` grid: :math:`T_1`-weighted (TR 600 ms, TE 12 ms),
# :math:`T_2`-weighted (TR 4000 ms, TE 100 ms) and proton-density-weighted
# (TR 4000 ms, TE 12 ms), each with its own smooth background phase. They come
# as one ``(3, z, y, x)`` complex tensor; how they are simulated is hidden.
# You train on subject 0 and test on subject 4, whose volume you corrupt with
# complex Gaussian noise of 6 per cent of each contrast's peak, an SNR typical
# of a fast high-resolution scan.
import lightning
import torch
import torchio as tio

from bartorch import learning
from torch.utils.data import DataLoader

PATCH = 32
NOISE = 0.06

_ = torch.manual_seed(0)

train_volume = brain_volume(subject=0)
test_volume = brain_volume(subject=4)
noisy = test_volume + NOISE * torch.randn_like(test_volume)
print(f"test volume {tuple(test_volume.shape)}, {test_volume.dtype}")

# sphinx_gallery_start_ignore
figure, axes = panels(2, 3)
for column, name in enumerate(CONTRASTS):
    show(axes[0, column], test_volume[column, SIZE // 2], name)
    show(axes[1, column], noisy[column, SIZE // 2], f"{name}, noisy")
plt.show()
# sphinx_gallery_end_ignore

# %%
# An axial slice through the middle of the test volume, clean on top and with
# the noise you will remove below. The three contrasts share the anatomy and
# differ in which tissue is bright, which is what a joint denoiser can use.
#
# The network
# -----------
#
# :class:`~bartorch.learning.UNet` is a residual U-Net whose last convolution
# starts at zero, so the untrained network returns its input. A U-Net works on
# real channels, so you wrap it in :class:`~bartorch.learning.ComplexNet`,
# which lays the three complex contrasts out as six real channels -- the real
# parts, then the imaginary parts -- and lays the output back out as complex
# contrasts.
#
# With ``normalize="whiten"``, ``ComplexNet`` subtracts each channel's mean
# and multiplies by the inverse square root of the channels' covariance before
# the call, and undoes both after it. The network then sees uncorrelated
# channels of unit variance, whatever the relative energy of the contrasts.
net = learning.UNet(6, spatial=3, widths=(8, 16, 32))
denoiser = learning.ComplexNet(net, spatial=3, channels=1, normalize="whiten")
weights = sum(p.numel() for p in net.parameters())
print(f"{weights} weights, {4 * weights / 1e6:.2f} MB in single precision")

with torch.no_grad():
    identity = torch.allclose(denoiser(noisy[None])[0], noisy, atol=1e-5)
print("untrained network returns its input:", identity)

# %%
# Patches and augmentation
# ------------------------
#
# ``torchio`` holds the training volume as a :class:`torchio.ScalarImage` of
# real channels (:func:`~bartorch.learning.as_real`) and draws :math:`32^3`
# patches from it through a :class:`torchio.Queue`.
#
# Choose augmentations that turn one MR image into another the acquisition
# could have produced: a flip, and :class:`~bartorch.learning.RandomGain`, a
# receiver gain and global phase shared by the contrasts. An intensity
# transform applied to the real and imaginary channels separately, such as a
# gamma correction, would produce a signal no acquisition can. The global
# phase varies over a limited range here: a network trained over every phase
# has to learn to commute with a rotation of its real and imaginary channels,
# which takes more training than this tour runs.
volume = learning.as_real(train_volume).flatten(0, 1)
subject = tio.Subject(image=tio.ScalarImage(tensor=volume))
augment = tio.Compose(
    [
        tio.RandomFlip(axes=(0, 1, 2)),
        learning.RandomGain(phase=0.3, log_scale=0.2),
    ]
)
queue = tio.Queue(
    tio.SubjectsDataset([subject], transform=augment),
    max_length=64,
    samples_per_volume=32,
    sampler=tio.UniformSampler(PATCH),
    num_workers=0,
)

# %%
# Each patch becomes a training pair when you add a new draw of noise to it,
# so the network sees a different noise realization at every epoch. The test
# volume serves as the validation pair.


def pairs(patches):
    made = []
    for patch in patches:
        real = patch["image"][tio.DATA].unflatten(0, (2, 3))
        clean = learning.as_complex(real)
        noisy_patch = clean + NOISE * torch.randn_like(clean)
        made.append({"input": noisy_patch, "target": clean})
    return made


validation = [{"input": noisy, "target": test_volume}]
trainer = lightning.Trainer(
    max_epochs=30,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)
trainer.fit(
    learning.Reconstruction(denoiser, "denoiser", lr=2e-3),
    DataLoader(queue, batch_size=4, collate_fn=pairs),
    DataLoader(validation, batch_size=1, collate_fn=list),
)

# %%
# Patchwise inference
# -------------------
#
# :class:`~bartorch.learning.Patchwise` keeps the volume in host memory, sends
# it to the network's device a few patches at a time, runs the network there
# in mixed precision, and assembles the result on the host. On a GPU only the
# network and one group of ``batch`` patches are resident -- two groups at
# inference, where the copies of one group overlap the computation on the
# other. That is how you denoise a volume larger than the GPU memory, such as
# a whole-brain fingerprinting series on a scanner's 16 GB GPU, with a network
# trained on patches of it. Here, on the host, the whole volume is small
# enough to denoise in one call too, and that is your reference.
#
# A U-Net is not translation invariant at a patch boundary: its receptive
# field extends past the patch, where it sees zeros rather than the
# neighbouring voxels. On a fixed grid of patches the errors fall along the
# same planes every time. With ``shift=True`` the grid is offset at random at
# every call, and averaging a few calls with
# :func:`~bartorch.learning.moments` spreads the boundary errors across the
# volume.
grid = (PATCH,) * 3
whole_net = learning.ComplexNet(net, spatial=3, channels=1, normalize="whiten")
fixed = learning.ComplexNet(
    learning.Patchwise(net, grid, shift=False),
    spatial=3,
    channels=1,
    normalize="whiten",
)
shifted = learning.ComplexNet(
    learning.Patchwise(net, grid, shift=True),
    spatial=3,
    channels=1,
    normalize="whiten",
)

with torch.no_grad():
    whole = whole_net(noisy[None])[0]
    on_grid = fixed(noisy[None])[0]
    averaged, spread = learning.moments(
        lambda: shifted(noisy[None])[0],
        samples=8,
    )

# sphinx_gallery_start_ignore


def error(made):
    return float((made - test_volume).norm() / test_volume.norm())


print(f"noisy                  relative error {error(noisy):.4f}")
for name, made in (("whole volume", whole), ("fixed grid", on_grid), ("8 random grids", averaged)):
    print(
        f"{name:<22} relative error {error(made):.4f}, "
        f"departure from the whole volume {float((made - whole).norm() / whole.norm()):.4f}"
    )

cut = (1, slice(None), slice(None), SIZE // 2 + 3)  # the T2w contrast, a sagittal slice
compare(
    test_volume[cut],
    {"noisy": noisy[cut], "denoised, whole": whole[cut]},
    crop=(slice(14, 46), slice(16, 48)),
)
plt.show()

figure, axes = panels(1, 2, bar=True)
top = float(test_volume[cut].abs().max())
departure_grid = (on_grid - whole)[cut].abs() / top
departure_avg = (averaged - whole)[cut].abs() / top
scale = float(departure_grid.max())
show(axes[0, 0], departure_grid, "fixed grid − whole", vmax=scale, cmap="magma")
handle = show(axes[0, 1], departure_avg, "8 random grids − whole", vmax=scale, cmap="magma")
figure.colorbar(handle, ax=axes[0], fraction=0.046, label="|departure| / peak")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The network, 0.14 million weights trained on patches of one head, reduces
# the relative error of the other head's volume from about 0.19 to about 0.11,
# and in the sagittal :math:`T_2`-weighted slice the corpus callosum and the
# cortical folding stay sharp. A real training set and a wider network remove
# more.
#
# The departure of the fixed grid from the whole-volume result lies on the
# planes between patches -- the seams -- and on the same planes at every call.
# A shifted grid covers the volume with one more patch along each axis and so
# has more boundaries, including ones at the edges of the volume, and in norm
# the average of eight shifted calls departs further from the whole-volume
# result than the fixed grid does. The average has no planes, however: the
# departure is spread over the volume instead of concentrated where it would
# read as an anatomical edge. Inside an iteration, which applies the denoiser
# once per step, one shifted grid per call is enough, because no plane
# receives the boundary error at every step. The variance
# :func:`~bartorch.learning.moments` returns is a map of how much the result
# depends on where the patches fall, one of the spreads of
# :doc:`05-uncertainty`.
#
# Size of the network
# -------------------
#
# The weights set the storage footprint and, with the patch size, the memory
# of a call. The default widths of :class:`~bartorch.learning.UNet`,
# ``(32, 64, 128, 256)``, give a 3D network of a few million weights.
#
# For a series of frames -- a cine, a functional run -- you pass
# ``frames=True``. The network then convolves each frame spatially and the
# frames with a separate one-dimensional convolution, and never downsamples
# the frame axis; ``periodic=True`` pads that axis circularly, which suits a
# cardiac cycle. The factorization costs few weights beyond the spatial
# network.
candidates = {
    "3D, widths (8, 16, 32)": net,
    "3D, default widths": learning.UNet(6, spatial=3),
    "3D + frames, default widths": learning.UNet(
        6,
        spatial=3,
        frames=True,
        periodic=True,
    ),
}

# sphinx_gallery_start_ignore
for name, candidate in candidates.items():
    count = sum(p.numel() for p in candidate.parameters())
    print(f"{name:<28} {count / 1e6:5.2f} M weights, {2 * count / 1e6:5.1f} MB in half precision")
# sphinx_gallery_end_ignore

# %%
# A frame-resolved network takes a ``(n, channels, frames, z, y, x)`` input:
frames = torch.randn(1, 6, 10, 16, 16, 16)
cine = learning.UNet(6, spatial=3, widths=(8, 16), frames=True, periodic=True)
print("(n, channels, frames, z, y, x):", tuple(cine(frames).shape))

# %%
# Exercise: random patch grids
# ----------------------------
#
# Averaging more shifted calls costs one network pass each. How far from the
# whole-volume result is the average of 2, 4, 8 and 16 calls?
# :func:`~bartorch.learning.moments` needs at least two, since it also returns
# a variance. No training is involved:
counts = (2, 4, 8, 16)
departures = []
with torch.no_grad():
    for count in counts:
        mean, _ = learning.moments(
            lambda: shifted(noisy[None])[0],
            samples=count,
        )
        departures.append(float((mean - whole).norm() / whole.norm()))

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * PAGE_WIDTH, 0.42 * PAGE_WIDTH))
axis.plot(counts, departures, "o-", lw=1.8, label="random grids, averaged")
fixed_departure = float((on_grid - whole).norm() / whole.norm())
axis.axhline(fixed_departure, color="C1", ls="--", label="one fixed grid")
axis.set_xscale("log", base=2)
axis.set_xticks(counts, [str(c) for c in counts])
axis.set_xlabel("shifted calls averaged")
axis.set_ylabel("departure from whole volume")
axis.set_ylim(0, None)
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The departure decreases with the number of calls averaged, slowly, and stays
# above that of the fixed grid at every count tried. What averaging shifted
# grids buys is the distribution of the boundary error, off fixed planes, not
# a result closer to the whole-volume one.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Train a 3D denoiser for a complex (3, 64, 64, 64) multi-contrast volume
#    with bartorch.learning: a UNet(6, spatial=3, widths=(8, 16, 32)) inside
#    ComplexNet(spatial=3, channels=1, normalize="whiten"). Draw 32^3
#    patches of the training volume with a torchio Queue (as_real channels,
#    RandomFlip and learning.RandomGain(phase=0.3, log_scale=0.2)), add
#    complex Gaussian noise of 0.06 per patch, and fit 30 epochs of
#    learning.Reconstruction(..., "denoiser", lr=2e-3) with lightning on CPU.
#    Apply it to another subject with learning.Patchwise on a 32^3 grid, fixed
#    and with shift=True averaged over 8 calls by learning.moments, and report
#    each one's departure from the whole-volume result.

# %%
#
# References
# ----------
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
