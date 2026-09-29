r"""
===============================
Networks for complex volumes
===============================

A convolutional denoiser for complex, multi-contrast volumes, trained on
patches and applied to a whole volume patch by patch.

The networks of the previous lessons denoise one complex slice. The data a
learned reconstruction is most needed for are larger: a volume of several
contrasts, of subspace coefficients, or of frames of a cine. Three things
change. The contrasts are denoised jointly, as channels of one image, and
differ in energy, so they are balanced before the network sees them. The
volume does not fit a network's activations on a card, so the network is
trained on patches of it and applied to it patch by patch. And a network
applied on a fixed grid of patches leaves seams at the patch boundaries,
which a random offset of the grid averages out.

**Learning objectives**

- Build a three-dimensional :class:`bartorch.learning.UNet` for complex
  multi-contrast images with :class:`bartorch.learning.ComplexNet`, and
  balance the contrasts by whitening.
- Train it on patches drawn by ``torchio``, with augmentations that preserve
  the complex signal model.
- Apply it to a whole volume with :class:`bartorch.learning.Patchwise`, and
  average the seams out with :func:`bartorch.learning.moments`.
- Compare the size of spatial and spatiotemporal networks.

It follows :doc:`02-modl-with-admm`. The next lesson,
:doc:`04-staged-training`, trains an unrolled network in stages.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 11,
        "figure.constrained_layout.use": True,
    }
)

PAGE_WIDTH = 8.0  # inches, the width of the documentation column


def panels(rows, columns, height=1.0):
    """A grid of square image panels filling the documentation column."""
    side = PAGE_WIDTH / columns
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(PAGE_WIDTH, rows * side * height + 0.4)
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    return figure, axes


def show(axis, values, title=None, vmax=None, cmap="gray"):
    """One panel, of a magnitude."""
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


# sphinx_gallery_end_ignore
import csv
import logging
from pathlib import Path

import brainweb_dl
import lightning
import numpy as np
import torch
import torchio as tio
from brainweb_dl import get_mri
from torch.utils.data import DataLoader

from bartorch import learning
from bartorch.learning import training

SIZE = 64
PATCH = 32
NOISE = 0.06

_ = torch.manual_seed(0)

# %%
#
# A multi-contrast complex volume
# -------------------------------
#
# Three spin-echo contrasts of a BrainWeb subject -- T1-, T2- and proton
# density-weighted -- on a :math:`64^3` grid, each with its own smooth phase,
# as a ``(3, z, y, x)`` complex tensor. Subject 0 is the training volume and
# subject 4 the test volume. Each contrast of a voxel is the sum over the
# tissues it contains of their spin-echo signals.

# sphinx_gallery_start_ignore
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
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
train_volume = brain_volume(subject=0)
test_volume = brain_volume(subject=4)
noisy = test_volume + NOISE * torch.randn_like(test_volume)
print(f"test volume {tuple(test_volume.shape)}, {test_volume.dtype}")

# sphinx_gallery_start_ignore
figure, axes = panels(2, 3)
for column, name in enumerate(CONTRASTS):
    show(axes[0, column], test_volume[column, SIZE // 2], name)
    show(axes[1, column], noisy[column, SIZE // 2], "noisy" if 0 == column else None)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The network
# -----------
#
# :class:`~bartorch.learning.UNet` is a residual U-Net whose last convolution
# starts at zero, so that the untrained network returns its input.
# :class:`~bartorch.learning.ComplexNet` lays the three complex contrasts out
# as its six real channels -- the real parts, then the imaginary parts -- and
# lays its output back out as complex contrasts. With
# ``normalize="whiten"`` it subtracts each channel's mean and multiplies by the
# inverse square root of the channels' covariance before the call, and undoes
# both after it, so that the network sees uncorrelated channels of unit
# variance whatever the relative energy of the contrasts.

net = learning.UNet(6, spatial=3, widths=(8, 16, 32))
denoiser = learning.ComplexNet(net, spatial=3, channels=1, normalize="whiten")
weights = sum(p.numel() for p in net.parameters())
print(f"{weights} weights, {4 * weights / 1e6:.2f} MB in single precision")

with torch.no_grad():
    print(
        "untrained network returns its input:",
        torch.allclose(denoiser(noisy[None])[0], noisy, atol=1e-5),
    )

# %%
#
# Patches and augmentation
# ------------------------
#
# ``torchio`` holds the training volume as a :class:`torchio.ScalarImage` of
# real channels (:func:`~bartorch.learning.as_real`) and draws patches from it
# through a :class:`torchio.Queue`. The augmentations are the ones that map a
# complex image to another the acquisition could have produced: a flip, and
# :class:`~bartorch.learning.training.RandomGain`, a complex gain shared by the
# contrasts, which varies the overall scale and phase. An intensity
# transformation applied to the real and imaginary channels separately, such
# as a gamma correction, would not. The phase is varied over a limited range:
# a network trained over every global phase has to learn to commute with a
# rotation of its real and imaginary channels, which takes more training than
# this lesson runs.
#
# Each patch becomes a training pair when a new draw of noise is added to it,
# so the network sees a different noise realization at every epoch.

subject = tio.Subject(image=tio.ScalarImage(tensor=learning.as_real(train_volume).flatten(0, 1)))
augment = tio.Compose(
    [tio.RandomFlip(axes=(0, 1, 2)), training.RandomGain(phase=0.3, log_scale=0.2)]
)
queue = tio.Queue(
    tio.SubjectsDataset([subject], transform=augment),
    max_length=64,
    samples_per_volume=32,
    sampler=tio.UniformSampler(PATCH),
    num_workers=0,
)


def pairs(patches):
    made = []
    for patch in patches:
        clean = learning.as_complex(patch["image"][tio.DATA].unflatten(0, (2, 3)))
        made.append({"input": clean + NOISE * torch.randn_like(clean), "target": clean})
    return made


validation = [{"input": noisy, "target": test_volume}]
trainer = lightning.Trainer(
    max_epochs=12,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)
trainer.fit(
    training.Reconstruction(denoiser, "denoiser", lr=2e-3),
    DataLoader(queue, batch_size=4, collate_fn=pairs),
    DataLoader(validation, batch_size=1, collate_fn=list),
)

# %%
#
# Applying the network patch by patch
# -----------------------------------
#
# :class:`~bartorch.learning.Patchwise` applies the network to an image held
# on the host a few patches at a time, on the network's device, under mixed
# precision there, and assembles the result on the host. On a card only the
# network and ``batch`` patches are resident, so a volume larger than the
# card's memory is denoised by a network trained on patches of it. Here, on the
# host, the whole volume is also small enough to be denoised in one call,
# which is the reference the patchwise result is compared with.
#
# A U-Net is not translation invariant at a patch boundary: its receptive field
# extends past the patch, where it sees zeros rather than the neighbouring
# voxels. On a fixed grid of patches the errors fall along the same planes
# every time. With ``shift=True``, the grid is offset at random at every call,
# and averaging a few calls with :func:`~bartorch.learning.moments` spreads
# the boundary errors across the volume.

whole_net = learning.ComplexNet(net, spatial=3, channels=1, normalize="whiten")
fixed = learning.ComplexNet(
    learning.Patchwise(net, (PATCH,) * 3, shift=False), spatial=3, channels=1, normalize="whiten"
)
shifted = learning.ComplexNet(
    learning.Patchwise(net, (PATCH,) * 3, shift=True), spatial=3, channels=1, normalize="whiten"
)

with torch.no_grad():
    whole = whole_net(noisy[None])[0]
    grid = fixed(noisy[None])[0]
    averaged, spread = learning.moments(lambda: shifted(noisy[None])[0], samples=8)


def error(made):
    return float((made - test_volume).norm() / test_volume.norm())


print(f"noisy                  relative error {error(noisy):.4f}")
for name, made in (("whole volume", whole), ("fixed grid", grid), ("8 random grids", averaged)):
    print(
        f"{name:<22} relative error {error(made):.4f}, "
        f"departure from the whole volume {float((made - whole).norm() / whole.norm()):.4f}"
    )

# sphinx_gallery_start_ignore
figure, axes = panels(1, 4)
top = float(test_volume[0, :, :, SIZE // 2 + 3].abs().max())
show(axes[0, 0], whole[0, :, :, SIZE // 2 + 3], "whole volume", vmax=top)
departure_grid = (grid - whole)[0, :, :, SIZE // 2 + 3].abs()
departure_avg = (averaged - whole)[0, :, :, SIZE // 2 + 3].abs()
scale = float(departure_grid.max())
show(axes[0, 1], grid[0, :, :, SIZE // 2 + 3], "fixed grid", vmax=top)
show(axes[0, 2], departure_grid, "fixed grid, departure", vmax=scale, cmap="magma")
show(axes[0, 3], departure_avg, "8 random grids, departure", vmax=scale, cmap="magma")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The departure of the fixed grid lies on the planes between patches, the same
# planes at every call. A shifted grid covers the volume with one more patch
# along each axis and so has more boundaries, and each call departs further
# from the whole-volume result; but the boundaries move from call to call, and
# the average of eight calls spreads the departure across the volume instead
# of concentrating it on planes. Neither changes the error against the
# reference beyond the third digit. Inside an iteration, which applies the
# denoiser once per step, a single shifted grid per call is enough: no plane
# receives the boundary error at every step. The variance
# :func:`~bartorch.learning.moments` returns is a map of how much the result
# depends on where the patches fall, one of the spreads of
# :doc:`07-uncertainty`.

# %%
#
# Size of the network
# -------------------
#
# The weights determine the storage footprint and, with the patch size, the
# memory of a call. The default widths of :class:`~bartorch.learning.UNet`,
# ``(32, 64, 128, 256)``, give a three-dimensional network of a few million
# weights. A series of frames -- a cine, a functional run -- is taken with
# ``frames=True``: the network then convolves each frame spatially and the
# frames with a separate one-dimensional convolution, and never downsamples
# the frame axis; ``periodic=True`` pads it circularly, which suits a cardiac
# cycle. This factorization costs few weights beyond the spatial network.

for name, candidate in (
    ("3D, widths (8, 16, 32)", net),
    ("3D, default widths", learning.UNet(6, spatial=3)),
    ("3D + frames, default widths", learning.UNet(6, spatial=3, frames=True, periodic=True)),
):
    count = sum(p.numel() for p in candidate.parameters())
    print(f"{name:<28} {count / 1e6:5.2f} M weights, {2 * count / 1e6:5.1f} MB in half precision")

frames = torch.randn(1, 6, 10, 16, 16, 16)
cine = learning.UNet(6, spatial=3, widths=(8, 16), frames=True, periodic=True)
print("(n, channels, frames, z, y, x):", tuple(cine(frames).shape))
