r"""
============================
Training without a reference
============================

The networks of lesson 11 and of :doc:`02-staged-training` learn from fully
sampled reference images. Dynamic and high-dimensional acquisitions -- a cine,
a functional run, a fingerprinting series -- are rarely acquired fully
sampled: undersampling is what makes them fit in a breath-hold or a scan
time, so you have no reference to train against. In this tour you train the
unrolled network of :doc:`02-staged-training` from undersampled k-space alone
and measure how much of the supervised network's image quality it keeps.

The idea is self-supervised learning via data undersampling (SSDU) [#ssdu]_:
train against the measured k-space itself. You split the acquired phase
encodes :math:`\Omega` into two disjoint sets, :math:`\Theta` and
:math:`\Lambda`. The network reconstructs from :math:`\Theta`, and the loss
compares the k-space of its reconstruction with the measured data on the
held-out set :math:`\Lambda`,

.. math::

   \mathcal{L} = \frac{\|y_\Lambda - A_\Lambda f_\theta(y_\Theta)\|_2}{\|y_\Lambda\|_2}
   + \frac{\|y_\Lambda - A_\Lambda f_\theta(y_\Theta)\|_1}{\|y_\Lambda\|_1},

where :math:`A_\Lambda` is the SENSE encoding restricted to :math:`\Lambda`.
A new split is drawn at every step, so over the training every acquired line
is both reconstructed from and held out [#multimask]_. At inference the
network reconstructs from all of :math:`\Omega`.

Prerequisites: lesson 11, :doc:`../06-learning/02-modl-with-admm`, and the
tour :doc:`02-staged-training`.

**Learning objectives**

- Partition the acquired phase encodes with :func:`bartorch.learning.split`.
- Train an unrolled network self-supervised with
  :class:`bartorch.learning.Reconstruction`, by giving items the sampling
  pattern instead of a reference.
- Compare with the same network trained against references, and with
  CG-SENSE.
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

    With at most three images, the whole images and the magnified regions are
    the two rows of one figure; with more, each is a figure of its own. The
    last figure holds the magnitude error of each result relative to the peak
    of the reference, on one window ``scale`` shared by all of them (by
    default the 99th percentile of the largest error), and the NRMSE.
    """
    top = float(truth.abs().max())
    names = ["reference", *results]
    images = [truth, *results.values()]
    if len(images) <= 3:
        figure, axes = panels(2, len(images))
        whole, magnified = axes[0], axes[1]
    else:
        columns = (len(images) + 1) // 2
        whole = panels(2, columns)[1].ravel()
        magnified = panels(2, columns)[1].ravel()
    for axis, name, image in zip(whole, names, images):
        show(axis, image, name, vmax=top)
    outline(whole[0], crop)
    for axis, name, image in zip(magnified, names, images):
        show(axis, image[crop], f"{name}, enlarged" if len(images) > 3 else None, vmax=top)
    if len(images) <= 3:
        magnified[0].set_ylabel("enlarged")

    difference = [(made.abs() - truth.abs()).abs() / top for made in results.values()]
    if scale is None:
        scale = max(float(torch.quantile(d.flatten(), 0.99)) for d in difference)
    figure, axes = panels(1, len(results), bar=True)
    for axis, name, made, error in zip(axes[0], results, results.values(), difference):
        handle = show(
            axis, error, f"{name}\nNRMSE {nrmse(made, truth):.3f}", vmax=scale, cmap="magma"
        )
    figure.colorbar(handle, ax=axes[0], fraction=0.046, label="|error| / peak")


# The phantom: axial slices of BrainWeb subjects turned into
# T1-weighted spin-echo images with a smooth phase, as in lesson 11.
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
SIZE = 96
TISSUES = (1, 2, 3, 4, 5, 6, 8)
MARGIN = 0.25
TR, TE = 600.0, 12.0  # ms

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue = {
    name: torch.as_tensor(
        np.array([float(row[f"{name} (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
    )
    for name in ("T1", "T2", "PD")
}
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
phase = torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))


def brain_slices(subject, count):
    """``count`` axial slices of a BrainWeb subject as complex spin-echo images."""
    volume = get_mri(sub_id=subject, contrast="fuzzy")
    first, last = int(0.33 * volume.shape[0]), int(0.62 * volume.shape[0])
    made = []
    for index in np.linspace(first, last, count).astype(int):
        fractions = np.flipud(volume[index])[..., list(TISSUES)].copy()
        occupied = np.nonzero(fractions.sum(-1) > 0.5)
        middle = [int((axis.min() + axis.max()) / 2) for axis in occupied]
        half = int(round((1 + MARGIN) * max(axis.max() - axis.min() for axis in occupied) / 2))
        source = tuple(
            slice(max(0, c - half), min(n, c + half)) for c, n in zip(middle, fractions.shape[:2])
        )
        box = np.zeros((2 * half, 2 * half, fractions.shape[-1]), dtype=np.float32)
        box[
            tuple(slice(s.start - (c - half), s.stop - (c - half)) for s, c in zip(source, middle))
        ] = fractions[source]
        memberships = torch.nn.functional.interpolate(
            torch.as_tensor(box).permute(2, 0, 1)[None], size=(SIZE, SIZE), mode="area"
        )[0]
        weights = memberships * tissue["PD"][:, None, None]
        share = weights.sum(0).clamp(min=1e-6)
        T1 = (weights * tissue["T1"][:, None, None]).sum(0) / share
        T2 = (weights * tissue["T2"][:, None, None]).sum(0) / share
        density = weights.sum(0) / weights.sum(0).max().clamp(min=1e-6)
        signal = density * (1 - torch.exp(-TR / T1.clamp(min=1e-3)))
        signal = signal * torch.exp(-TE / T2.clamp(min=1e-3))
        signal = torch.where(T1 > 0, signal, torch.zeros(()))
        made.append(((signal / signal.max().clamp(min=1e-6)) * phase).to(torch.complex64))
    return made


# sphinx_gallery_end_ignore

# %%
# Data
# ----
#
# You use the slices, coils and fourfold undersampling of
# :doc:`02-staged-training`: BrainWeb [#brainweb]_ subject 0 to train on and
# subject 4 to validate on. The references stay here only to score the
# results; the self-supervised network never sees them.
import lightning
import torch

import bartorch
import bartorch.tools as bt

from bartorch import learning, linop, optim, priors
from monai.metrics import PSNRMetric, SSIMMetric
from torch.utils.data import DataLoader

COILS = 8
ITERATIONS = 4
ACCELERATION = 4
EPOCHS = 16
NOISE = 0.02

_ = torch.manual_seed(0)

train_images = brain_slices(subject=0, count=24)
valid_images = brain_slices(subject=4, count=8)

coils = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = coils / bartorch.rss(coils, axes=(0,), keepdim=True)

ky = torch.arange(SIZE) - SIZE / 2
density = torch.exp(-0.5 * (ky / (SIZE / 6)) ** 2)
chance = density / density.sum() * (SIZE / ACCELERATION)
draw = torch.rand(SIZE, generator=torch.Generator().manual_seed(1))
lines = draw < chance
lines[SIZE // 2 - 4 : SIZE // 2 + 4] = True
pattern = lines.to(torch.complex64)[:, None].expand(SIZE, SIZE).contiguous()
A = linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=pattern)

generator = torch.Generator().manual_seed(3)


def measure(x):
    noise = torch.randn(A.oshape, dtype=torch.complex64, generator=generator)
    return A(x) + NOISE * noise


kspace = {
    "train": [measure(x) for x in train_images],
    "valid": [measure(x) for x in valid_images],
}

# %%
# Sample splitting
# ----------------
#
# The readout is fully sampled, so the unit of the split is the phase-encode
# line: the pattern you give :func:`~bartorch.learning.split` has one entry
# per line and broadcasts over the coils and the readout. A quarter of the
# acquired lines are held out, drawn with a Gaussian density across
# :math:`k_y`, and the eight central lines always stay in :math:`\Theta`. A
# reconstruction without the centre of k-space would lose the image contrast,
# and the loss would be dominated by it.
acquired = lines.float()[:, None]
split_seed = torch.Generator().manual_seed(0)
keep, held = learning.split(acquired, 0.25, keep=(8, 1), generator=split_seed)
print(
    f"{int(acquired.sum())} acquired lines: {int(keep.sum())} to reconstruct "
    f"from, {int(held.sum())} held out"
)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 1.6))
for lane, (row, colour) in enumerate(
    zip((acquired[:, 0], keep[:, 0], held[:, 0]), ("0.55", "C0", "C1"))
):
    lines_in = torch.nonzero(row).ravel().numpy()
    axis.bar(lines_in, 0.8, bottom=lane - 0.4, width=0.85, color=colour)
axis.set_yticks([0, 1, 2], [r"acquired $\Omega$", r"reconstruct $\Theta$", r"held out $\Lambda$"])
axis.set_ylim(2.6, -0.6)
axis.set_xlim(-1, SIZE)
axis.set_xlabel("phase-encode line $k_y$")
for side in ("top", "right", "left"):
    axis.spines[side].set_visible(False)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Of the 24 acquired lines, 20 are kept for the reconstruction and 4 are held
# out. The eight central lines are in :math:`\Theta` in every split, and the
# held-out lines are drawn from the acquired lines outside them.
#
# Supervised and self-supervised training
# ---------------------------------------
#
# Both are the iteration-conditioned unrolled network of
# :doc:`02-staged-training`, trained end to end for the same number of epochs
# from the same initialization; only the items differ. A self-supervised item
# carries the acquired ``pattern`` in place of a ``target``, and
# :class:`~bartorch.learning.Reconstruction` then draws a split at every step.
# Its validation loss is the held-out loss on a split fixed for the whole run,
# and needs no reference either.


def unrolled():
    torch.manual_seed(0)
    network = learning.UNet(2, spatial=2, widths=(16, 32, 64), steps=True)
    denoiser = learning.ComplexNet(network, spatial=2)
    block = optim.ISTBlock(priors.ImplicitPrior(denoiser, step=True), step=1.0)
    block.step.requires_grad_()
    return learning.Unrolled(block, iterations=ITERATIONS, checkpoint=True)


def items(part, supervised):
    images = train_images if part == "train" else valid_images
    made = []
    for x, y in zip(images, kspace[part]):
        item = {"y": y, "A": A}
        if supervised:
            item["target"] = x
        else:
            item["pattern"] = acquired
        made.append(item)
    return made


models = {}
for name, supervised in (("supervised", True), ("self-supervised", False)):
    models[name] = unrolled()
    trainer = lightning.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    stage = learning.Reconstruction(
        models[name],
        "end-to-end",
        lr=1e-3,
        fraction=0.25,
        split_options={"keep": (8, 1)},
    )
    trainer.fit(
        stage,
        DataLoader(
            items("train", supervised),
            batch_size=4,
            shuffle=True,
            collate_fn=list,
        ),
        DataLoader(items("valid", supervised), batch_size=4, collate_fn=list),
    )

# %%
# Results
# -------
#
# Both networks reconstruct from all acquired lines of the validation subject,
# and you score them, and CG-SENSE, against its reference:
psnr = PSNRMetric(max_val=1.0)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0)
truth = torch.stack(valid_images).abs()[:, None]


def quality(images):
    made = images.abs()[:, None]
    return float(psnr(made, truth).mean()), float(ssim(made, truth).mean())


results = {}
with torch.no_grad():
    for name, model in models.items():
        results[name] = torch.stack([model(y, A) for y in kspace["valid"]])
results["CG SENSE, 20 iterations"] = torch.stack(
    [optim.cg(y, A, maxiter=20) for y in kspace["valid"]]
)

# sphinx_gallery_start_ignore
for name, made in results.items():
    p, s = quality(made)
    print(f"{name:>24}   PSNR {p:5.2f} dB   SSIM {s:.3f}")
# sphinx_gallery_end_ignore

# %%
# Both networks score well above CG-SENSE, and the supervised one scores
# above the self-supervised one in PSNR and, by a wider margin, in SSIM. The
# self-supervised network is trained on less information: each step
# reconstructs from three quarters of the acquired lines and receives no
# signal about the lines never acquired, which is where the supervised network
# learns most. The gap between the two is what a fully sampled reference
# adds; the self-supervised network needs nothing beyond the data the
# protocol already acquires.

# sphinx_gallery_start_ignore
compare(
    valid_images[5],
    {
        "CG-SENSE": results["CG SENSE, 20 iterations"][5],
        "supervised": results["supervised"][5],
        "self-supervised": results["self-supervised"][5],
    },
    crop=(slice(22, 58), slice(30, 66)),
)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Both networks suppress the noise that the CG-SENSE unfolding amplifies
# across the whole field of view. The self-supervised error map keeps more
# residual structure, including faint horizontal bands that extend outside the
# head: aliasing along the phase-encode direction (vertical) from the lines
# never acquired, against which the self-supervised loss has no data to
# compare.
#
# Exercise: held-out fraction
# ---------------------------
#
# During training the self-supervised network only ever reconstructs from
# :math:`\Theta`. Reconstruct the validation slices from the :math:`\Theta` of
# the split above, without the held-out quarter, and compare with the full
# :math:`\Omega`. No training is involved:
subset = linop.CartesianSense(
    sensitivities, (SIZE, SIZE), pattern=(pattern * keep).to(torch.complex64)
)
from_theta = {}
with torch.no_grad():
    for name, model in models.items():
        made = [model(y * keep, subset) for y in kspace["valid"]]
        from_theta[name] = torch.stack(made)

# sphinx_gallery_start_ignore
for name, made in from_theta.items():
    print(
        f"{name:>16}: PSNR {quality(results[name])[0]:5.2f} dB from Omega, "
        f"{quality(made)[0]:5.2f} dB from Theta"
    )
# sphinx_gallery_end_ignore

# %%
# Without the held-out quarter both networks lose PSNR, the supervised one by
# about 2.7 dB and the self-supervised one by under 1 dB, so that the
# self-supervised network is now the better of the two. It was trained on
# inputs with that sampling; the supervised network was trained on all of
# :math:`\Omega` and has not seen an input with these lines missing.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Train the 4-iteration unrolled ISTA network of the staged-training tour
#    (UNet(2, spatial=2, widths=(16, 32, 64), steps=True), learned step) on
#    2D Cartesian SENSE data (8 coils, R = 4, 8 central lines, noise 0.02)
#    twice with learning.Reconstruction(..., "end-to-end", lr=1e-3), 16
#    epochs each: supervised, with a "target" per item, and self-supervised
#    (SSDU), with the acquired line "pattern" per item, fraction=0.25 and
#    split_options={"keep": (8, 1)}. Reconstruct the validation subject from
#    all acquired lines and report PSNR and SSIM of both and of CG-SENSE.

# %%
#
# References
# ----------
#
# .. [#ssdu] Yaman B, Hosseini SAH, Moeller S, Ellermann J, Ugurbil K, Akcakaya M.
#    Self-supervised learning of physics-guided reconstruction neural networks
#    without fully sampled reference data. *Magn Reson Med* 84(6):3172-3191
#    (2020). https://doi.org/10.1002/mrm.28378
#
# .. [#multimask] Yaman B, Hosseini SAH, Moeller S, Ellermann J, Ugurbil K,
#    Akcakaya M. Multi-mask self-supervised learning for physics-guided neural
#    networks in highly accelerated magnetic resonance imaging. *NMR Biomed*
#    35(12):e4798 (2022). https://doi.org/10.1002/nbm.4798
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
