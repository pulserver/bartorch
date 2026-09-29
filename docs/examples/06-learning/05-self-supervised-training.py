r"""
============================
Training without a reference
============================

**Aim.** Train the unrolled network of :doc:`04-staged-training` from
undersampled k-space alone, with no fully sampled reference, and measure how
much of the supervised network's image quality it retains.

Dynamic and high-dimensional acquisitions -- a cine, a functional run, a
fingerprinting series -- are rarely acquired fully sampled: undersampling is
what makes them feasible within a breath-hold or a scan time. There is then no
reference image to train against. Self-supervised learning via data
undersampling (SSDU) [#ssdu]_ trains against the measured k-space itself. The
acquired phase encodes :math:`\Omega` are split into two disjoint sets,
:math:`\Theta` and :math:`\Lambda`; the network reconstructs from
:math:`\Theta`, and the loss compares the k-space of its reconstruction with
the measured data on the held-out set :math:`\Lambda`,

.. math::

   \mathcal{L} = \frac{\|y_\Lambda - A_\Lambda f_\theta(y_\Theta)\|_2}{\|y_\Lambda\|_2}
   + \frac{\|y_\Lambda - A_\Lambda f_\theta(y_\Theta)\|_1}{\|y_\Lambda\|_1},

where :math:`A_\Lambda` is the SENSE encoding restricted to :math:`\Lambda`.
A new split is drawn at every step, so over the training every acquired line
is both reconstructed from and held out [#multimask]_. At inference the
network reconstructs from all of :math:`\Omega`.

**Learning objectives**

- Partition the acquired phase encodes with :func:`bartorch.learning.split`.
- Train an unrolled network self-supervised with
  :class:`bartorch.learning.training.Reconstruction`, by giving items the
  sampling pattern instead of a reference.
- Compare with the same network trained against references, and with
  CG-SENSE.

It follows :doc:`04-staged-training`. The next lesson,
:doc:`06-annealed-plug-and-play`, uses a denoiser trained once for any
acquisition.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PAGE_WIDTH = 8.0  # inches, the width of the documentation column


def panels(rows, columns):
    """A grid of square image panels filling the documentation column."""
    side = PAGE_WIDTH / columns
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(PAGE_WIDTH, rows * side))
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


def compare(truth, results, crop, gain=3.0):
    """The reference and each result, whole, magnified on ``crop``, and their errors.

    Row one holds whole images, row two the region ``crop`` magnified, row
    three the magnitude error multiplied by ``gain`` on the scale of the
    reference, with the NRMSE of each result.
    """
    columns = 1 + len(results)
    figure, axes = panels(3, columns)
    top = float(truth.abs().max())
    rows, cols = crop
    show(axes[0, 0], truth, "reference", vmax=top)
    axes[0, 0].add_patch(
        Rectangle(
            (cols.start, rows.start),
            cols.stop - cols.start,
            rows.stop - rows.start,
            fill=False,
            edgecolor="#e8a33d",
            linewidth=1.2,
        )
    )
    show(axes[1, 0], truth[crop], vmax=top)
    axes[2, 0].text(
        0.5, 0.5, f"error\n× {gain:g}", ha="center", va="center", transform=axes[2, 0].transAxes
    )
    for column, (name, made) in enumerate(results.items(), start=1):
        show(axes[0, column], made, name, vmax=top)
        show(axes[1, column], made[crop], vmax=top)
        show(
            axes[2, column],
            gain * (made.abs() - truth.abs()),
            f"NRMSE {nrmse(made, truth):.3f}",
            vmax=top,
            cmap="magma",
        )
    return figure


# sphinx_gallery_end_ignore
import csv
import logging
from pathlib import Path

import brainweb_dl
import lightning
import numpy as np
import torch
from brainweb_dl import get_mri
from monai.metrics import PSNRMetric, SSIMMetric
from torch.utils.data import DataLoader

import bartorch
import bartorch.tools as bt
from bartorch import learning, linop, optim, priors
from bartorch.learning import training

SIZE = 96
COILS = 8
ITERATIONS = 4
ACCELERATION = 4
EPOCHS = 16

_ = torch.manual_seed(0)

# %%
#
# Data
# ----
#
# The slices, coils and fourfold undersampling of :doc:`04-staged-training`:
# subject 0 to train on and subject 4 to validate on. The references are kept
# only to score the results; the self-supervised network never sees them.

# sphinx_gallery_start_ignore
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
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
train_images = brain_slices(subject=0, count=24)
valid_images = brain_slices(subject=4, count=8)

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)

density = torch.exp(-0.5 * ((torch.arange(SIZE) - SIZE / 2) / (SIZE / 6)) ** 2)
lines = torch.rand(SIZE, generator=torch.Generator().manual_seed(1)) < density / density.sum() * (
    SIZE / ACCELERATION
)
lines[SIZE // 2 - 4 : SIZE // 2 + 4] = True
pattern = lines.to(torch.complex64)[:, None].expand(SIZE, SIZE).contiguous()
A = linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=pattern)
NOISE = 0.02

generator = torch.Generator().manual_seed(3)
kspace = {
    "train": [
        A(x) + NOISE * torch.randn(A.oshape, dtype=torch.complex64, generator=generator)
        for x in train_images
    ],
    "valid": [
        A(x) + NOISE * torch.randn(A.oshape, dtype=torch.complex64, generator=generator)
        for x in valid_images
    ],
}

# %%
#
# The split
# ---------
#
# The readout is fully sampled, so the unit of the split is the phase-encode
# line: the pattern given to :func:`~bartorch.learning.split` has one entry per
# line and broadcasts over the coils and the readout. A quarter of the acquired
# lines are held out, drawn with a Gaussian density across :math:`k_y`, and
# the eight central lines always stay in :math:`\Theta`: a reconstruction
# without the centre of k-space would lose the image contrast, and the loss
# would be dominated by it.

acquired = lines.float()[:, None]
keep, held = learning.split(acquired, 0.25, keep=(8, 1), generator=torch.Generator().manual_seed(0))
print(
    f"{int(acquired.sum())} acquired lines: {int(keep.sum())} to reconstruct from, "
    f"{int(held.sum())} held out"
)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 1.6))
for lane, (row, colour) in enumerate(
    zip((acquired[:, 0], keep[:, 0], held[:, 0]), ("0.55", "#4c9be8", "#e8a33d"))
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
#
# Two networks, one trained each way
# ----------------------------------
#
# Both are the iteration-conditioned unrolled network of the previous lesson,
# trained end to end for the same number of epochs from the same
# initialization; only the items differ. A self-supervised item carries the
# acquired ``pattern`` in place of a ``target``, and
# :class:`~bartorch.learning.training.Reconstruction` then draws a split at
# every step. Its validation loss is the held-out loss on a split fixed for
# the whole run, and needs no reference either.


def unrolled():
    torch.manual_seed(0)
    network = learning.UNet(2, spatial=2, widths=(16, 32, 64), steps=True)
    prior = priors.ImplicitPrior(learning.ComplexNet(network, spatial=2), step=True)
    block = optim.ISTBlock(prior, step=1.0)
    block.step.requires_grad_()
    return learning.Unrolled(block, iterations=ITERATIONS, checkpoint=True)


def items(part, supervised):
    images = train_images if "train" == part else valid_images
    made = []
    for x, y in zip(images, kspace[part]):
        item = {"y": y, "A": A}
        item.update({"target": x} if supervised else {"pattern": acquired})
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
    trainer.fit(
        training.Reconstruction(
            models[name], "end-to-end", lr=1e-3, fraction=0.25, split_options={"keep": (8, 1)}
        ),
        DataLoader(items("train", supervised), batch_size=4, shuffle=True, collate_fn=list),
        DataLoader(items("valid", supervised), batch_size=4, collate_fn=list),
    )

# %%
#
# Results
# -------
#
# Both networks reconstruct from all acquired lines of the validation subject
# and are scored against its reference.

psnr = PSNRMetric(max_val=1.0)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0)
truth = torch.stack(valid_images).abs()[:, None]

with torch.no_grad():
    results = {name: torch.stack([m(y, A) for y in kspace["valid"]]) for name, m in models.items()}
results["CG SENSE, 20 iterations"] = torch.stack(
    [optim.cg(y, A, maxiter=20) for y in kspace["valid"]]
)
for name, made in results.items():
    a = made.abs()[:, None]
    print(
        f"{name:>24}   PSNR {float(psnr(a, truth).mean()):5.2f} dB   "
        f"SSIM {float(ssim(a, truth).mean()):.3f}"
    )

# %%
#
# The self-supervised network is trained on less information: each step
# reconstructs from three quarters of the acquired lines and is told nothing
# about the lines never acquired, which is where the supervised network learns
# most. The gap between the two is what a fully sampled reference would have
# bought; the self-supervised network needs nothing beyond the data the
# protocol already acquires.
#
# In the images below, both networks suppress the noise that the CG-SENSE
# unfolding amplifies across the whole field of view. The self-supervised
# network keeps more residual aliasing along the phase-encode direction
# (vertical), which its error map shows as horizontal striping: the lines never
# acquired are the ones it cannot score against.

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
