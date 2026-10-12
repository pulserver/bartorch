r"""
======================
Uncertainty estimation
======================

A learned reconstruction returns an image without saying where it may be
wrong. Where the undersampling leaves the image underdetermined, the network
fills in what its training data suggest, and a hallucinated structure looks
like real anatomy. In this tour you attach a voxel-wise error bar to the
unrolled reconstruction of :doc:`02-staged-training`, calibrate it so that it
contains the true error in a stated fraction of voxels, and look at where in
the head the reconstruction is least certain.

The method has two parts. First, you randomize the reconstruction and
repeat it: leave dropout active in the network (Monte Carlo dropout
[#gal]_), reconstruct from random subsets of the acquired phase encodes, or
shift a patch grid. The spread of the repeats measures one source of
variability, and none of them is the error itself. Second, split conformal
calibration [#angelopoulos]_ relates the spread to the error on held-out
slices with fully sampled references: it finds the factor by which you must
multiply the spread for the interval to contain the error at a chosen rate,
a guarantee that holds whatever the spread measures.

Prerequisites: lesson 11, :doc:`../06-learning/02-modl-with-admm`, and the
tour :doc:`02-staged-training`.

**Learning objectives**

- Obtain a spread from Monte Carlo dropout and from k-space subsets with
  :func:`bartorch.learning.moments`.
- Calibrate it to a coverage with :func:`bartorch.learning.calibrate`, and
  check the coverage on other slices.
- Compare the calibrated interval with the error made.
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
# :doc:`02-staged-training`, with BrainWeb [#brainweb]_ subject 0 to train on
# and sixteen slices of subject 4: the first eight to calibrate on, the last
# eight to check the calibration on.
import lightning
import torch

import bartorch
import bartorch.tools as bt

from bartorch import learning, linop, optim, priors
from torch.utils.data import DataLoader

COILS = 8
ITERATIONS = 4
ACCELERATION = 4
EPOCHS = 8
DROPOUT = 0.1
NOISE = 0.02

_ = torch.manual_seed(0)

train_images = brain_slices(subject=0, count=24)
valid_images = brain_slices(subject=4, count=16)

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
# Monte Carlo dropout
# -------------------
#
# The network is the iteration-conditioned unrolled network of
# :doc:`02-staged-training`, with dropout in every residual block of its
# U-Net, trained end to end against references. During training dropout is a
# regularizer; left active at inference it makes each reconstruction one draw
# from a family of networks.
torch.manual_seed(0)
network = learning.UNet(
    2,
    spatial=2,
    widths=(16, 32, 64),
    steps=True,
    dropout=DROPOUT,
)
prior = priors.ImplicitPrior(learning.ComplexNet(network, spatial=2), step=True)
block = optim.ISTBlock(prior, step=1.0)
block.step.requires_grad_()
model = learning.Unrolled(block, iterations=ITERATIONS, checkpoint=True)

pairs = {
    "train": zip(train_images, kspace["train"]),
    "valid": zip(valid_images[:8], kspace["valid"]),
}
train_items = [{"y": y, "A": A, "target": x} for x, y in pairs["train"]]
valid_items = [{"y": y, "A": A, "target": x} for x, y in pairs["valid"]]
trainer = lightning.Trainer(
    max_epochs=EPOCHS,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)
trainer.fit(
    learning.Reconstruction(model, "end-to-end", lr=1e-3),
    DataLoader(train_items, batch_size=4, shuffle=True, collate_fn=list),
    DataLoader(valid_items, batch_size=4, collate_fn=list),
)

# %%
# Reconstruction spread
# ---------------------
#
# :func:`~bartorch.learning.moments` calls a randomized reconstruction a
# number of times and returns the voxel-wise mean and variance. Your first
# randomized reconstruction leaves the dropout modules in training mode and
# everything else in evaluation mode:
model.eval()
dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]


def with_dropout(y):
    for module in dropouts:
        module.train()
    try:
        return model(y, A)
    finally:
        for module in dropouts:
            module.eval()


# %%
# The second keeps the network deterministic and reconstructs each time from a
# random eighty per cent of the acquired lines, drawn by
# :func:`~bartorch.learning.split`. It measures how much the image depends on
# individual samples.
acquired = lines.float()[:, None]


def from_a_subset(y):
    keep, _ = learning.split(acquired, 0.2, keep=(8, 1))
    kept = (pattern * keep).to(torch.complex64)
    subset = linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=kept)
    return model(y * keep, subset)


spreads = {}
for name, reconstruct in (
    ("dropout", with_dropout),
    ("k-space subsets", from_a_subset),
):
    means, deviations = [], []
    for y in kspace["valid"]:
        mean, variance = learning.moments(lambda: reconstruct(y), samples=8)
        means.append(mean)
        deviations.append(variance.sqrt())
    spreads[name] = (torch.stack(means), torch.stack(deviations))

truth = torch.stack(valid_images)
head = truth.abs() > 0.05

# %%
# Calibration
# -----------
#
# On the first eight slices, whose references you know,
# :func:`~bartorch.learning.calibrate` finds the factor that makes
# ``|error| <= factor * spread`` hold for ninety per cent of the voxels in the
# head. On the other eight you measure the fraction of voxels whose error
# falls within ``factor * spread``. Split conformal calibration guarantees
# that fraction on average over voxels and subjects drawn alike, not voxel by
# voxel.
COVERAGE = 0.9
calibration, test = slice(0, 8), slice(8, 16)


def calibrated(mean, deviation, coverage=COVERAGE):
    """The factor, and the coverage it reaches on the test slices."""
    error = (mean - truth).abs()
    on, off = head[calibration], head[test]
    factor = learning.calibrate(
        error[calibration][on],
        deviation[calibration][on],
        coverage,
    )
    inside = error[test] <= factor * deviation[test]
    return factor, float(inside[off].float().mean())


# sphinx_gallery_start_ignore
for name, (mean, deviation) in spreads.items():
    factor, reached = calibrated(mean, deviation)
    error = (mean - truth).abs()
    correlation = torch.corrcoef(
        torch.stack([error[test][head[test]], deviation[test][head[test]]])
    )[0, 1]
    print(
        f"{name:>16}: factor {factor:6.2f}, coverage {reached:.3f}"
        f" (asked {COVERAGE}), correlation of error and spread {float(correlation):.2f}"
    )
# sphinx_gallery_end_ignore

# %%
# Both spreads reach close to 90 per cent coverage on the test slices,
# although their factors differ by more than a factor of two: the calibration
# absorbs whatever scale the spread has. What differs between them is how well
# the spread follows the error voxel by voxel, which the correlation measures
# and the maps below show. A spread that is large where the error is large
# gives narrow intervals where the reconstruction is reliable and wide ones
# where it is not; a spread unrelated to the error gives intervals of the right
# average width in the wrong places.

# sphinx_gallery_start_ignore
row = 12
figure, axes = panels(2, 2, bar=True)
top = float(truth[row].abs().max())
show(axes[0, 0], spreads["dropout"][0][row], "dropout mean", vmax=top)
error = (spreads["dropout"][0][row] - truth[row]).abs()
scale = float(error.max())
handle = show(axes[0, 1], error, "|error|", vmax=scale, cmap="magma")
figure.colorbar(
    handle, ax=axes.ravel().tolist(), fraction=0.046, label="|error|, interval half-width"
)
for column, name in enumerate(spreads):
    mean, deviation = spreads[name]
    factor, _ = calibrated(mean, deviation)
    show(
        axes[1, column], factor * deviation[row], f"90% interval, {name}", vmax=scale, cmap="magma"
    )
plt.show()
# sphinx_gallery_end_ignore

# %%
# The two spreads measure different things. The dropout interval is diffuse
# over the brain and follows neither its anatomy nor the error. The
# k-space-subset interval is largest at the scalp and in horizontal bands, the
# pattern of aliasing along the phase-encode direction (vertical): removing
# lines moves the aliasing, and that is the variability it records. The error
# itself is concentrated in the cortex. Both correlations are weak, about 0.2,
# so the intervals are wider than the error over much of the white matter and
# narrower than it in parts of the cortex. The coverage holds on average over
# voxels, as the calibration guarantees, not voxel by voxel.
#
# Exercise: target coverage
# -------------------------
#
# The calibration works for any coverage you ask for. Calibrate the
# k-space-subset spread at 50, 80 and 95 per cent and check what the test
# slices reach; nothing is reconstructed again:
mean, deviation = spreads["k-space subsets"]
for coverage in (0.5, 0.8, 0.95):
    factor, reached = calibrated(mean, deviation, coverage)
    print(f"asked {coverage:.2f}: factor {factor:5.2f}, reached {reached:.3f}")

# %%
# The coverage reached on the test slices follows the one asked for at every
# level, and the factor grows with it: the calibration is a quantile of the
# ratio of error to spread on the calibration slices, read off at the level
# you ask for.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Train the 4-iteration unrolled ISTA network of the staged-training tour
#    with dropout 0.1 in its UNet, end to end for 8 epochs, on 2D Cartesian
#    SENSE data (8 coils, R = 4, noise 0.02). On 16 test slices, compute a
#    voxel-wise mean and standard deviation with learning.moments over 8
#    draws, twice: with only the dropout modules in training mode, and from
#    random 80 per cent subsets of the acquired lines (learning.split, 8
#    central lines kept). Calibrate each with learning.calibrate to 90 per
#    cent coverage of the head voxels on the first 8 slices, and report the
#    coverage reached on the other 8 and the error-spread correlation.

# %%
#
# References
# ----------
#
# .. [#gal] Gal Y, Ghahramani Z. Dropout as a Bayesian approximation:
#    representing model uncertainty in deep learning. *Proc Int Conf Mach
#    Learn* 48:1050-1059 (2016). https://proceedings.mlr.press/v48/gal16.html
#
# .. [#angelopoulos] Angelopoulos AN, Bates S. Conformal prediction: a gentle
#    introduction. *Found Trends Mach Learn* 16(4):494-591 (2023).
#    https://doi.org/10.1561/2200000101
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
