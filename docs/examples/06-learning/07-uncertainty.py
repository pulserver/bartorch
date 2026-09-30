r"""
=======================
Uncertainty estimation
=======================

**Aim.** Attach a voxel-wise error bar to a learned reconstruction of
undersampled data, calibrated so that it contains the true error in a stated
fraction of voxels, and see where in the head the reconstruction is least
certain.

A learned reconstruction returns an image without saying where it may be
wrong. Where the undersampling leaves the image underdetermined, the network
fills in what its training data suggest, and a hallucinated structure looks
like real anatomy. An uncertainty map is obtained by randomizing the
reconstruction and repeating it: leaving dropout active in the network (Monte
Carlo dropout [#gal]_), reconstructing from random subsets of the acquired
phase encodes, or shifting a patch grid. Each spread measures one source of
variability, and none is the error itself. Split conformal calibration
[#angelopoulos]_ relates the spread to the error on held-out slices with
fully sampled references: it finds the factor by which the spread must be
multiplied for the interval to contain the error at a chosen rate, a
guarantee that holds whatever the spread measures.

**Learning objectives**

- Obtain a spread from Monte Carlo dropout and from k-space subsets with
  :func:`bartorch.learning.moments`.
- Calibrate it to a coverage with :func:`bartorch.learning.calibrate`, and
  check the coverage on other slices.
- Compare the calibrated interval with the error made.

It follows :doc:`06-annealed-plug-and-play`. This lesson ends the course; the
standalone examples of :doc:`../07-tours/index` apply the package to
individual problems.
"""
# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
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


# sphinx_gallery_end_ignore
import csv
import logging
from pathlib import Path

import brainweb_dl
import lightning
import numpy as np
import torch
from brainweb_dl import get_mri
from torch.utils.data import DataLoader

import bartorch
import bartorch.tools as bt
from bartorch import learning, linop, optim, priors

SIZE = 96
COILS = 8
ITERATIONS = 4
ACCELERATION = 4
EPOCHS = 8
DROPOUT = 0.1

_ = torch.manual_seed(0)

# %%
#
# Data
# ----
#
# The slices, coils and fourfold undersampling of :doc:`04-staged-training`,
# with sixteen slices of subject 4: the first eight to calibrate on, the last
# eight to check the calibration on.

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
valid_images = brain_slices(subject=4, count=16)

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
    part: [
        A(x) + NOISE * torch.randn(A.oshape, dtype=torch.complex64, generator=generator)
        for x in images
    ]
    for part, images in (("train", train_images), ("valid", valid_images))
}

# %%
#
# A network with dropout
# ----------------------
#
# The iteration-conditioned unrolled network of :doc:`04-staged-training`,
# with dropout in every residual block of its U-Net, trained end to end
# against references. Dropout is a regularizer during training; left active
# at inference it makes each reconstruction one draw from a family of
# networks.

torch.manual_seed(0)
network = learning.UNet(2, spatial=2, widths=(16, 32, 64), steps=True, dropout=DROPOUT)
prior = priors.ImplicitPrior(learning.ComplexNet(network, spatial=2), step=True)
block = optim.ISTBlock(prior, step=1.0)
block.step.requires_grad_()
model = learning.Unrolled(block, iterations=ITERATIONS, checkpoint=True)

items = {
    part: [{"y": y, "A": A, "target": x} for x, y in zip(images, kspace[part])]
    for part, images in (("train", train_images), ("valid", valid_images[:8]))
}
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
    DataLoader(items["train"], batch_size=4, shuffle=True, collate_fn=list),
    DataLoader(items["valid"], batch_size=4, collate_fn=list),
)

# %%
#
# Two spreads
# -----------
#
# :func:`~bartorch.learning.moments` calls a randomized reconstruction a
# number of times and returns the voxel-wise mean and variance. The first
# reconstruction leaves the dropout modules in training mode and everything
# else in evaluation mode. The second keeps the network deterministic and
# reconstructs each time from a random eighty per cent of the acquired lines,
# drawn by :func:`~bartorch.learning.split`, which measures how much the image
# depends on individual samples.

model.eval()
dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
acquired = lines.float()[:, None]


def with_dropout(y):
    for module in dropouts:
        module.train()
    try:
        return model(y, A)
    finally:
        for module in dropouts:
            module.eval()


def from_a_subset(y):
    keep, _ = learning.split(acquired, 0.2, keep=(8, 1))
    subset = linop.CartesianSense(
        sensitivities, (SIZE, SIZE), pattern=(pattern * keep).to(torch.complex64)
    )
    return model(y * keep, subset)


spreads = {}
for name, reconstruct in (("dropout", with_dropout), ("k-space subsets", from_a_subset)):
    means, deviations = [], []
    for y in kspace["valid"]:
        mean, variance = learning.moments(lambda: reconstruct(y), samples=8)
        means.append(mean)
        deviations.append(variance.sqrt())
    spreads[name] = (torch.stack(means), torch.stack(deviations))

truth = torch.stack(valid_images)
head = truth.abs() > 0.05

# %%
#
# Calibration
# -----------
#
# On the first eight slices, whose references are known, the factor that makes
# ``|error| <= factor * spread`` hold for ninety per cent of the voxels in the
# head is found with :func:`~bartorch.learning.calibrate`. On the other eight
# the fraction of voxels whose error falls within ``factor * spread`` is
# measured. Split conformal calibration guarantees that fraction on average
# over voxels and subjects drawn alike, not voxel by voxel.

COVERAGE = 0.9
calibration, test = slice(0, 8), slice(8, 16)
for name, (mean, deviation) in spreads.items():
    error = (mean - truth).abs()
    factor = learning.calibrate(
        error[calibration][head[calibration]], deviation[calibration][head[calibration]], COVERAGE
    )
    inside = error[test] <= factor * deviation[test]
    correlation = torch.corrcoef(
        torch.stack([error[test][head[test]], deviation[test][head[test]]])
    )[0, 1]
    print(
        f"{name:>16}: factor {factor:6.2f}, coverage {float(inside[head[test]].float().mean()):.3f}"
        f" (asked {COVERAGE}), correlation of error and spread {float(correlation):.2f}"
    )

# %%
#
# The coverage on the test slices is close to the one asked for, for both
# spreads, although their factors differ: the calibration absorbs whatever
# scale the spread has. What differs between them is how well the spread
# follows the error voxel by voxel, which the correlation measures and the
# maps below show. A spread that is large where the error is large gives
# narrow intervals where the reconstruction is reliable and wide ones where it
# is not; a spread unrelated to the error gives intervals of the right average
# width in the wrong places.
#
# The two spreads measure different things, and the maps show it. The
# dropout interval is diffuse over the brain and follows neither its anatomy
# nor the error. The k-space-subset interval is largest at the scalp and in
# horizontal bands, the pattern of aliasing along the phase-encode direction
# (vertical): removing lines moves the aliasing, and that is the variability
# it records. The error itself is concentrated in the cortex. Both correlations
# are weak, so the intervals are wider than the error over much of the white
# matter and narrower than it in parts of the cortex. The coverage is met on
# average over voxels, as the calibration guarantees, not voxel by voxel.

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
    factor = learning.calibrate(
        (mean - truth).abs()[calibration][head[calibration]],
        deviation[calibration][head[calibration]],
        COVERAGE,
    )
    show(
        axes[1, column], factor * deviation[row], f"90% interval, {name}", vmax=scale, cmap="magma"
    )
plt.show()
# sphinx_gallery_end_ignore

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
