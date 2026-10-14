r"""
======================
Annealed plug-and-play
======================

In lesson 10 you plugged a denoiser someone else had trained, on
photographs, into ADMM at one noise level you chose. In this tour you train
the denoiser yourself, on MR images alone and without any encoding, and use
it as the regularizer of an ADMM reconstruction whose denoising strength
decreases over the iterations. You then apply the same network, unchanged, at an
acceleration where CG-SENSE breaks down.

A plug-and-play reconstruction [#pnp]_ replaces the proximal step of an
iterative algorithm by a denoiser. Because the denoiser is trained on images,
not on k-space, one network serves every protocol whose images resemble its
training images: a change of acceleration, sampling pattern or coil array
needs no retraining. The proximal step of a regularizer :math:`\lambda\phi`
with ADMM penalty :math:`\rho` is a Gaussian denoiser of noise variance
:math:`\sigma^2 = \lambda/\rho`. The first iterates carry the strong
incoherent aliasing of the undersampling and the last are nearly consistent
with the data, so a noise level that decreases from one to the other [#dpir]_
removes the aliasing first and preserves fine anatomy at the end. The penalty
follows as :math:`\rho_k = \lambda/\sigma_k^2`, which keeps the balance
between data consistency and denoising that :math:`\lambda` sets.

Prerequisites: lesson 10, :doc:`../06-learning/01-plug-and-play`, and
lesson 11, :doc:`../06-learning/02-modl-with-admm`.

**Learning objectives**

- Train a denoiser conditioned on the noise level, ``noise=True`` in
  :class:`bartorch.learning.UNet`, with
  :class:`bartorch.learning.Reconstruction`.
- Give :class:`bartorch.priors.ImplicitPrior` a schedule of noise levels and
  :class:`bartorch.optim.ADMMBlock` the matching schedule of penalties.
- Compare an annealed schedule with a fixed noise level, iteration by
  iteration, and apply the same denoiser at a higher acceleration.
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
# :doc:`02-staged-training`: BrainWeb [#brainweb]_ subject 0 to train the
# denoiser on, subject 4 to reconstruct. The acquisition is wrapped in a
# function, because you will change the acceleration at the end.
import lightning
import torch

import bartorch
import bartorch.tools as bt

from bartorch import learning, linop, optim, priors
from monai.metrics import PSNRMetric
from torch.utils.data import DataLoader

COILS = 8
ITERATIONS = 12
NOISE = 0.02

_ = torch.manual_seed(0)

train_images = brain_slices(subject=0, count=24)
valid_images = brain_slices(subject=4, count=8)

coils = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = coils / bartorch.rss(coils, axes=(0,), keepdim=True)


def acquisition(acceleration, seed):
    """Cartesian SENSE, one line in ``acceleration`` on average."""
    ky = torch.arange(SIZE) - SIZE / 2
    density = torch.exp(-0.5 * (ky / (SIZE / 6)) ** 2)
    chance = density / density.sum() * (SIZE / acceleration)
    draw = torch.rand(SIZE, generator=torch.Generator().manual_seed(seed))
    lines = draw < chance
    lines[SIZE // 2 - 4 : SIZE // 2 + 4] = True
    pattern = lines.to(torch.complex64)[:, None].expand(SIZE, SIZE)
    pattern = pattern.contiguous()
    return linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=pattern)


def measure(A, generator):
    made = []
    for x in valid_images:
        shape, kind = A.oshape, torch.complex64
        noise = torch.randn(shape, dtype=kind, generator=generator)
        made.append(A(x) + NOISE * noise)
    return made


A = acquisition(4, seed=1)
kspace = measure(A, torch.Generator().manual_seed(3))

# %%
# Noise-conditioned denoiser
# --------------------------
#
# With ``noise=True`` the U-Net takes the noise level as an input and
# modulates its features with it, so one network covers a range of SNRs. You
# train it on pairs of a slice and the same slice with complex white Gaussian
# noise, at a standard deviation drawn log-uniformly between 0.5 and 20 per
# cent of the image's peak for each pair
# (:class:`~bartorch.learning.ComplexNet` scales each image to unit peak). No
# coil sensitivities, sampling pattern or k-space enter the training.
LOW, HIGH = 0.005, 0.2

network = learning.UNet(2, spatial=2, widths=(16, 32, 64), noise=True)
denoiser = learning.ComplexNet(network, spatial=2)


def pairs(images):
    made = []
    for x in images:
        sigma = LOW * (HIGH / LOW) ** torch.rand(())
        noisy = x + sigma * torch.randn_like(x)
        made.append({"input": noisy, "target": x, "sigma": sigma.reshape(1)})
    return made


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
    DataLoader(train_images, batch_size=4, shuffle=True, collate_fn=pairs),
    DataLoader(valid_images, batch_size=4, collate_fn=pairs),
)

# %%
# You score every image by the mean PSNR of its magnitude over the eight
# slices of subject 4. First the denoiser on its own, at three noise levels:
psnr = PSNRMetric(max_val=1.0)
truth = torch.stack(valid_images).abs()[:, None]


def score(images):
    return float(psnr(images.abs()[:, None], truth).mean())


with torch.no_grad():
    for sigma in (0.01, 0.05, 0.1):
        noisy = torch.stack(valid_images)
        noisy = noisy + sigma * torch.randn_like(noisy)
        level = torch.tensor([sigma])
        denoised = torch.stack([denoiser(x[None], level)[0] for x in noisy])
        before, after = score(noisy), score(denoised)
        print(f"sigma {sigma}: {before:5.2f} dB noisy, {after:5.2f} denoised")

# %%
# The gain grows with the noise level: about 2 dB at 0.01, about 8 dB at 0.1.
# One set of weights covers the whole range, which is what lets the noise
# level change from one iteration to the next below.
#
# Annealing schedules
# -------------------
#
# :class:`~bartorch.priors.ImplicitPrior` takes a sequence of noise levels,
# one per iteration, and :class:`~bartorch.optim.ADMMBlock` a sequence of
# penalties; the last value of each is repeated past its end. The annealed
# schedule decreases the noise level geometrically from 0.1 to 0.01 of the
# peak over twelve iterations, and the penalty follows as
# :math:`\rho_k = \lambda/\sigma_k^2`. When the penalty changes, the block
# rescales ADMM's scaled dual variable by the ratio of the old penalty to the
# new, so the unscaled one carries over. The two fixed schedules hold the
# noise level at 0.03 and at 0.01, with the penalty given by the same
# :math:`\lambda`.
LAMBDA = 1e-4

schedules = {
    "annealed, 0.1 to 0.01": torch.logspace(-1.0, -2.0, ITERATIONS),
    "fixed, 0.03": torch.full((ITERATIONS,), 0.03),
    "fixed, 0.01": torch.full((ITERATIONS,), 0.01),
}


def plug_and_play(sigma, lam=LAMBDA):
    prior = priors.ImplicitPrior(denoiser, sigma=sigma.tolist())
    rho = (lam / sigma**2).tolist()
    block = optim.ADMMBlock(prior, rho=rho, cg_maxiter=5)
    return learning.Unrolled(block, iterations=ITERATIONS)


# %%
# :meth:`~bartorch.learning.Unrolled.steps` yields every iterate, so you can
# follow each schedule iteration by iteration:
curves, results = {}, {}
with torch.no_grad():
    for name, sigma in schedules.items():
        runs = [list(plug_and_play(sigma).steps(y, A)) for y in kspace]
        iterates = [torch.stack(images) for images in zip(*runs)]
        curves[name] = [score(images) for images in iterates]
        results[name] = iterates[-1]
cg = torch.stack([optim.cg(y, A, maxiter=20) for y in kspace])
results["CG SENSE, 20 iterations"] = cg

# sphinx_gallery_start_ignore
for name, images in results.items():
    print(f"{name:>24}   PSNR {score(images):5.2f} dB")

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
for name, curve in curves.items():
    axis.plot(range(1, ITERATIONS + 1), curve, marker="o", label=name)
axis.axhline(score(results["CG SENSE, 20 iterations"]), color="0.55", ls="--", label="CG-SENSE")
axis.set_xlabel("iteration")
axis.set_ylabel("PSNR (dB)")
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The fixed level of 0.03 converges within about four iterations, to an image
# limited by the smoothing the denoiser applies at that level. The fixed level
# of 0.01 preserves more detail, but its large penalty makes each ADMM step
# move little from the previous one: it starts far below CG-SENSE and is still
# rising after twelve iterations. The annealed schedule is ahead from the
# first iteration, and ends slightly ahead of the better fixed level, without
# a level having been tuned for this acquisition.
#
# In the error maps of one slice, the CG-SENSE error is spread over the whole
# field of view; the plug-and-play error is confined to the head and largest
# at the cortex.

# sphinx_gallery_start_ignore
compare(
    valid_images[5],
    {
        "CG-SENSE": results["CG SENSE, 20 iterations"][5],
        "annealed plug-and-play": results["annealed, 0.1 to 0.01"][5],
    },
    crop=(slice(22, 58), slice(30, 66)),
)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Transfer to another acquisition
# -------------------------------
#
# You trained the denoiser without an encoding, so it applies unchanged to
# :math:`R = 6`, a sampling pattern it has never been used with:
A6 = acquisition(6, seed=2)
kspace6 = measure(A6, torch.Generator().manual_seed(4))
annealed_pnp = plug_and_play(schedules["annealed, 0.1 to 0.01"])
with torch.no_grad():
    annealed = torch.stack([annealed_pnp(y, A6) for y in kspace6])
cg6 = torch.stack([optim.cg(y, A6, maxiter=20) for y in kspace6])
print(f"R = 6: annealed plug-and-play {score(annealed):5.2f} dB")
print(f"R = 6: CG SENSE {score(cg6):5.2f} dB")

# sphinx_gallery_start_ignore
compare(
    valid_images[5],
    {"CG-SENSE, R = 6": cg6[5], "annealed, R = 6": annealed[5]},
    crop=(slice(22, 58), slice(30, 66)),
)
plt.show()
# sphinx_gallery_end_ignore

# %%
# At this acceleration eight coils no longer unfold the aliasing well:
# CG-SENSE loses about 2 dB with respect to :math:`R = 4`, and its error map
# shows the folded outline of the head on top of the g-factor noise. The
# plug-and-play reconstruction loses a fraction of a decibel and keeps the
# anatomy.
#
# Exercise: the weight :math:`\lambda`
# ------------------------------------
#
# :math:`\lambda` sets the weight of the denoising step relative to data
# consistency at every noise level. Rerun the annealed schedule at
# :math:`R = 4` with a tenth and ten times the weight above; the denoiser stays
# as it is:
for lam in (LAMBDA / 10, LAMBDA, LAMBDA * 10):
    pnp = plug_and_play(schedules["annealed, 0.1 to 0.01"], lam=lam)
    with torch.no_grad():
        made = torch.stack([pnp(y, A) for y in kspace])
    print(f"lambda {lam:.0e}: PSNR {score(made):5.2f} dB")

# %%
# Both a tenth and ten times the weight score more than 1 dB below
# :math:`\lambda = 10^{-4}`. A smaller :math:`\lambda` lowers every penalty
# :math:`\rho_k`, so the denoiser's output enters the data-consistency step
# with less weight; a larger one weights it more, and the reconstruction moves
# away from the measured data.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Train a noise-conditioned denoiser, ComplexNet(UNet(2, spatial=2,
#    widths=(16, 32, 64), noise=True)), on clean complex brain slices with
#    complex Gaussian noise of a log-uniform level in [0.005, 0.2], 30 epochs
#    of learning.Reconstruction(..., "denoiser", lr=2e-3). Use it as the
#    proximal step of 12 ADMM iterations (optim.ADMMBlock, 5 CG steps) for 2D
#    Cartesian SENSE at R = 4: priors.ImplicitPrior with sigma decreasing
#    geometrically from 0.1 to 0.01 and rho_k = 1e-4 / sigma_k^2. Compare it
#    with fixed sigma 0.03 and 0.01 and with CG-SENSE by PSNR per iteration,
#    and rerun it unchanged at R = 6.

# %%
#
# References
# ----------
#
# .. [#pnp] Venkatakrishnan SV, Bouman CA, Wohlberg B. Plug-and-play priors for
#    model based reconstruction. *IEEE Global Conference on Signal and
#    Information Processing* 945-948 (2013).
#    https://doi.org/10.1109/GlobalSIP.2013.6737048
#
# .. [#dpir] Zhang K, Li Y, Zuo W, Zhang L, Van Gool L, Timofte R. Plug-and-play
#    image restoration with deep denoiser prior. *IEEE Trans Pattern Anal Mach
#    Intell* 44(10):6360-6376 (2022). https://doi.org/10.1109/TPAMI.2021.3088914
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
