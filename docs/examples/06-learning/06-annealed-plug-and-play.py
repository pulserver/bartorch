r"""
===========================
Annealed plug-and-play
===========================

A denoiser conditioned on the noise level, trained once on images and used in
ADMM with a noise level that decreases over the iterations.

A plug-and-play reconstruction [#pnp]_ uses a denoiser as the proximal step of
an iteration, and the denoiser is trained on images alone, so one network
serves every acquisition whose images resemble its training images. The
proximal step of :math:`\lambda\phi` with penalty :math:`\rho` is a Gaussian
denoiser of variance :math:`\sigma^2 = \lambda/\rho`. Early iterations start
from an image corrupted by undersampling artefacts, far from the solution, and
late iterations from a nearly consistent one; a noise level decreasing from
one to the other [#dpir]_ removes the artefacts first and retains detail at
the end. The penalty follows as :math:`\rho_k = \lambda/\sigma_k^2`, so that
the denoiser's noise level and the weight of the data remain in the ratio the
regularization weight :math:`\lambda` sets.

**Learning objectives**

- Train a denoiser conditioned on the noise level, ``noise=True`` in
  :class:`bartorch.learning.UNet`, with
  :class:`bartorch.learning.training.Reconstruction`.
- Give :class:`bartorch.priors.ImplicitPrior` a schedule of noise levels and
  :class:`bartorch.optim.ADMMBlock` the matching schedule of penalties.
- Compare an annealed schedule with a fixed noise level, iteration by
  iteration, and apply the same denoiser to another undersampling.

It follows :doc:`05-self-supervised-training`. The next lesson,
:doc:`07-uncertainty`, attaches error bars to a learned reconstruction.
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
from brainweb_dl import get_mri
from monai.metrics import PSNRMetric
from torch.utils.data import DataLoader

import bartorch
import bartorch.tools as bt
from bartorch import learning, linop, optim, priors
from bartorch.learning import training

SIZE = 96
COILS = 8
ITERATIONS = 12

_ = torch.manual_seed(0)

# %%
#
# Data
# ----
#
# The slices, coils and fourfold undersampling of :doc:`04-staged-training`:
# subject 0 to train the denoiser on, subject 4 to reconstruct.

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
NOISE = 0.005


def acquisition(acceleration, seed):
    """A Cartesian SENSE encoding sampling one line in ``acceleration`` on average."""
    density = torch.exp(-0.5 * ((torch.arange(SIZE) - SIZE / 2) / (SIZE / 6)) ** 2)
    chance = density / density.sum() * (SIZE / acceleration)
    lines = torch.rand(SIZE, generator=torch.Generator().manual_seed(seed)) < chance
    lines[SIZE // 2 - 4 : SIZE // 2 + 4] = True
    pattern = lines.to(torch.complex64)[:, None].expand(SIZE, SIZE).contiguous()
    return linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=pattern)


def measure(A, generator):
    return [
        A(x) + NOISE * torch.randn(A.oshape, dtype=torch.complex64, generator=generator)
        for x in valid_images
    ]


A = acquisition(4, seed=1)
kspace = measure(A, torch.Generator().manual_seed(3))

# %%
#
# A denoiser for every noise level
# --------------------------------
#
# With ``noise=True`` the U-Net takes the noise level as an input and
# modulates its features with it, so one network denoises across a range of
# levels. It is trained on pairs of a slice and the slice with white Gaussian
# noise added, at a level drawn log-uniformly for each pair, in units of the
# slice's peak (:class:`~bartorch.learning.ComplexNet` scales each image to
# unit peak). No encoding enters the training.

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
    training.Reconstruction(denoiser, "denoiser", lr=2e-3),
    DataLoader(train_images, batch_size=4, shuffle=True, collate_fn=pairs),
    DataLoader(valid_images, batch_size=4, collate_fn=pairs),
)

psnr = PSNRMetric(max_val=1.0)
truth = torch.stack(valid_images).abs()[:, None]


def score(images):
    return float(psnr(images.abs()[:, None], truth).mean())


with torch.no_grad():
    for sigma in (0.01, 0.05, 0.1):
        noisy = torch.stack(valid_images)
        noisy = noisy + sigma * torch.randn_like(noisy)
        denoised = torch.stack([denoiser(x[None], torch.tensor([sigma]))[0] for x in noisy])
        print(
            f"sigma {sigma:4.2f}: noisy {score(noisy):5.2f} dB, denoised {score(denoised):5.2f} dB"
        )

# %%
#
# Schedules of noise level and penalty
# ------------------------------------
#
# :class:`~bartorch.priors.ImplicitPrior` takes a sequence of noise levels,
# one per iteration, and :class:`~bartorch.optim.ADMMBlock` a sequence of
# penalties; the last value of each is repeated past its end. The annealed
# schedule decreases the noise level geometrically from 0.1 to 0.01 of the
# peak over twelve iterations, and the penalty follows as
# :math:`\rho_k = \lambda/\sigma_k^2`. When the penalty changes, the block
# rescales ADMM's scaled dual variable by the ratio of the old penalty to the
# new, so that the unscaled one carries over. The two fixed schedules hold the
# noise level at either end of the annealed one, with the penalty given by the
# same :math:`\lambda`.

LAMBDA = 1e-4

schedules = {
    "annealed, 0.1 to 0.01": torch.logspace(-1.0, -2.0, ITERATIONS),
    "fixed, 0.03": torch.full((ITERATIONS,), 0.03),
    "fixed, 0.01": torch.full((ITERATIONS,), 0.01),
}


def plug_and_play(sigma):
    prior = priors.ImplicitPrior(denoiser, sigma=sigma.tolist())
    block = optim.ADMMBlock(prior, rho=(LAMBDA / sigma**2).tolist(), cg_maxiter=5)
    return learning.Unrolled(block, iterations=ITERATIONS)


curves, results = {}, {}
with torch.no_grad():
    for name, sigma in schedules.items():
        runs = [list(plug_and_play(sigma).steps(y, A)) for y in kspace]
        iterates = [torch.stack([run[k] for run in runs]) for k in range(ITERATIONS)]
        curves[name] = [score(images) for images in iterates]
        results[name] = iterates[-1]
    results["CG SENSE, 20 iterations"] = torch.stack([optim.cg(y, A, maxiter=20) for y in kspace])

for name, images in results.items():
    print(f"{name:>24}   PSNR {score(images):5.2f} dB")

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
for name, curve in curves.items():
    axis.plot(range(1, ITERATIONS + 1), curve, marker="o", label=name)
axis.axhline(score(results["CG SENSE, 20 iterations"]), color="gray", ls="--", label="CG SENSE")
axis.set_xlabel("iteration")
axis.set_ylabel("PSNR (dB)")
axis.legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# A large fixed noise level converges within a few iterations to a
# reconstruction limited by the smoothing the denoiser applies at that level;
# a small one retains detail, but its large penalty makes each step move
# little from the previous one, and twelve iterations do not reach its fixed
# point. The annealed schedule takes the large steps first and the small ones
# last, and ends slightly ahead of the better fixed level without that level
# having to be chosen for the acquisition.

# sphinx_gallery_start_ignore
figure, axes = panels(1, 4)
top = float(truth[5].max())
show(axes[0, 0], valid_images[5], "reference", vmax=top)
for column, name in enumerate(
    ("annealed, 0.1 to 0.01", "fixed, 0.03", "CG SENSE, 20 iterations"), start=1
):
    show(axes[0, column], results[name][5], name.split(",")[0], vmax=top)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Another acquisition, the same denoiser
# --------------------------------------
#
# The denoiser was trained without an encoding, so it applies unchanged to a
# sixfold undersampling it has never been used with.

A6 = acquisition(6, seed=2)
kspace6 = measure(A6, torch.Generator().manual_seed(4))
with torch.no_grad():
    annealed = torch.stack(
        [plug_and_play(schedules["annealed, 0.1 to 0.01"])(y, A6) for y in kspace6]
    )
    cg = torch.stack([optim.cg(y, A6, maxiter=20) for y in kspace6])
print(f"sixfold: annealed plug-and-play {score(annealed):5.2f} dB, CG SENSE {score(cg):5.2f} dB")

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
