r"""
===================================
Staged training of an unrolled network
===================================

An unrolled proximal-gradient network with one denoiser shared by every
iteration and told which iteration it is in, trained in three stages: the
denoiser alone, the iterations one at a time, and the whole stack.

The iteration is BART's iterative soft thresholding with the threshold
replaced by a network :math:`D_\theta`,

.. math::

   x^{k+1} = D_\theta\!\left(x^{k} - \tau\, A^H (A x^{k} - y),\; k\right),

a gradient step on the data term followed by the denoiser. One set of weights
serves every iteration, and the index :math:`k` enters the network by
feature-wise modulation (FiLM), so the network can remove the strong aliasing
of the first iterations and the residual noise of the last with the same
weights. The step :math:`\tau` is learned with them.

Training such a stack end to end stores every iteration's activations, which
for a three-dimensional volume exceeds a single card. The staged schedule of
Urman et al. [#urman]_ reaches the end-to-end optimum with a bounded memory:

1. **Denoiser pretraining.** The network is trained alone on degraded images
   paired with references, each assigned the iteration index its degradation
   corresponds to.
2. **Greedy training.** The stack is trained with a loss on each iteration's
   image, back-propagated before the next iteration runs, so memory holds one
   iteration [#gleam]_.
3. **End-to-end fine-tuning.** The loss is taken on the last image only,
   through the whole stack, with each iteration recomputed during the
   backward pass rather than stored.

:class:`bartorch.learning.training.Reconstruction` runs each stage in
``lightning``; ``torchio`` holds and augments the images.

**Learning objectives**

- Condition a :class:`bartorch.learning.UNet` on the iteration index and hand
  the index to it through :class:`bartorch.priors.ImplicitPrior`.
- Train an unrolled :class:`bartorch.optim.ISTBlock` in the three stages of
  :class:`bartorch.learning.training.Reconstruction`.
- Split a dataset by subject and augment it with transforms that respect
  complex values.

It follows :doc:`03-networks-for-complex-volumes`. The next lesson,
:doc:`05-self-supervised-training`, trains the same network without fully
sampled references.
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
import torchio
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

_ = torch.manual_seed(0)

# %%
#
# Images and acquisition
# ----------------------
#
# Axial slices of two BrainWeb subjects, converted into :math:`T_1`-weighted
# spin-echo images with a smooth phase as in :doc:`02-modl-with-admm`. The
# split is by subject: the slices of subject 4 are never trained on, so the
# validation measures what a network trained on one head does on another.
# Splitting slices or patches of the same heads at random would put
# neighbouring slices of one head on both sides.
#
# The encoding is eight-channel Cartesian SENSE with a fourfold
# variable-density undersampling of the phase encodes.

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
    SIZE / 4
)
lines[SIZE // 2 - 4 : SIZE // 2 + 4] = True
pattern = lines.to(torch.complex64)[:, None].expand(SIZE, SIZE).contiguous()
A = linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=pattern)
NOISE = 0.005

print(f"{len(train_images)} slices of subject 0 to train on, {len(valid_images)} of subject 4")
print(f"{int(lines.sum())} of {SIZE} phase encodes")

# %%
#
# Dataset
# -------
#
# Each ``torchio`` subject holds one reference image as two real channels.
# :class:`~bartorch.learning.training.RandomGain` multiplies it by a random
# complex number -- a global phase and a scale within 20 per cent -- and a flip
# and a small rotation vary the anatomy's orientation. The k-space is simulated
# from the augmented reference in the collate function, so the measured data
# stay consistent with it. ``torchio``'s other intensity transforms act on each
# channel alone and would not.
#
# A batch is a list of dictionaries, the form
# :class:`~bartorch.learning.training.Reconstruction` takes: the data, the
# operator, the reference, and the adjoint reconstruction the iteration starts
# from.

augmentation = torchio.Compose(
    [
        training.RandomGain(log_scale=0.2),
        torchio.RandomFlip(axes=(0,), flip_probability=0.5),
        torchio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, -8, 8), translation=0),
    ]
)


def subjects(images):
    return [
        torchio.Subject(image=torchio.ScalarImage(tensor=learning.as_real(x)[..., None]))
        for x in images
    ]


def measured(x, generator=None):
    """One item: simulated k-space of ``x``, the operator, the reference and ``A^H y``."""
    y = A(x) + NOISE * torch.randn(A.oshape, dtype=torch.complex64, generator=generator)
    return {"y": y, "A": A, "target": x, "x0": A.H(y)}


def collate(batch):
    return [measured(learning.as_complex(s["image"][torchio.DATA][..., 0])) for s in batch]


train_loader = DataLoader(
    torchio.SubjectsDataset(subjects(train_images), transform=augmentation),
    batch_size=4,
    shuffle=True,
    collate_fn=collate,
)
generator = torch.Generator().manual_seed(2)
valid_items = [measured(x, generator) for x in valid_images]
valid_loader = DataLoader(valid_items, batch_size=4, collate_fn=list)

# %%
#
# Network
# -------
#
# A two-dimensional :class:`~bartorch.learning.UNet` on the real and imaginary
# planes, conditioned on the iteration index (``steps=True``);
# :class:`~bartorch.learning.ComplexNet` lays the complex image out as those
# planes and scales it to unit peak around the call. The network is residual
# and starts as the identity, so the untrained stack is plain gradient
# descent. :class:`~bartorch.priors.ImplicitPrior` with ``step=True`` passes
# each iteration's index to it.

network = learning.UNet(2, spatial=2, widths=(16, 32, 64), steps=True)
prior = priors.ImplicitPrior(learning.ComplexNet(network, spatial=2), step=True)
block = optim.ISTBlock(prior, step=1.0)
block.step.requires_grad_()

weights = sum(p.numel() for p in network.parameters())
print(f"{weights} weights, {2 * weights / 1e6:.2f} MB stored in half precision")

# %%
#
# Stage 1: the denoiser alone
# ---------------------------
#
# The degraded inputs are reconstructions of increasing quality -- the adjoint,
# and conjugate-gradient SENSE after 2, 5 and 20 iterations -- paired with the
# reference and with the index at which the unrolled iteration is expected to
# meet an image of that quality. The pairs are made once; no unrolling and no
# encoding operator enters this stage, which is what makes it the cheap one.

STAGES = {0: 0, 2: 1, 5: 2, 20: 3}  # CG iterations -> iteration index

pairs = []
for x in train_images:
    item = measured(x)
    for iterations, index in STAGES.items():
        degraded = item["x0"] if 0 == iterations else optim.cg(item["y"], A, maxiter=iterations)
        pairs.append({"input": degraded, "target": x, "step": index})

trainer = lightning.Trainer(
    max_epochs=15,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)
stage = training.Reconstruction(learning.ComplexNet(network, spatial=2), "denoiser", lr=2e-3)
trainer.fit(
    stage,
    DataLoader(pairs, batch_size=8, shuffle=True, collate_fn=list),
    DataLoader(pairs[:16], batch_size=8, collate_fn=list),
)

# %%
#
# Stage 2: one iteration at a time
# --------------------------------
#
# ``detach=True`` starts every iteration from a detached state, and the greedy
# stage back-propagates each iteration's loss before the next iteration runs.
# The losses grow geometrically along the stack, the last ten times the first,
# since the last image is the one delivered. The images the network now sees
# are its own iterates, which pretraining could only approximate.

psnr = PSNRMetric(max_val=1.0)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0)


def quality(model):
    """Mean PSNR and SSIM of the magnitude over the validation slices."""
    with torch.no_grad():
        made = torch.stack([model(i["y"], A, i["x0"]) for i in valid_items]).abs()[:, None]
    truth = torch.stack(valid_images).abs()[:, None]
    return float(psnr(made, truth).mean()), float(ssim(made, truth).mean())


scores = {"untrained": None}
greedy = learning.Unrolled(block, iterations=ITERATIONS, detach=True)
scores["denoiser pretrained"] = quality(greedy)

trainer = lightning.Trainer(
    max_epochs=8,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)
trainer.fit(training.Reconstruction(greedy, "greedy", lr=1e-3), train_loader, valid_loader)
scores["greedy"] = quality(greedy)

# %%
#
# Stage 3: the whole stack
# ------------------------
#
# The loss is now on the last image alone, so the earlier iterations are free
# to produce whatever serves it best rather than their own best image.
# ``checkpoint=True`` keeps the states between iterations and recomputes the
# inside of each during the backward pass; the gradient is the end-to-end one
# at the memory of the states plus one iteration.

stack = learning.Unrolled(block, iterations=ITERATIONS, checkpoint=True)
trainer = lightning.Trainer(
    max_epochs=4,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)
trainer.fit(training.Reconstruction(stack, "end-to-end", lr=3e-4), train_loader, valid_loader)
scores["end to end"] = quality(stack)

# %%
#
# Results
# -------
#
# The untrained stack is four gradient steps from the adjoint, since the
# residual network starts as the identity. Conjugate-gradient SENSE at twenty
# iterations is the reference without a prior.

untrained = learning.Unrolled(
    optim.ISTBlock(priors.ImplicitPrior(lambda v: v), step=1.0), iterations=ITERATIONS
)
scores["untrained"] = quality(untrained)
cg = torch.stack([optim.cg(i["y"], A, maxiter=20) for i in valid_items])
truth = torch.stack(valid_images)
scores["CG SENSE, 20 iterations"] = (
    float(psnr(cg.abs()[:, None], truth.abs()[:, None]).mean()),
    float(ssim(cg.abs()[:, None], truth.abs()[:, None]).mean()),
)
for name, (p, s) in scores.items():
    print(f"{name:>24}   PSNR {p:5.2f} dB   SSIM {s:.3f}")
print(f"learned step: {float(block.step.detach()):.3f}")

# %%
#
# The pretrained denoiser can score below the untrained stack: inside the
# iteration it meets its own iterates rather than the conjugate-gradient images
# it was trained on, and the greedy stage is what closes that gap. Each stage
# starts from the weights the previous one left. On a single
# training subject and a few epochs per stage the numbers show the order of
# the stages rather than what any of them reaches on a real dataset, where the
# published schedule spent days on each.

# sphinx_gallery_start_ignore
with torch.no_grad():
    shown = stack(valid_items[3]["y"], A, valid_items[3]["x0"])
figure, axes = panels(1, 4)
top = float(truth[3].abs().max())
show(axes[0, 0], truth[3], "reference", vmax=top)
show(axes[0, 1], valid_items[3]["x0"], "adjoint", vmax=top)
show(axes[0, 2], cg[3], "CG SENSE", vmax=top)
show(axes[0, 3], shown, "unrolled, staged", vmax=top)
figure.suptitle("a validation slice of subject 4")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Storing the weights
# -------------------
#
# The weights are stored in half precision, which halves the file and loses
# nothing a float16 or bfloat16 inference would keep. ``load_state_dict``
# casts them back to the network's precision.

compact = {name: value.half() for name, value in stack.state_dict().items()}
restored = learning.Unrolled(
    optim.ISTBlock(
        priors.ImplicitPrior(
            learning.ComplexNet(
                learning.UNet(2, spatial=2, widths=(16, 32, 64), steps=True), spatial=2
            ),
            step=True,
        )
    ),
    iterations=ITERATIONS,
)
restored.load_state_dict(compact)
print(f"restored: PSNR {quality(restored)[0]:5.2f} dB")

# %%
#
# References
# ----------
#
# .. [#urman] Urman Y, Nishimura M, Abraham DR, Cao X, Setsompop K. Fully 3D unrolled
#    magnetic resonance fingerprinting reconstruction via staged pretraining and
#    implicit gridding. *Magn Reson Med* 96(5):2516-2529 (2026).
#    https://doi.org/10.1002/mrm.70500
#
# .. [#gleam] Wang K, Kellman M, Sandino CM, Zhang K, Vasanawala SS, Tamir JI,
#    Yu SX, Lustig M. Memory-efficient learning for high-dimensional MRI
#    reconstruction. *MICCAI* 2021, LNCS 12906:461-470.
#    https://doi.org/10.1007/978-3-030-87231-1_45
