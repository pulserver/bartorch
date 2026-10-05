r"""
======================================
Staged training of an unrolled network
======================================

**Aim.** Train an unrolled reconstruction network for fourfold undersampled,
eight-channel Cartesian brain data within the memory of one iteration, and
show that it removes the residual aliasing and the g-factor noise that
CG-SENSE leaves at this acceleration.

An unrolled network alternates data consistency with a learned regularizer
for a fixed number of iterations. Here the iteration is BART's iterative soft
thresholding with the thresholding replaced by a convolutional network
:math:`D_\theta`,

.. math::

   x^{k+1} = D_\theta\!\left(x^{k} - \tau\, A^H (A x^{k} - y),\; k\right),

where :math:`A` is the SENSE encoding (coil sensitivities, Fourier transform,
sampling mask) and :math:`\tau` a learned step size. One set of network
weights serves every iteration. The iteration index :math:`k` enters the
network by feature-wise modulation (FiLM), so that the same weights remove
the strong incoherent aliasing of the first iterates and the fine residual
noise of the last.

Training the stack end to end stores the activations of every iteration for
the backward pass; for a 3D volume, or a series of them, that exceeds a
single GPU. Urman et al. [#urman]_ reach the end-to-end result in three
stages whose memory is bounded by one iteration:

1. **Denoiser pretraining.** The network alone learns to map degraded images
   to the fully sampled reference, each image tagged with the iteration index
   at which the unrolled iteration will meet that level of degradation.
2. **Greedy training.** Each iteration's loss is back-propagated before the
   next iteration runs [#gleam]_.
3. **End-to-end fine-tuning.** One loss on the final image, through the whole
   stack, with each iteration recomputed during the backward pass
   (gradient checkpointing) instead of stored.

:class:`bartorch.learning.Reconstruction` runs each stage in
``lightning``; ``torchio`` holds and augments the training images.

**Prerequisites.** :doc:`../06-learning/02-modl-with-admm`.

**Learning objectives**

- Condition a :class:`bartorch.learning.UNet` on the iteration index and pass
  the index to it through :class:`bartorch.priors.ImplicitPrior`.
- Train an unrolled :class:`bartorch.optim.ISTBlock` in the three stages of
  :class:`bartorch.learning.Reconstruction`.
- Split a dataset by subject and augment it with transforms that preserve the
  complex MR signal.

The network is trained without fully sampled references in
:doc:`03-self-supervised-training`.
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
import torchio
from brainweb_dl import get_mri
from monai.metrics import PSNRMetric, SSIMMetric
from torch.utils.data import DataLoader

import bartorch
import bartorch.tools as bt
from bartorch import learning, linop, optim, priors

SIZE = 96
COILS = 8
ITERATIONS = 4
ACCELERATION = 4

_ = torch.manual_seed(0)

# %%
#
# Images and acquisition
# ----------------------
#
# Axial slices of two BrainWeb subjects, simulated as :math:`T_1`-weighted
# spin-echo images (TR 600 ms, TE 12 ms) with a smooth background phase, as in
# :doc:`../06-learning/02-modl-with-admm`. Training and validation are split by subject: no
# slice of subject 4 is trained on, so the validation scores measure how a
# network trained on one head generalizes to another. A random split of slices
# would place neighbouring, nearly identical slices of the same head on both
# sides and overstate the result.
#
# The acquisition is an eight-channel receive array with Cartesian
# variable-density undersampling of the phase encodes at :math:`R = 4`, with a
# fully sampled centre of eight lines. Complex Gaussian noise of standard
# deviation 0.02 (relative to an image peak of one) is added to k-space, an
# SNR at which the unfolding of CG-SENSE amplifies the noise visibly.

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

print(f"{len(train_images)} slices of subject 0 to train on, {len(valid_images)} of subject 4")
print(f"{int(lines.sum())} of {SIZE} phase encodes")

# %%
#
# Dataset
# -------
#
# Each ``torchio`` subject holds one reference image as two real channels.
# The augmentations are those that turn one MR image into another the scanner
# could have produced: :class:`~bartorch.learning.RandomGain` applies
# a random receiver gain and global phase (a complex scale within 20 per cent
# in magnitude), and a flip and a small in-plane rotation vary the head's
# orientation. k-space is simulated from the augmented reference in the
# collate function, so the measured data stay consistent with it. ``torchio``'s
# intensity transforms such as a gamma correction act on the real and
# imaginary channels independently and would break that consistency.
#
# A batch is a list of dictionaries, the form
# :class:`~bartorch.learning.Reconstruction` takes: the data, the
# operator, the reference, and the adjoint reconstruction the iteration starts
# from.

augmentation = torchio.Compose(
    [
        learning.RandomGain(log_scale=0.2),
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
# The inputs are reconstructions of increasing quality -- the zero-filled
# coil combination :math:`A^H y`, and CG-SENSE after 2, 5 and 20 iterations --
# each paired with the fully sampled reference and with the iteration index at
# which the unrolled network is expected to meet an image of that quality. The
# pairs are computed once; neither the unrolling nor the encoding enters this
# stage, which makes it the cheapest of the three.

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
stage = learning.Reconstruction(learning.ComplexNet(network, spatial=2), "denoiser", lr=2e-3)
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
# stage back-propagates each iteration's loss before the next iteration runs,
# so memory holds one iteration. The losses are weighted geometrically along
# the stack, the last ten times the first, since the last image is the one
# delivered. The network now sees its own iterates, which the CG-SENSE images
# of stage 1 only approximated.

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
trainer.fit(learning.Reconstruction(greedy, "greedy", lr=1e-3), train_loader, valid_loader)
scores["greedy"] = quality(greedy)

# %%
#
# Stage 3: the whole stack
# ------------------------
#
# The loss is now on the final image alone, so the earlier iterations are free
# to produce whatever intermediate image serves the last one best.
# ``checkpoint=True`` stores only the image between iterations and recomputes
# each iteration's activations during the backward pass: the gradient is the
# exact end-to-end one, at the memory of those images plus one iteration.

stack = learning.Unrolled(block, iterations=ITERATIONS, checkpoint=True)
trainer = lightning.Trainer(
    max_epochs=4,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)
trainer.fit(learning.Reconstruction(stack, "end-to-end", lr=3e-4), train_loader, valid_loader)
scores["end to end"] = quality(stack)

# %%
#
# Results
# -------
#
# The untrained stack is four gradient steps from :math:`A^H y`, since the
# residual network starts as the identity. CG-SENSE with twenty iterations is
# the baseline without a learned prior. Scores are the mean PSNR and SSIM of
# the magnitude over the eight validation slices of subject 4.

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
# Each stage starts from the weights the previous one left, and each raises
# the PSNR. The pretrained denoiser scores a high SSIM but a low PSNR: inside
# the iteration it meets its own iterates rather than the CG-SENSE images it
# was trained on, and the greedy stage adapts it to them. With one training subject and a few
# epochs per stage, the numbers show the ordering of the stages, not what each
# reaches on a real dataset.
#
# In the images below, CG-SENSE at :math:`R = 4` keeps a grainy, spatially
# varying noise -- the g-factor amplification of the coil unfolding -- and
# faint aliasing along the phase-encode direction (vertical). The unrolled
# network removes both; its error concentrates at tissue boundaries, where it
# slightly smooths the cortex.

# sphinx_gallery_start_ignore
with torch.no_grad():
    shown = stack(valid_items[3]["y"], A, valid_items[3]["x0"])
compare(
    truth[3],
    {"CG-SENSE": cg[3], "unrolled, staged": shown},
    crop=(slice(22, 58), slice(30, 66)),
)
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
