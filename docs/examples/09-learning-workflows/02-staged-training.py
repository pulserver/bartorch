r"""
======================================
Staged training of an unrolled network
======================================

In lesson 11 you trained an unrolled network end to end, which keeps the
activations of every iteration in memory for the backward pass. For a 2D
slice that memory is affordable; for a 3D volume, or a series of them, it
exceeds a single GPU. In this tour you train an unrolled network for
fourfold undersampled, eight-channel Cartesian brain data in three stages
whose memory is bounded by one iteration, and you check that it removes the
residual aliasing and the g-factor noise that CG-SENSE leaves at this
acceleration.

The iteration is BART's iterative soft thresholding with the thresholding
replaced by a convolutional network :math:`D_\theta`,

.. math::

   x^{k+1} = D_\theta\!\left(x^{k} - \tau\, A^H (A x^{k} - y),\; k\right),

where :math:`A` is the SENSE encoding (coil sensitivities, Fourier transform,
sampling pattern) and :math:`\tau` a learned step size. One set of weights
serves every iteration. The iteration index :math:`k` enters the network by
feature-wise modulation (FiLM), so the same weights can remove the strong
incoherent aliasing of the first iterates and the fine residual noise of the
last.

Urman et al. [#urman]_ reach the end-to-end result in three stages:

1. **Denoiser pretraining.** The network alone learns to map degraded images
   to the fully sampled reference, each image tagged with the iteration index
   at which the unrolled iteration will meet that level of degradation.
2. **Greedy training.** Each iteration's loss is back-propagated before the
   next iteration runs [#gleam]_.
3. **End-to-end fine-tuning.** One loss on the final image, through the whole
   stack, with each iteration recomputed during the backward pass (gradient
   checkpointing) instead of stored.

:class:`bartorch.learning.Reconstruction` runs each stage in ``lightning``,
and ``torchio`` holds and augments the training images.

Prerequisites: lesson 11, :doc:`../06-learning/02-modl-with-admm`.

**Learning objectives**

- Condition a :class:`bartorch.learning.UNet` on the iteration index and pass
  the index to it through :class:`bartorch.priors.ImplicitPrior`.
- Train an unrolled :class:`bartorch.optim.ISTBlock` in the three stages of
  :class:`bartorch.learning.Reconstruction`.
- Split a dataset by subject and augment it with transforms that preserve the
  complex MR signal.

:doc:`03-self-supervised-training` trains the same network without fully
sampled references.
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
# Images and acquisition
# ----------------------
#
# Your images are axial slices of two BrainWeb subjects [#brainweb]_,
# simulated as :math:`T_1`-weighted spin-echo images (TR 600 ms, TE 12 ms)
# with a smooth background phase, as in lesson 11. You split training and
# validation by subject: no slice of subject 4 is trained on, so the
# validation scores measure how a network trained on one head generalizes to
# another. A random split of slices would put neighbouring, nearly identical
# slices of the same head on both sides and overstate the result.
import lightning
import torch
import torchio

import bartorch
import bartorch.tools as bt

from bartorch import learning, linop, optim, priors
from monai.metrics import PSNRMetric, SSIMMetric
from torch.utils.data import DataLoader

COILS = 8
ITERATIONS = 4
ACCELERATION = 4
NOISE = 0.02

_ = torch.manual_seed(0)

train_images = brain_slices(subject=0, count=24)
valid_images = brain_slices(subject=4, count=8)

# %%
# The acquisition is an eight-channel receive array with variable-density
# Cartesian undersampling of the phase encodes at :math:`R = 4`, around a
# fully sampled centre of eight lines. Complex Gaussian noise of standard
# deviation 0.02 (relative to an image peak of one) goes into k-space, an SNR
# at which the unfolding of CG-SENSE amplifies the noise visibly.
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

print(f"{len(train_images)} slices of subject 0, {len(valid_images)} of 4")
print(f"{int(lines.sum())} of {SIZE} phase encodes")

# %%
# Dataset
# -------
#
# Each ``torchio`` subject holds one reference image as two real channels.
# Pick augmentations that turn one MR image into another the scanner could
# have produced: :class:`~bartorch.learning.RandomGain` applies a random
# receiver gain and global phase (a complex scale within 20 per cent in
# magnitude), and a flip and a small in-plane rotation vary the head's
# orientation. You simulate k-space from the augmented reference in the
# collate function, so the measured data stay consistent with it.
# ``torchio``'s intensity transforms, such as a gamma correction, act on the
# real and imaginary channels independently and would break that consistency.
#
# A batch is a list of dictionaries, the form
# :class:`~bartorch.learning.Reconstruction` takes: the data, the operator,
# the reference, and the adjoint reconstruction the iteration starts from.
augmentation = torchio.Compose(
    [
        learning.RandomGain(log_scale=0.2),
        torchio.RandomFlip(axes=(0,), flip_probability=0.5),
        torchio.RandomAffine(
            scales=0,
            degrees=(0, 0, 0, 0, -8, 8),
            translation=0,
        ),
    ]
)


def subjects(images):
    made = []
    for x in images:
        image = torchio.ScalarImage(tensor=learning.as_real(x)[..., None])
        made.append(torchio.Subject(image=image))
    return made


def measured(x, generator=None):
    """One item: k-space of ``x``, the operator, the reference, ``A^H y``."""
    noise = torch.randn(A.oshape, dtype=torch.complex64, generator=generator)
    y = A(x) + NOISE * noise
    return {"y": y, "A": A, "target": x, "x0": A.H(y)}


def collate(batch):
    images = [s["image"][torchio.DATA][..., 0] for s in batch]
    return [measured(learning.as_complex(x)) for x in images]


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
# Network
# -------
#
# The network is a 2D :class:`~bartorch.learning.UNet` on the real and
# imaginary planes, conditioned on the iteration index (``steps=True``).
# :class:`~bartorch.learning.ComplexNet` lays the complex image out as those
# planes and scales it to unit peak around the call. The U-Net is residual and
# starts as the identity, so the untrained stack is plain gradient descent.
# :class:`~bartorch.priors.ImplicitPrior` with ``step=True`` passes each
# iteration's index to it.
network = learning.UNet(2, spatial=2, widths=(16, 32, 64), steps=True)
prior = priors.ImplicitPrior(learning.ComplexNet(network, spatial=2), step=True)
block = optim.ISTBlock(prior, step=1.0)
block.step.requires_grad_()

weights = sum(p.numel() for p in network.parameters())
print(f"{weights} weights, {2 * weights / 1e6:.2f} MB in half precision")

# %%
# The three stages differ only in the model and the data you hand
# ``lightning``; the trainer is the same each time:


def trainer(epochs):
    return lightning.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


# %%
# Stage 1: the denoiser alone
# ---------------------------
#
# The inputs are reconstructions of increasing quality -- the zero-filled coil
# combination :math:`A^H y`, and CG-SENSE after 2, 5 and 20 iterations --
# each paired with the fully sampled reference and with the iteration index at
# which the unrolled network is expected to meet an image of that quality. You
# compute the pairs once; neither the unrolling nor the encoding enters this
# stage, which makes it the cheapest of the three.
STAGES = {0: 0, 2: 1, 5: 2, 20: 3}  # CG iterations -> iteration index

pairs = []
for x in train_images:
    item = measured(x)
    for iterations, index in STAGES.items():
        if iterations == 0:
            degraded = item["x0"]
        else:
            degraded = optim.cg(item["y"], A, maxiter=iterations)
        pairs.append({"input": degraded, "target": x, "step": index})

denoiser = learning.ComplexNet(network, spatial=2)
stage = learning.Reconstruction(denoiser, "denoiser", lr=2e-3)
trainer(15).fit(
    stage,
    DataLoader(pairs, batch_size=8, shuffle=True, collate_fn=list),
    DataLoader(pairs[:16], batch_size=8, collate_fn=list),
)

# %%
# Stage 2: one iteration at a time
# --------------------------------
#
# ``detach=True`` starts every iteration from a detached state, and the greedy
# stage back-propagates each iteration's loss before the next iteration runs,
# so memory holds one iteration. The losses are weighted geometrically along
# the stack, the last ten times the first, since the last image is the one you
# deliver. The network now sees its own iterates, which the CG-SENSE images of
# stage 1 only approximated.
#
# To follow the stages you score the magnitude of each reconstruction of the
# validation slices by its mean PSNR and SSIM:
psnr = PSNRMetric(max_val=1.0)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0)
truth = torch.stack(valid_images)


def quality(images):
    made, reference = images.abs()[:, None], truth.abs()[:, None]
    return (
        float(psnr(made, reference).mean()),
        float(ssim(made, reference).mean()),
    )


def reconstruct(model):
    with torch.no_grad():
        return torch.stack([model(i["y"], A, i["x0"]) for i in valid_items])


greedy = learning.Unrolled(block, iterations=ITERATIONS, detach=True)
scores = {"untrained": None}
scores["denoiser pretrained"] = quality(reconstruct(greedy))

trainer(8).fit(
    learning.Reconstruction(greedy, "greedy", lr=1e-3),
    train_loader,
    valid_loader,
)
scores["greedy"] = quality(reconstruct(greedy))

# %%
# Stage 3: the whole stack
# ------------------------
#
# The loss is now on the final image alone, so the earlier iterations are free
# to produce whatever intermediate image serves the last one best.
# ``checkpoint=True`` stores only the image between iterations and recomputes
# each iteration's activations during the backward pass: the gradient is the
# exact end-to-end one, at the memory of those images plus one iteration.
stack = learning.Unrolled(block, iterations=ITERATIONS, checkpoint=True)
trainer(4).fit(
    learning.Reconstruction(stack, "end-to-end", lr=3e-4),
    train_loader,
    valid_loader,
)
scores["end to end"] = quality(reconstruct(stack))

# %%
# Results
# -------
#
# Two baselines frame the scores. The untrained stack is four gradient steps
# from :math:`A^H y`, since the residual network starts as the identity.
# CG-SENSE with twenty iterations is the reconstruction without a learned
# prior.
identity = priors.ImplicitPrior(lambda v: v)
gradient_steps = optim.ISTBlock(identity, step=1.0)
untrained = learning.Unrolled(gradient_steps, iterations=ITERATIONS)
scores["untrained"] = quality(reconstruct(untrained))
cg = torch.stack([optim.cg(i["y"], A, maxiter=20) for i in valid_items])
scores["CG SENSE, 20 iterations"] = quality(cg)

# sphinx_gallery_start_ignore
for name, (p, s) in scores.items():
    print(f"{name:>24}   PSNR {p:5.2f} dB   SSIM {s:.3f}")
print(f"learned step: {float(block.step.detach()):.3f}")
# sphinx_gallery_end_ignore

# %%
# Each stage starts from the weights the previous one left, and each raises
# the PSNR over the one before, from about 26.5 dB for the untrained stack to
# about 29.8 dB end to end, against about 24.3 dB for CG-SENSE. The pretrained
# denoiser has the highest SSIM and a low PSNR: inside the iteration it meets
# its own iterates rather than the CG-SENSE images it was trained on, and the
# greedy stage adapts it to them. With one training subject and a few epochs
# per stage, the numbers show the ordering of the stages, not what each
# reaches on a real dataset.

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
# CG-SENSE at :math:`R = 4` keeps a grainy, spatially varying noise -- the
# g-factor amplification of the coil unfolding -- and faint aliasing along the
# phase-encode direction (vertical). The unrolled network removes both; its
# error is concentrated at tissue boundaries, where it slightly smooths the
# cortex.
#
# Storing the weights
# -------------------
#
# Store the weights in half precision: that halves the file and loses nothing
# a float16 or bfloat16 inference would keep. ``load_state_dict`` casts them
# back to the network's precision.
compact = {name: value.half() for name, value in stack.state_dict().items()}

fresh = learning.UNet(2, spatial=2, widths=(16, 32, 64), steps=True)
fresh_denoiser = learning.ComplexNet(fresh, spatial=2)
fresh_prior = priors.ImplicitPrior(fresh_denoiser, step=True)
restored = learning.Unrolled(optim.ISTBlock(fresh_prior), iterations=ITERATIONS)
restored.load_state_dict(compact)
print(f"restored: PSNR {quality(reconstruct(restored))[0]:5.2f} dB")

# %%
# Your turn: what each iteration adds
# -----------------------------------
#
# :meth:`~bartorch.learning.Unrolled.steps` yields the image after every
# iteration, so you can watch the trained stack and the untrained one
# converge. No training is involved:
curves = {}
for name, model in (("staged", stack), ("untrained", untrained)):
    with torch.no_grad():
        runs = [list(model.steps(i["y"], A, i["x0"])) for i in valid_items]
    iterates = [torch.stack(images) for images in zip(*runs)]
    curves[name] = [quality(images)[0] for images in iterates]

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * PAGE_WIDTH, 0.42 * PAGE_WIDTH))
for name, curve in curves.items():
    axis.plot(range(1, ITERATIONS + 1), curve, "o-", lw=1.8, label=name)
axis.axhline(scores["CG SENSE, 20 iterations"][0], color="C2", ls="--", label="CG-SENSE")
axis.set_xticks(range(1, ITERATIONS + 1))
axis.set_xlabel("iteration")
axis.set_ylabel("PSNR (dB)")
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The staged stack improves the PSNR at every iteration, and its first
# iterate already scores above the fourth iterate of the untrained stack. The
# untrained stack is gradient descent on the data term, and its four steps
# also score above twenty iterations of CG-SENSE.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Train an unrolled ISTA network for 2D Cartesian SENSE (8 coils, R = 4,
#    8 central lines, k-space noise 0.02) with bartorch: optim.ISTBlock with a
#    learned step over priors.ImplicitPrior(ComplexNet(UNet(2, spatial=2,
#    widths=(16, 32, 64), steps=True)), step=True), 4 iterations. Train in
#    three lightning stages with learning.Reconstruction: "denoiser" on A^H y
#    and CG-SENSE images after 2, 5, 20 iterations tagged with steps 0-3
#    (15 epochs); "greedy" with Unrolled(detach=True) (8 epochs); "end-to-end"
#    with Unrolled(checkpoint=True) (4 epochs). Train on subject 0 with
#    torchio RandomGain, flip and rotation, validate on subject 4, and report
#    PSNR and SSIM after each stage against CG-SENSE.

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
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
