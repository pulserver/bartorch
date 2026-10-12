"""
=====================================
9. Nonlinear inversion for T2 mapping
=====================================

In lesson 8 you reconstructed one image from undersampled radial data, and in
lesson 4 you solved a model that is nonlinear in its unknowns, image times
coil sensitivities, by Gauss-Newton. In this lesson you apply both to
quantitative imaging. You estimate a :math:`T_2` map from an undersampled
multi-echo spin-echo scan twice: first by reconstructing an image per echo
and fitting the decay voxel by voxel, then by fitting the signal model
directly to the k-space data, and you compare the accuracy of the two at the
same undersampling.

In a multi-echo spin-echo (CPMG) scan the signal of each voxel decays from
echo to echo as :math:`M_0 \\exp(-\\mathrm{TE}/T_2)`. With each echo
undersampled, each echo image is a separate ill-posed reconstruction with its
own aliasing and noise, and a voxelwise fit cannot distinguish those from
decay.
Model-based reconstruction [#sumpf]_ [#wang]_ puts the signal model inside
the forward operator,

.. math::

   y_{c,e} = P_e F \\, (S_c \\cdot M_e(\\theta)),

where :math:`P_e` is the sampling pattern of echo :math:`e`, :math:`F` the
Fourier transform, :math:`S_c` the sensitivity of coil :math:`c`, :math:`M`
the signal model and :math:`\\theta` the parameter maps. You then solve for
:math:`\\theta` from the k-space of all echoes at once. The unknowns are three
real maps rather than eight complex images, and every echo constrains all of
them. The operator is nonlinear in :math:`\\theta`, so you solve it with the
iteratively regularized Gauss-Newton method (IRGNM) of lesson 4, over a
different model.

**Learning objectives**

- Write a :math:`T_2` decay as a signal model,
  :class:`bartorch.nlop.MultiEcho`, a BlochSim simulator wrapped as a BART
  nonlinear operator.
- Fit it to reconstructed echo images with :func:`bartorch.apps.mobafit`,
  and directly to k-space by composing it with the encoding and solving with
  :class:`bartorch.nlop.IRGNM`.
- Run the k-space fit in one call with :func:`bartorch.apps.moba`.
- Explain, from the echo images and the error maps, why the model-based fit
  is more accurate at the same undersampling.

Previous: :doc:`../04-non-cartesian/02-radial-sense`. Next:
:doc:`../06-learning/01-plug-and-play`, where you replace a penalty you wrote
down with a learned denoiser. The Tours
:doc:`../08-workflows/02-subspace-t1-mapping` and
:doc:`../08-workflows/03-maps-from-scanner-images` apply a linear subspace
model and the same signal model to DICOM images.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap

WIDTH = 7.8  # inches, the width of the documentation column

# Fuderer et al. (Magn Reson Med 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
NAVIA = Colormap("crameri:navia").to_matplotlib()

# The T2 window stops short of cerebrospinal fluid, so that white and grey
# matter -- 70 against 83 ms -- take up most of the scale and CSF saturates.
T2_LIMITS = (0.0, 120.0)
T2_LABEL = "$T_2$ [ms]"


def panels(columns, rows=1, width=WIDTH):
    """A row (or grid) of frameless square image panels."""
    side = width / columns
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(width, rows * side + 0.5))
    for axis in axes.flat:
        axis.set_axis_off()
    return figure, axes


def show(axis, values, title=None, vmax=None, cmap="gray", vmin=0.0):
    """One panel, of a magnitude by default."""
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


def t2_map(axis, values, title=None):
    """One T2 map in its colormap and window."""
    return show(axis, values, title, vmin=T2_LIMITS[0], vmax=T2_LIMITS[1], cmap=NAVIA)


def t2_bar(figure, axes):
    """One T2 colorbar for a group of panels."""
    handle = plt.cm.ScalarMappable(plt.Normalize(*T2_LIMITS), NAVIA)
    return figure.colorbar(handle, ax=axes, fraction=0.046, label=T2_LABEL)


def scaled(estimate, reference):
    """``estimate`` scaled to ``reference`` in the least-squares sense."""
    a, b = estimate.abs().double(), reference.abs().double()
    return (float((a * b).sum() / (a * a).sum()) * a).float()


# The phantom: the BrainWeb segmentation of lesson 2, with the proton density
# and T2 of each tissue class combined by membership.
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

import torch

import bartorch
import bartorch.tools as bt

SIZE = 64
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times
MARGIN = 0.25  # what the field of view leaves around the head

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
fractions = np.flipud(get_mri(sub_id=0, contrast="fuzzy")[SLICE])[..., list(TISSUES)].copy()

occupied = np.nonzero(fractions.sum(-1) > 0.5)
middle = [int((axis.min() + axis.max()) / 2) for axis in occupied]
half = int(round((1 + MARGIN) * max(axis.max() - axis.min() for axis in occupied) / 2))
source = tuple(
    slice(max(0, c - half), min(n, c + half)) for c, n in zip(middle, fractions.shape[:2])
)
box = np.zeros((2 * half, 2 * half, fractions.shape[-1]), dtype=np.float32)
box[tuple(slice(s.start - (c - half), s.stop - (c - half)) for s, c in zip(source, middle))] = (
    fractions[source]
)
memberships = torch.nn.functional.interpolate(
    torch.as_tensor(box).permute(2, 0, 1)[None],
    size=(SIZE, SIZE),
    mode="bilinear",
    align_corners=False,
)[0]
weights = memberships * torch.as_tensor(tissue_pd)[:, None, None]
share = weights.sum(0).clamp(min=1e-6)
T2 = (weights * torch.as_tensor(tissue_t2)[:, None, None]).sum(0) / share
proton_density = weights.sum(0) / weights.sum(0).max()

support = memberships.sum(0) > 0.5
amplitude = torch.where(support, proton_density, torch.zeros(()))
t2 = torch.where(support, T2.clamp(20.0, 400.0), torch.tensor(50.0))

# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=8)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

# %%
# Multi-echo phantom
# ------------------
#
# The object is the BrainWeb slice of lesson 2 on a 64 matrix, described by
# two maps: a proton-density amplitude and a :math:`T_2` per voxel, each
# combined from the tissue classes in the voxel by membership. Eight spin
# echoes are acquired 12.5 ms apart. The echo images follow the
# mono-exponential decay, written out explicitly here rather than taken from
# the model that is fitted below.
import time

import torch

import bartorch.tools as bt

from bartorch import apps, linop, nlop, optim

ECHOES = 8
ECHO_TIMES = torch.tensor([12.5 * (echo + 1) for echo in range(ECHOES)])  # ms

decay = torch.exp(-ECHO_TIMES[:, None, None] / t2[None])
contrasts = (amplitude[None] * decay).to(torch.complex64)
print(f"echo images {tuple(contrasts.shape)}: (echo, y, x)")

# sphinx_gallery_start_ignore
figure, axes = panels(2, width=0.7 * WIDTH)
show(axes[0, 0], amplitude, "amplitude", vmax=1.0)
t2_map(axes[0, 1], torch.where(support, t2, torch.zeros(())), "$T_2$")
t2_bar(figure, axes[0, 1])
plt.show()
# sphinx_gallery_end_ignore

# %%
# These are the maps to be estimated. In the BrainWeb table white matter
# has a :math:`T_2` of 70 ms and grey matter 83 ms; the cerebrospinal fluid in
# the ventricles, at 329 ms, saturates the window, which is chosen to spread
# white and grey matter over the colour scale [#fuderer]_.
#
# Echo undersampling
# ------------------
#
# Each echo is sampled at a quarter of the phase encodes (:math:`R = 4`):
# eight fully sampled central lines and the rest drawn at random, with a
# different draw per echo, so that the missing lines differ between echoes.
# The echoes are a batch of the encoding rather than an axis inside it: they
# share the coils and the transform and differ only in the pattern. So the
# encoding :math:`E` is the Cartesian SENSE operator of lesson 6 followed by
# a diagonal that applies each echo's pattern to its samples.
ACCELERATION = 4

# sphinx_gallery_start_ignore
generator = torch.Generator().manual_seed(4)
lines = torch.zeros(ECHOES, 1, SIZE, 1)
for echo in range(ECHOES):
    lines[echo, 0, torch.randperm(SIZE, generator=generator)[: SIZE // ACCELERATION]] = 1.0
    lines[echo, 0, SIZE // 2 - 4 : SIZE // 2 + 4] = 1.0
# sphinx_gallery_end_ignore

encoding = linop.CartesianSense(
    sensitivities,
    (ECHOES, SIZE, SIZE),
    ndim=2,
)
pattern = linop.Diagonal(lines.to(torch.complex64), encoding.oshape)
E = pattern @ encoding

measured = bt.noise(E(contrasts), n=1e-6, s=9)
data = measured / float(E.H(measured).abs().max())

print(f"{E.ishape} -> {E.oshape}")

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.85 * WIDTH, 2.6))
axis.imshow(
    lines[:, 0, :, 0].numpy(),
    cmap="gray",
    aspect="auto",
    vmin=0.0,
    vmax=1.0,
    extent=(-SIZE / 2, SIZE / 2, ECHOES + 0.5, 0.5),
)
axis.set_xlabel("phase encode")
axis.set_ylabel("echo")
axis.set_yticks(range(1, ECHOES + 1))
axis.set_title("sampled phase encodes (white)")
for spine in axis.spines.values():
    spine.set_visible(False)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each row is one echo, each white column an acquired phase encode. The
# central block is common to all echoes; outside it, each echo has its own
# lines, so the aliasing differs from echo to echo.
#
# The signal model
# ----------------
#
# :class:`bartorch.nlop.MultiEcho` is the decay as a nonlinear operator: it
# maps parameter maps to one image per echo. Its unknowns are :math:`T_2` and
# a complex amplitude, carried as three real maps in a bounded
# parameterisation rather than in their own units. So you build a starting
# point from values in physical units with
# :meth:`~bartorch.nlop.SignalModel.initial`, and convert a fit back to
# milliseconds with :meth:`~bartorch.nlop.SignalModel.split`.
M = nlop.MultiEcho([float(te) for te in ECHO_TIMES], (SIZE, SIZE))
start = M.initial(T2=80.0)

print(f"unknowns {M.names}: {M.ishapes[0]} -> {M.oshapes[0]}")

# %%
# Two-step fit
# ------------
#
# The conventional route has two steps. Reconstruct every echo by conjugate
# gradients on the SENSE normal equations, then fit the model to the echo
# images voxel by voxel with :func:`bartorch.apps.mobafit`, a Gauss-Newton
# loop over the model alone.
STEPS = 20

start_time = time.perf_counter()
images = optim.CG(maxiter=40)(data, E)
two_step = apps.mobafit(images, M, iterations=STEPS, T2=80.0)
print(f"reconstruct, then fit: {time.perf_counter() - start_time:5.1f} s")

# sphinx_gallery_start_ignore
shown = (0, 3, 7)
figure, axes = panels(len(shown), 2)
top = float(contrasts.abs().max())
for column, echo in enumerate(shown):
    show(axes[0, column], contrasts[echo], f"TE = {float(ECHO_TIMES[echo]):.1f} ms", vmax=top)
    show(axes[1, column], scaled(images[echo], contrasts[echo]), vmax=top)
for row, label in enumerate(("reference", "SENSE, R = 4")):
    axes[row, 0].text(
        -0.04, 0.5, label, rotation=90, va="center", ha="right", transform=axes[row, 0].transAxes
    )
plt.show()
# sphinx_gallery_end_ignore

# %%
# These images are the input of the voxelwise fit. Each reconstructed echo
# carries residual aliasing and noise, different in every echo because every
# echo has its own pattern. At the later echoes the signal has decayed and
# the relative error grows, so the late echoes, which determine :math:`T_2`
# most, are also the least accurate.
#
# Model-based forward operator
# ----------------------------
#
# The model-based route composes the model with the encoding, ``E @ M``, and
# fits the k-space directly with :class:`bartorch.nlop.IRGNM`, the
# Gauss-Newton solver of lesson 4. The derivative of the composition is the
# encoding applied to the derivative of the model, which is the derivative a
# Gauss-Newton step requires.
start_time = time.perf_counter()
solver = nlop.IRGNM(iterations=STEPS, cg_maxiter=100, cg_tol=0.1)
model_based = solver(data, E @ M, x0=start)
print(f"model inside the operator: {time.perf_counter() - start_time:5.1f} s")

# %%
# Both routes are Gauss-Newton loops of the same number of steps. They differ
# only in the forward operator that maps the unknowns to the data: the model
# alone, against echo images, or the model followed by the encoding, against
# k-space.
#
# Model-based reconstruction app
# ------------------------------
#
# :func:`bartorch.apps.moba` assembles the model-based fit from the
# k-space, the model, the sensitivities and the sampling pattern, and returns
# the maps in their own units. It scales the data by the rule of
# :func:`bartorch.optim.data_scaling` and regularizes each Gauss-Newton step
# towards the starting maps rather than towards zero, so its result is not
# identical to the fit above.
start_time = time.perf_counter()
one_call = apps.moba(
    measured,
    M,
    sensitivities,
    pattern=lines,
    iterations=STEPS,
    T2=80.0,
)
print(f"apps.moba: {time.perf_counter() - start_time:5.1f} s")

# %%
# Map comparison
# --------------
#
# Read the three :math:`T_2` maps back and compare them with the phantom,
# inside the head:
estimates = {
    "two-step": two_step["T2"],
    "model-based": M.split(model_based)["T2"],
    "apps.moba": one_call["T2"],
}

truth = t2[support]
for name, estimate in estimates.items():
    error = float((estimate[support] - truth).norm() / truth.norm())
    median = float(estimate[support].median())
    print(f"{name:>12}: median {median:5.1f} ms, relative error {error:.3f}")
print(f"{'phantom':>12}: median {float(truth.median()):5.1f} ms")

# sphinx_gallery_start_ignore
figure, axes = panels(2, 2, width=0.8 * WIDTH)
t2_map(axes[0, 0], torch.where(support, t2, torch.zeros(())), "reference")
for axis, (name, values) in zip(axes.flat[1:], estimates.items()):
    t2_map(axis, torch.where(support, values, torch.zeros(())).detach(), name)
t2_bar(figure, axes)
plt.show()

figure, axes = panels(3)
for axis, (name, values) in zip(axes[0], estimates.items()):
    difference = torch.where(support, (values.detach() - t2).abs(), torch.zeros(()))
    handle = show(axis, difference.numpy(), name, vmax=40.0, cmap="magma")
figure.colorbar(handle, ax=axes[0], fraction=0.046, label="$|\\Delta T_2|$ [ms]")
figure.suptitle("$T_2$ error")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The two-step map is dominated by the errors of the echo images: a voxelwise
# fit cannot distinguish residual aliasing from decay, and where a late echo
# is too bright or too dark the fitted :math:`T_2` is far off. Its median is
# close to the phantom's, but its relative error is the largest of the three.
# The model-based fits reach a much lower error from the same data, because
# the model admits only echo images that decay exponentially, and the
# aliasing of eight different sampling patterns is not such an image.
#
# The fit inside the operator reproduces the phantom almost exactly, because
# every voxel of the phantom decays with a single :math:`T_2`, as the model
# assumes. A measured voxel holding two tissues decays with two, and a single
# exponential cannot represent it. :func:`bartorch.apps.moba` regularizes
# each Gauss-Newton step towards the starting maps, and its residual error is
# largest where :math:`T_2` is farthest from the starting value of 80 ms: in
# the cerebrospinal fluid of the ventricles and at the scalp.
#
# Because the maps are the solver's unknowns, a regularizer passed to the
# linearized problem, the ``inner`` solver of :func:`bartorch.apps.moba`,
# penalizes the maps rather than the echo images. Without ``sensitivities``,
# :func:`bartorch.apps.moba` estimates the coils jointly with the maps, as
# lesson 4 estimates them jointly with an image.
# :doc:`../../explanation/nonlinear` describes the model-based approach in
# more detail.
#
# Exercise: fully sampled echoes
# ------------------------------
#
# Is the error of the two-step route caused by the voxelwise fit or by the
# echo images it is given? Run the same fit on the fully sampled, noise-free
# echo images of the phantom:
fully_sampled = apps.mobafit(contrasts, M, iterations=STEPS, T2=80.0)

estimate = fully_sampled["T2"][support]
error = float((estimate - truth).norm() / truth.norm())
print(f"voxelwise fit, fully sampled: relative error {error:.3f}")

# %%
# On the fully sampled echoes the voxelwise fit reaches a relative error far
# below the two-step result on the undersampled data, of the same order as the
# model-based fit. The fit itself is therefore not the limitation; the
# undersampling artefacts of the echo images are. Try fitting the
# undersampled echo images with more Gauss-Newton steps, or with fewer
# echoes, for example ``images[:4]`` with a model built on the first four
# echo times, and compare the errors.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Estimate a T2 map from an undersampled 2D multi-echo spin-echo scan with
#    bartorch: 8 echoes at TE = 12.5, 25, ..., 100 ms, 8 coils, 64x64 matrix,
#    R = 4 per echo with 8 central lines and a different random pattern per
#    echo. Encode with linop.CartesianSense over (echo, y, x) followed by a
#    linop.Diagonal of the patterns. Model the decay with
#    nlop.MultiEcho(TE, (64, 64)), starting at T2 = 80 ms. Compare (a) CG
#    (40 iterations) per echo followed by apps.mobafit, (b) nlop.IRGNM on
#    E @ M (20 steps, inner CG 100 iterations, tol 0.1), and (c) apps.moba
#    with the sensitivities and patterns (20 steps). Report the median and the
#    relative error of T2 inside the head, and show the maps and |error| maps.

# %%
#
# References
# ----------
#
# .. [#sumpf] Sumpf TJ, Uecker M, Boretius S, Frahm J. Model-based nonlinear inverse
#    reconstruction for T2 mapping using highly undersampled spin-echo MRI.
#    *J Magn Reson Imaging* 34(2):420-428 (2011).
#    https://doi.org/10.1002/jmri.22634
#
# .. [#wang] Wang X, Tan Z, Scholand N, Roeloffs V, Uecker M. Physics-based
#    reconstruction methods for magnetic resonance imaging. *Phil Trans R Soc A*
#    379(2200):20200196 (2021). https://doi.org/10.1098/rsta.2020.0196
#
# .. [#fuderer] Fuderer M, Wichtmann B, Crameri F, de Souza NM, Baeßler B, Gulani V,
#    et al. Color-map recommendation for MR relaxometry maps. *Magn Reson Med*
#    93(2):490-506 (2025). https://doi.org/10.1002/mrm.30290
