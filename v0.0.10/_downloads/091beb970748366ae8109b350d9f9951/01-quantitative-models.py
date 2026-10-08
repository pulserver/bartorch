"""
====================================
Parameter maps straight from k-space
====================================

This lesson estimates a :math:`T_2` map from an undersampled multi-echo
spin-echo acquisition in two ways, and compares them: reconstructing an image
per echo and fitting the decay voxel by voxel afterwards, and fitting the
signal model directly to the k-space data. The aim is to show why the second,
model-based reconstruction, tolerates undersampling that ruins the first.

In a multi-echo spin-echo (CPMG) acquisition the signal of each voxel decays
from echo to echo as :math:`M_0 \\exp(-\\mathrm{TE}/T_2)`. Undersampling each
echo shortens the scan, but a reconstruction of each echo on its own is an
ill-posed problem, and its aliasing and noise differ from echo to echo; a
voxelwise fit cannot tell them apart from decay, and carries them into the
map. The model-based approach [#sumpf]_ [#wang]_ puts the signal model inside
the forward operator,

.. math::

   y_{c,e} = P_e F \\, (S_c \\cdot M_e(\\theta)),

where :math:`P_e` is the sampling pattern of echo :math:`e`, :math:`F` the
Fourier transform, :math:`S_c` the sensitivity of coil :math:`c`,
:math:`M` the signal model and :math:`\\theta` the parameter maps, and solves
for :math:`\\theta` from the k-space data of all echoes at once. The unknowns
are then three real maps rather than eight complex images, and every echo
constrains all of them. The operator is nonlinear in :math:`\\theta`, so the
problem is solved by the iteratively regularized Gauss-Newton method of
:doc:`../02-parallel-imaging/02-nonlinear-inversion`, over a different model.

The model here is :class:`bartorch.nlop.MultiEcho`, a BlochSim simulator as a
BART nonlinear operator. The phantom and the coil sensitivities are built as
in :doc:`../01-basics/02-from-kspace-to-image`; the cell that does it is
hidden on this page and present in the script this page can be downloaded as.

**Learning objectives**

- Represent a relaxation model as a BlochSim-backed
  :class:`bartorch.nlop.SignalModel`.
- Fit it to reconstructed echo images, and directly to k-space by composing
  it with the encoding, with :class:`bartorch.nlop.IRGNM`.
- Run the same fits through :func:`bartorch.apps.mobafit` and
  :func:`bartorch.apps.moba`.
- Explain, from the echo images and the error maps, why the model-based fit
  is more accurate at the same undersampling.

It follows :doc:`../04-non-cartesian/02-radial-sense`; the Gauss-Newton solver
is that of :doc:`../02-parallel-imaging/02-nonlinear-inversion`. The next
lesson, :doc:`../06-learning/01-plug-and-play`, replaces a specified
regularizer with a learned denoiser. The Tours
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
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()

# Colormap, window and unit per parameter.  Both relaxation windows stop short
# of cerebrospinal fluid, so that white and grey matter -- 500 against 833 ms
# in T1, 70 against 83 ms in T2 -- take up most of the scale and CSF saturates.
STYLE = {
    "T1": (LIPARI, (0.0, 1200.0), "$T_1$ [ms]"),
    "T2": (NAVIA, (0.0, 120.0), "$T_2$ [ms]"),
}


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


def parameter(axis, values, name, title=None):
    """One relaxation map in the colormap and window its parameter is read in."""
    cmap, limits, _ = STYLE[name]
    return show(axis, values, title, vmax=limits[1], cmap=cmap, vmin=limits[0])


def scalebar(figure, axes, handle=None, label=None, name=None):
    """One colorbar for a group of panels, so none gives up width to its own."""
    if name is not None:
        cmap, limits, label = STYLE[name]
        handle = plt.cm.ScalarMappable(plt.Normalize(*limits), cmap)
    return figure.colorbar(handle, ax=axes, fraction=0.046, label=label)


def errors(figure, axes, estimates, reference, scale):
    """|estimate - reference| relative to the reference's peak, on one scale."""
    peak = float(reference.abs().max())
    for axis, estimate in zip(axes, estimates):
        difference = (scaled(estimate, reference) - reference.abs()).abs() / peak
        handle = show(axis, difference, cmap="magma", vmax=scale)
    return figure.colorbar(handle, ax=axes, fraction=0.046, label="|error| / peak")


def scaled(estimate, reference):
    """``estimate`` scaled to ``reference`` in the least-squares sense."""
    a, b = estimate.abs().double(), reference.abs().double()
    return (float((a * b).sum() / (a * a).sum()) * a).float()


# sphinx_gallery_end_ignore
import csv
import time
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import apps, linop, nlop, optim

SIZE = 64
COILS = 8
ECHOES = 8
ACCELERATION = 4

ECHO_TIMES = torch.tensor([12.5 * (echo + 1) for echo in range(ECHOES)])  # ms

# %%
#
# Phantom
# -------
#
# The :math:`T_2` of each tissue class from the BrainWeb table, combined by
# membership, and the echo images from the mono-exponential decay
# :math:`M_0 \exp(-\mathrm{TE}/T_2)` written out here rather than taken from
# the model that will be fitted.

# sphinx_gallery_start_ignore
# The phantom, the relaxation maps behind it and the coil sensitivities, built
# as :doc:`/auto_examples/01-basics/02-from-kspace-to-image` builds them.
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times
MARGIN = 0.25  # what the field of view leaves around the head

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]

# BrainWeb's volume is indexed (inferior-superior, posterior-anterior,
# left-right), so its first axis selects an axial slice; an image is drawn from
# its first row down, so flipping puts anterior at the top.
fractions = np.flipud(get_mri(sub_id=0, contrast="fuzzy")[SLICE])[..., list(TISSUES)].copy()

# A square field of view around the head, with a margin for the aliasing of an
# undersampled acquisition to fold into.
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

# Where each class sits in ``memberships``, by the name the table gives it.
CLASS = {entries[label]["Tissue"]: index for index, label in enumerate(TISSUES)}

weights = memberships * torch.as_tensor(tissue_pd)[:, None, None]
share = weights.sum(0).clamp(min=1e-6)
T1 = (weights * torch.as_tensor(tissue_t1)[:, None, None]).sum(0) / share
T2 = (weights * torch.as_tensor(tissue_t2)[:, None, None]).sum(0) / share
proton_density = weights.sum(0) / weights.sum(0).max()

# A T1-weighted spin echo, at a repetition time of 600 ms and an echo time of
# 12 ms.
signal = (
    proton_density
    * (1 - torch.exp(-600.0 / T1.clamp(min=1e-3)))
    * torch.exp(-12.0 / T2.clamp(min=1e-3))
)
signal = torch.where(T1 > 0, signal, torch.zeros(()))
signal = signal / signal.max()

# A smooth quadratic phase, so that nothing depends on the image being real.
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
support = memberships.sum(0) > 0.5
amplitude = torch.where(support, proton_density, torch.zeros(()))
t2 = torch.where(support, T2.clamp(20.0, 400.0), torch.tensor(50.0))
# sphinx_gallery_end_ignore

contrasts = (amplitude[None] * torch.exp(-ECHO_TIMES[:, None, None] / t2[None])).to(torch.complex64)

# %%
#
# Acquisition
# -----------
#
# Eight echoes at an echo spacing of 12.5 ms, each sampled at a quarter of the
# phase encodes (:math:`R = 4`) around eight fully sampled central lines, with
# a different random draw per echo, so that the missing phase encodes differ
# between echoes. The echoes are a batch of the encoding rather than an axis
# inside it: the sensitivities and the transform are shared, and only the
# pattern differs, so the operator is the Cartesian SENSE encoding of
# :doc:`../03-regularization/02-operators-and-solvers` with the pattern of
# each echo applied to its samples.

# sphinx_gallery_start_ignore
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
generator = torch.Generator().manual_seed(4)
lines = torch.zeros(ECHOES, 1, SIZE, 1)
for echo in range(ECHOES):
    lines[echo, 0, torch.randperm(SIZE, generator=generator)[: SIZE // ACCELERATION]] = 1.0
    lines[echo, 0, SIZE // 2 - 4 : SIZE // 2 + 4] = 1.0

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

encoding = linop.CartesianSense(sensitivities, (ECHOES, SIZE, SIZE), ndim=2)
E = linop.Diagonal(lines.to(torch.complex64), encoding.oshape) @ encoding

measured = bt.noise(E(contrasts), n=1e-6, s=9)
data = measured / float(E.H(measured).abs().max())

print(f"{E.ishape} -> {E.oshape}")

# %%
#
# The signal model
# ----------------
#
# :class:`bartorch.nlop.MultiEcho` maps parameter maps to one image per echo.
# Its unknowns are :math:`T_2` and a complex amplitude, carried as three real
# maps in a bounded parameterisation rather than in their own units, so
# :meth:`~bartorch.nlop.SignalModel.initial` builds a starting point from
# values and :meth:`~bartorch.nlop.SignalModel.split` reads the fit back.

M = nlop.MultiEcho([float(te) for te in ECHO_TIMES], (SIZE, SIZE))
start = M.initial(T2=80.0)

print(f"unknowns {M.names}: {M.ishapes[0]} -> {M.oshapes[0]}")

# %%
#
# Two routes
# ----------
#
# The first reconstructs the echo images by conjugate gradients on the SENSE
# normal equations and fits the model to them voxel by voxel, which is what
# :func:`bartorch.apps.mobafit` does given the images. The second composes the
# model with the encoding and fits the k-space with
# :class:`bartorch.nlop.IRGNM`. Both are Gauss-Newton loops of the same number
# of steps and differ only in the forward operator that maps the unknowns to
# the data.

STEPS = 20

start_time = time.perf_counter()
images = optim.CG(maxiter=40)(data, E)
two_step = apps.mobafit(images, M, iterations=STEPS, T2=80.0)
print(f"reconstruct, then fit:     {time.perf_counter() - start_time:5.1f} s")

start_time = time.perf_counter()
model_based = nlop.IRGNM(iterations=STEPS, cg_maxiter=100, cg_tol=0.1)(data, E @ M, x0=start)
print(f"model inside the operator: {time.perf_counter() - start_time:5.1f} s")

# %%
#
# ``E @ M`` composes a linear operator with a nonlinear one; the derivative of
# the composition at a point is the encoding applied to the derivative of the
# model, which is the derivative a Gauss-Newton step requires.
#
# The echo images of the first route show what its fit is given. They are
# compared here with the fully sampled echo images of the phantom.

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
#
# Each reconstructed echo carries residual aliasing and noise, which differ
# from echo to echo because each echo has its own sampling pattern. At the
# later echoes the signal has decayed and the relative error grows, so the
# late echoes, which determine :math:`T_2` most, are also the least accurate.
#
# :func:`bartorch.apps.moba` assembles the model-based composition from the
# k-space, the model, the sensitivities and the sampling pattern, and returns
# the maps in their own units. It scales the data by the rule of
# :func:`bartorch.optim.data_scaling` and regularizes each step towards the
# starting maps rather than towards zero, so its result is not identical to
# the fit above.

start_time = time.perf_counter()
one_call = apps.moba(measured, M, sensitivities, pattern=lines, iterations=STEPS, T2=80.0)
print(f"apps.moba:                 {time.perf_counter() - start_time:5.1f} s")

estimates = {
    "two-step": two_step["T2"],
    "model-based": M.split(model_based)["T2"],
    "apps.moba": one_call["T2"],
}

for name, estimate in estimates.items():
    error = float((estimate[support] - t2[support]).norm() / t2[support].norm())
    median = float(estimate[support].median())
    print(f"{name:>22}  median {median:5.1f} ms   relative error {error:.3f}")

print(f"{'phantom':>22}  median {float(t2[support].median()):5.1f} ms")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, 2, width=0.8 * WIDTH)
parameter(axes[0, 0], torch.where(support, t2, torch.zeros(())), "T2", "reference")
for axis, (name, values) in zip(axes.flat[1:], estimates.items()):
    parameter(axis, torch.where(support, values, torch.zeros(())).detach(), "T2", name)
scalebar(figure, axes, name="T2")
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
#
# The two-step :math:`T_2` map is dominated by the errors of the echo images:
# a voxelwise fit cannot distinguish residual aliasing from decay, and in
# voxels where a late echo is too bright or too dark the fitted :math:`T_2` is
# far off. The model-based fits reach a much lower error from the same data,
# since the model admits only images that decay exponentially from echo to
# echo, and the aliasing of eight different sampling patterns is not such an
# image. The fit inside the operator reproduces the phantom almost exactly,
# because every voxel of the phantom decays with a single :math:`T_2`, as the
# model assumes; a measured voxel holding two tissues decays with two, and a
# single exponential cannot represent it. :func:`bartorch.apps.moba`
# regularizes each Gauss-Newton step towards the starting maps, and its
# residual error is largest where :math:`T_2` is farthest from the starting
# value of 80 ms: in cerebrospinal fluid and the scalp. The maps are drawn with
# the navia colormap [#fuderer]_ in a window that spans white and grey matter.
#
# Since the maps are the solver's unknowns, a regularizer passed to the
# linearized problem -- the ``inner`` solver of :func:`bartorch.apps.moba` --
# penalizes the maps rather than the echo images. Without ``sensitivities``,
# :func:`bartorch.apps.moba` estimates the coils jointly with the maps, as
# :doc:`../02-parallel-imaging/02-nonlinear-inversion` estimates them jointly
# with an image. :doc:`../../explanation/nonlinear` explains the model-based
# approach in more detail.

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
