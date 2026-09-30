"""
===============================
Subspace-constrained T1 mapping
===============================

This lesson estimates a :math:`T_1` map from a single continuous
inversion-recovery acquisition in which each of four hundred time points is
encoded by one radial spoke. The aim is to show how a signal model turns a
hopelessly undersampled time series into a well-posed reconstruction: the
recovery curves of all plausible :math:`T_1` values span a subspace of low
dimension, and reconstructing the few coefficients of that subspace instead of
the individual frames reduces the number of unknowns by two orders of
magnitude.

The sequence is an inversion pulse followed by a train of spoiled gradient
echoes with a small flip angle, each echo read out along one golden-angle
spoke, as in MPnRAGE and radial Look-Locker methods. The longitudinal
magnetization recovers from inversion towards a steady state at an apparent
rate that depends on :math:`T_1`, the flip angle and the repetition time, so
each voxel follows one of a family of recovery curves. The family is
simulated as a dictionary, and its dominant singular vectors
:math:`\\Phi` form the basis of a subspace [#tamir]_. The time series is written
as :math:`x_t = \\sum_a \\Phi_{at} \\alpha_a`, and the basis enters the encoding
on the k-space side, between the transform of each frame and its samples:

.. math::

   y[c, t] = \\sum_a \\Phi_{at} \\, \\mathrm{NUFFT}_t \\!\\left( S_c \\, \\alpha_a \\right).

The coefficient maps :math:`\\alpha_a` are reconstructed under a locally
low-rank penalty, any frame of the series can be synthesized from them, and
:math:`T_1` is estimated by matching each voxel's coefficients against the
dictionary.

The phantom and the coil sensitivities are built as in
:doc:`../01-basics/02-from-kspace-to-image`; the cell that does it is hidden on
this page and present in the script this page can be downloaded as.

**Learning objectives**

- Simulate a dictionary of inversion-recovery curves and extract a
  low-dimensional subspace from it by the singular value decomposition.
- Include a subspace basis in a non-Cartesian encoding.
- Reconstruct coefficient maps under a locally low-rank penalty, and
  synthesize images at any inversion time from them.
- Estimate :math:`T_1` by dictionary matching in the subspace, and identify
  the partial-volume bias of a voxelwise fit.

It follows :doc:`../04-non-cartesian/03-dynamic-golden-angle`, whose frames
are constrained here by a linear signal model. The next lesson,
:doc:`02-quantitative-models`, fits a nonlinear one directly to k-space.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap
from matplotlib.colors import ListedColormap

WIDTH = 8.0  # inches, the width of the documentation column

# Fuderer et al. (Magn Reson Med 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()
# Phase is cyclic, so the colormap has to be: -pi and +pi are the same colour.
# mygbm, turned so that zero phase is yellow and +/-pi is blue.
MYGBM = Colormap("colorcet:CET_C2").to_matplotlib().reversed()
PHASE = ListedColormap(MYGBM([((step + 60) % 256) / 255 for step in range(256)]))

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


def domain(axis, values, title=None):
    """A complex map the way a coil sensitivity is read: phase in colour,
    magnitude in brightness."""
    values = values.detach().cpu()
    colours = PHASE((values.angle() / (2 * np.pi) + 0.5).numpy())[..., :3]
    magnitude = values.abs().numpy()
    magnitude = magnitude / max(float(magnitude.max()), 1e-12)
    axis.imshow(colours * magnitude[..., None])
    if title is not None:
        axis.set_title(title)


def phase_bar(figure, axes):
    """The colour-to-phase key for the panels beside it."""
    bar = figure.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(-np.pi, np.pi), PHASE),
        ax=axes,
        fraction=0.046,
        ticks=[-np.pi, 0.0, np.pi],
    )
    bar.ax.set_yticklabels(["$-\\pi$", "0", "$\\pi$"])
    bar.set_label("phase [rad]")


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
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri
from torchsim.simulators import MPnRAGESimulator

import bartorch
import bartorch.tools as bt
from bartorch import linop, optim, priors

SIZE = 128
COILS = 8
FRAMES = 400
RANK = 4

TR = 4.1  # ms
FLIP = 6.0  # degrees

# %%
#
# The dictionary and its subspace
# -------------------------------
#
# The dictionary is simulated rather than tabulated: one curve per
# :math:`T_1`, from the sequence that will be played.
# :class:`~torchsim.simulators.MPnRAGESimulator` is that sequence -- an
# inversion followed by a spoiled gradient-echo train with every shot read --
# and ``simulate`` evaluates it over an array of parameters at once, giving
# ``(entries, frames)``.
#
# The same object serves the fit: handed to :func:`bartorch.nlop.Bloch` it is
# a model operator, which is how :doc:`02-quantitative-models`
# solves for the maps directly. Here only its forward evaluation is wanted.

t1_values = torch.linspace(100.0, 4500.0, 200)  # ms

sequence = MPnRAGESimulator(nshots=FRAMES, flip=FLIP, TR=TR)
dictionary = torch.as_tensor(sequence.simulate(T1=t1_values))

left = torch.linalg.svd(dictionary.T.to(torch.complex64), full_matrices=False)[0]
basis = left[:, :RANK].T.contiguous()

print(f"dictionary {tuple(dictionary.shape)}, basis {tuple(basis.shape)}")

# %%
#
# The left panel shows a few entries of the dictionary: the signal starts
# negative after the inversion, passes through zero at a time that grows with
# :math:`T_1`, and approaches the steady state of the gradient-echo train. The
# curves are smooth and similar to each other, so a few singular vectors
# represent them all. The singular values in the right panel decay by more
# than two orders of magnitude over the first four, and the rank used here,
# four, is marked by the dashed line. The rank is a modelling decision: too
# few coefficients bias the recovered curves toward the span of the basis, too
# many increase the number of unknowns the undersampled data must determine.

# sphinx_gallery_start_ignore
spectrum = torch.linalg.svdvals(dictionary.T.to(torch.complex64))
figure, (left_axis, right_axis) = plt.subplots(
    1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"width_ratios": (1.4, 1.0)}, layout="constrained"
)
time_ms = np.arange(FRAMES) * TR
steady = dictionary[:, -1:].conj() / dictionary[:, -1:].abs()
for value in (300.0, 800.0, 1400.0, 4000.0):
    entry = int((t1_values - value).abs().argmin())
    curve = (torch.as_tensor(dictionary[entry]) * steady[entry]).real
    left_axis.plot(time_ms, curve.numpy(), label=f"$T_1$ = {t1_values[entry]:.0f} ms")
left_axis.axhline(0.0, color="#8a8a8a", lw=0.6)
left_axis.set_xlabel("time after the inversion [ms]")
left_axis.set_ylabel("signal [a.u.]")
left_axis.legend(fontsize=8)
right_axis.semilogy(range(1, 13), (spectrum[:12] / spectrum[0]).cpu().numpy(), marker="o", ms=4)
right_axis.axvline(RANK + 0.5, color="#8a8a8a", ls="--")
right_axis.set_xlabel("index")
right_axis.set_ylabel("singular value, relative to the first")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Phantom
# -------
#
# Each tissue class is given the :math:`T_1` of the BrainWeb table and its own
# curve from the same signal model, and the series is the membership-weighted
# sum of them. A voxel holding two tissues therefore follows a sum of two
# recovery curves, which is not itself an inversion-recovery curve -- the
# partial-volume error any voxelwise fit carries.

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
# One inversion-recovery curve per tissue class, from the same signal model the
# dictionary came from, combined by membership and proton density.
curves = torch.as_tensor(sequence.simulate(T1=torch.as_tensor(tissue_t1)))
series = torch.zeros(FRAMES, SIZE, SIZE, dtype=torch.complex64)
occupancy = torch.zeros(SIZE, SIZE)
for index in range(len(TISSUES)):
    weighted = memberships[index] * float(tissue_pd[index])
    series += weighted[None].to(torch.complex64) * curves[index][:, None, None]
    occupancy += weighted
# sphinx_gallery_end_ignore

# %%
#
# Acquisition and reconstruction
# ------------------------------
#
# One golden-angle spoke per repetition time, four hundred of them in
# 1.64 s after the inversion. Each frame is sampled by a single spoke, about
# :math:`1/200` of what a fully sampled frame needs; together the spokes of
# the train cover k-space densely. The trajectory indexes frames as well as
# samples, and the image the encoding operator maps from is the four
# coefficient maps rather than the four hundred frames, with ``basis``
# contracting the one into the other.

trajectory = bt.traj(readout=SIZE, spokes=FRAMES, radial=True, golden=True)
trajectory = trajectory.reshape(FRAMES, 1, SIZE, 3)

# sphinx_gallery_start_ignore
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

frames = linop.NoncartesianSense(sensitivities, (FRAMES, SIZE, SIZE), traj=trajectory)
measured = bt.noise(frames(series), n=1e-7, s=5)

A = linop.NoncartesianSense(sensitivities, (RANK, SIZE, SIZE), traj=trajectory, basis=basis)
print(f"{A.ishape} -> {A.oshape}")
print(A.plan)

# %%
#
# ``plan.contraction`` reports the subspace and its rank, and the normal
# operator is a point spread function over the basis as well as the
# trajectory, so an iteration does not transform the four hundred frames.
#
# The penalty is locally low rank [#llr]_: the coefficient maps are stacked
# into a matrix per block of voxels, and its nuclear norm is penalized.
# ``joint_axes`` makes the coefficients the columns of that matrix, so the
# penalty favours neighbouring voxels that follow the same few curves, rather
# than coefficient maps that are each sparse. Penalizing the maps one at a
# time does not couple the coefficients of a voxel.

data = measured / optim.data_scaling(measured[..., None], A=A)
term = priors.LocallyLowRank(axes=(-1, -2), weight=0.005, joint_axes=(-3,), block=8)
coefficients = optim.ADMM(term, maxiter=30)(data, A)

recovered = torch.einsum("af,ayx->fyx", basis.to(torch.complex64), coefficients)

# %%
#
# Parameter fit
# -------------
#
# The recovered coefficients are matched against the dictionary projected onto
# the same subspace, by the normalized inner product, which is dictionary
# matching performed in four dimensions rather than four hundred. Matching in
# the subspace and matching the reconstructed curves differ only by the
# component of the dictionary the basis discards.

atoms = basis.to(torch.complex64) @ dictionary.T.to(torch.complex64)
atoms = atoms / atoms.norm(dim=0, keepdim=True)
voxels = coefficients.reshape(RANK, -1)
voxels = voxels / voxels.norm(dim=0, keepdim=True).clamp(min=1e-12)

matched = (atoms.conj().T @ voxels).abs().argmax(0)
t1_map = t1_values[matched].reshape(SIZE, SIZE)

# %%
#
# The fit is reported where the proton density is high enough for a curve to be
# defined, and separately for the voxels each tissue class dominates.

support = occupancy > 0.2 * float(occupancy.max())
dominant = memberships.argmax(0)
pure = memberships.max(0).values > 0.7

for name, index in CLASS.items():
    selected = support & pure & (dominant == index)
    if int(selected.sum()) < 20:
        continue
    estimate = float(t1_map[selected].median())
    print(
        f"{name:>13}  table {tissue_t1[index]:6.0f} ms"
        f"   fitted {estimate:6.0f} ms   ({int(selected.sum())} voxels)"
    )

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(RANK, width=WIDTH)
for column in range(RANK):
    magnitude = coefficients[column].abs()
    show(axes[0, column], magnitude, f"$\\alpha_{column + 1}$", vmax=float(magnitude.max()))
figure.suptitle("coefficient maps, magnitude, each on its own scale")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Each coefficient map is the weight of one singular vector of the dictionary.
# The first resembles a proton-density-weighted image, since the first singular
# vector is close to the mean recovery curve; the later ones encode the
# differences between the curves of short and long :math:`T_1`, and are not
# images of a tissue contrast.
#
# Images at any inversion time
# ----------------------------
#
# The coefficient maps determine the whole series: multiplying by the basis
# synthesizes the image at every one of the four hundred time points, each of
# which was measured with a single spoke. Early after the inversion the
# longitudinal magnetization is negative in every tissue; each tissue then
# passes through zero at its own null time, shortest for white matter, and
# approaches the steady state of the gradient-echo train.


# sphinx_gallery_start_ignore
def null_frame(label):
    curve = curves[CLASS[label]]
    oriented = (curve * curve[-1].conj()).real
    return int(torch.nonzero(oriented > 0)[0])


shown = (8, null_frame("WM"), null_frame("GM"), FRAMES - 1)
top = float(series.abs().max())
reference_frames = series.abs()
recovered_frames = scaled(recovered, series)
figure, axes = panels(len(shown), 2, width=0.9 * WIDTH)
for column, frame in enumerate(shown):
    show(axes[0, column], reference_frames[frame], f"t = {frame * TR:.0f} ms", vmax=top)
    show(axes[1, column], recovered_frames[frame], vmax=top)
axes[0, 0].text(
    -0.06,
    0.5,
    "reference",
    rotation=90,
    va="center",
    ha="right",
    transform=axes[0, 0].transAxes,
    color="#8a8a8a",
)
axes[1, 0].text(
    -0.06,
    0.5,
    "recovered",
    rotation=90,
    va="center",
    ha="right",
    transform=axes[1, 0].transAxes,
    color="#8a8a8a",
)
figure.suptitle("magnitude images after the inversion")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The four frames are shortly after the inversion, when every tissue is
# inverted and bright in magnitude; at the null of white matter, which appears
# dark; at the null of grey matter, where white matter, only just past its own
# null, is dark as well; and at the end of the train, in the steady state.
# The recovered frames reproduce these contrast changes, each from its single
# spoke, with blurring at the tissue boundaries.
#
# The curve of a single voxel shows the same with its sign. The complex
# signal is rotated so that its steady state is positive and real, and the
# reconstruction, which determines the series up to a global complex scale
# because the data are normalized before the solve, is scaled to the
# reference in the least-squares sense.

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.7 * WIDTH, 3.2))
time_ms = np.arange(FRAMES) * TR
for (name, label), colour in zip(
    (("white matter", "WM"), ("grey matter", "GM"), ("CSF", "CSF")), ("C0", "C1", "C2")
):
    voxel = torch.nonzero(support & pure & (dominant == CLASS[label]))
    voxel = voxel[len(voxel) // 2]
    truth = series[:, voxel[0], voxel[1]]
    estimate = recovered[:, voxel[0], voxel[1]]
    estimate = estimate * (estimate.conj() @ truth) / (estimate.conj() @ estimate)
    rotation = truth[-1].conj() / truth[-1].abs()
    axis.plot(time_ms, (truth * rotation).real.numpy(), lw=2.6, color=colour, alpha=0.45)
    axis.plot(
        time_ms, (estimate * rotation).real.numpy(), lw=1.0, ls="--", color=colour, label=name
    )
axis.axhline(0.0, color="#8a8a8a", lw=0.6)
axis.set_xlabel("time after the inversion [ms]")
axis.set_ylabel("signal [a.u.]")
axis.set_title("reference (thick) and recovered (dashed)")
axis.legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The :math:`T_1` map
# -------------------
#
# The maps are drawn with the lipari colormap [#fuderer]_, in a window that
# spans white and grey matter; cerebrospinal fluid, beyond it, saturates. The
# difference map is in milliseconds.

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH / 3 + 0.4), layout="constrained")
for axis in axes:
    axis.set_axis_off()
parameter(axes[0], torch.where(support, T1, torch.zeros(())), "T1", "reference")
parameter(axes[1], torch.where(support, t1_map, torch.zeros(())), "T1", "fitted")
cmap, limits, label = STYLE["T1"]
figure.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(*limits), cmap), ax=axes[:2], shrink=0.8, label=label
)
difference = torch.where(support, (t1_map - T1).abs(), torch.zeros(()))
handle = show(axes[2], difference.numpy(), "|fitted - reference|", vmax=300.0, cmap="magma")
figure.colorbar(handle, ax=axes[2], shrink=0.8, label="[ms]")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The printed table compares the median fitted :math:`T_1` with the tabulated
# value in the voxels a single tissue class dominates. Grey matter agrees
# closely. White matter is overestimated, consistent with its recovered curve
# above, which lies between the reference curves of white and grey matter: the
# locally low-rank penalty shares information between neighbouring voxels,
# and the thin white-matter structures border grey matter everywhere.
#
# Cerebrospinal fluid is strongly underestimated, and the difference map
# saturates in the ventricles. Its :math:`T_1` is poorly determined by this
# acquisition: the recovery observed during a gradient-echo train is governed
# by the apparent relaxation time
# :math:`T_1^* = (1/T_1 - \ln\cos\alpha / T_R)^{-1}` [#deichmann]_, which
# for a flip angle :math:`\alpha` of 6 degrees and :math:`T_R` of 4.1 ms is
# below 750 ms for any :math:`T_1`. The curves of long :math:`T_1` therefore
# differ from each other by little more than the error of the reconstruction.
# A smaller flip angle or a longer train increases the sensitivity to long
# :math:`T_1`.
#
# At the boundaries between tissues the fit is biased for a different reason:
# a voxel holding two tissues follows the sum of two recovery curves, which is
# not itself a recovery curve, and the dictionary entry that matches it best
# has a :math:`T_1` between the two. This partial-volume bias belongs to any
# voxelwise fit, not to the subspace.
#
# Estimating the parameters directly from k-space, without an intermediate
# series or a subspace, is :doc:`02-quantitative-models`.

# %%
#
# References
# ----------
#
# .. [#tamir] Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. T2
#    shuffling: sharp, multicontrast, volumetric fast spin-echo imaging.
#    *Magn Reson Med* 77(1):180-195 (2017). https://doi.org/10.1002/mrm.26102
#
# .. [#llr] Zhang T, Pauly JM, Levesque IR. Accelerating parameter mapping with a
#    locally low rank constraint. *Magn Reson Med* 73(2):655-661 (2015).
#    https://doi.org/10.1002/mrm.25161
#
# .. [#deichmann] Deichmann R, Haase A. Quantification of T1 values by SNAPSHOT-FLASH
#    NMR imaging. *J Magn Reson* 96(3):608-612 (1992).
#    https://doi.org/10.1016/0022-2364(92)90347-A
#
# .. [#fuderer] Fuderer M, Wichtmann B, Crameri F, de Souza NM, Baeßler B, Gulani V,
#    et al. Color-map recommendation for MR relaxometry maps. *Magn Reson Med*
#    93(2):490-506 (2025). https://doi.org/10.1002/mrm.30290
