"""
===============================
Subspace-constrained T1 mapping
===============================

This example estimates a :math:`T_1` map from a single continuous
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

The coefficient maps :math:`\\alpha_a` are reconstructed under a
total-variation penalty, any frame of the series can be synthesized from them, and
:math:`T_1` is estimated by matching each voxel's coefficients against the
dictionary.

The phantom and the coil sensitivities are built as in
:doc:`../01-basics/02-from-kspace-to-image`; the cell that does it is hidden on
this page and present in the script this page can be downloaded as.

**Prerequisites.** :doc:`../04-non-cartesian/02-radial-sense` and
:doc:`../05-model-based/01-quantitative-models`.

**Learning objectives**

- Simulate a dictionary of inversion-recovery curves and extract a
  low-dimensional subspace from it by the singular value decomposition.
- Include a subspace basis in a non-Cartesian encoding.
- Reconstruct coefficient maps under a total-variation penalty, and
  synthesize images at any inversion time from them.
- Estimate :math:`T_1` by dictionary matching in the subspace, compare it
  with matching frames reconstructed one at a time, and identify the
  partial-volume bias of a voxelwise fit.

The frames of :doc:`01-dynamic-golden-angle` are constrained here by a linear
signal model; :doc:`../05-model-based/01-quantitative-models` fits a nonlinear
one directly to k-space.
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
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from blochsim.simulators import MPnRAGESimulator
from brainweb_dl import get_mri

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
# :class:`~blochsim.simulators.MPnRAGESimulator` is that sequence -- an
# inversion followed by a spoiled gradient-echo train with every shot read --
# and ``simulate`` evaluates it over an array of parameters at once, giving
# ``(entries, frames)``.
#
# The same object serves the fit: handed to :func:`bartorch.nlop.Bloch` it is
# a model operator, which is how :doc:`../05-model-based/01-quantitative-models`
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
    1, 2, figsize=(WIDTH, 3.6), gridspec_kw={"width_ratios": (1.5, 1.0)}, layout="constrained"
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
left_axis.legend(loc="lower right")
left_axis.set_title("dictionary entries")
right_axis.semilogy(range(1, 13), (spectrum[:12] / spectrum[0]).cpu().numpy(), marker="o", ms=4)
right_axis.axvline(RANK + 0.5, color="#8a8a8a", ls="--")
right_axis.set_xticks(range(2, 13, 2))
right_axis.set_xlabel("index")
right_axis.set_ylabel("relative singular value")
right_axis.set_title("singular values")
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
# contracting the one into the other. Complex Gaussian noise of variance
# :math:`10^{-5}` per sample is added to the simulated samples, which puts the
# signal-to-noise ratio of white matter, in a fully sampled image of the
# steady state, at the value printed below.

trajectory = bt.traj(readout=SIZE, spokes=FRAMES, radial=True, golden=True)
trajectory = trajectory.reshape(FRAMES, 1, SIZE, 3)

# sphinx_gallery_start_ignore
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

frames = linop.NoncartesianSense(sensitivities, (FRAMES, SIZE, SIZE), traj=trajectory)
measured = bt.noise(frames(series), n=1e-5, s=5)

# sphinx_gallery_start_ignore
# The noise as the signal-to-noise ratio of white matter in the steady state,
# in an image fully sampled with the same noise per sample: the NUFFT is
# scaled so that a unitary transform leaves the noise standard deviation
# per voxel equal to that per sample.
white_matter = memberships[CLASS["WM"]] > 0.9
steady = float(series[-1].abs()[white_matter].mean())
print(f"white-matter SNR of a fully sampled steady-state image: {steady / 1e-5**0.5:.0f}")
# sphinx_gallery_end_ignore

A = linop.NoncartesianSense(sensitivities, (RANK, SIZE, SIZE), traj=trajectory, basis=basis)
print(f"{A.ishape} -> {A.oshape}")
print(A.plan)

# %%
#
# ``plan.contraction`` reports the subspace and its rank, and the normal
# operator is a point spread function over the basis as well as the
# trajectory, so an iteration does not transform the four hundred frames.
#
# The penalty is the total variation [#rof]_ of each coefficient map over the
# two spatial axes. A tissue follows one recovery curve throughout, so its
# coefficients are piecewise constant, and the noise, which the data
# determine least well in the coefficients of the weaker singular vectors,
# is not. A locally low-rank penalty [#llr]_ over blocks of voxels is the
# other common choice; a block that straddles two tissues is rank two, and
# shrinking its second singular value mixes their curves and biases the
# fitted :math:`T_1` towards the neighbouring tissue.
#
# The ADMM penalty parameter ``rho`` weights the auxiliary variable, which
# starts at zero, against the data in each update of the coefficients. At
# the default of 0.5 the coefficients of the weaker singular vectors, which
# carry the differences between recovery curves, are still biased towards
# zero after forty iterations, and the fitted :math:`T_1` with them; 0.05
# lets the data determine them within that number of iterations.

data = measured / optim.data_scaling(measured[..., None], A=A)
term = priors.TotalVariation(axes=(-1, -2), weight=0.001)
coefficients = optim.ADMM(term, maxiter=40, rho=0.05)(data, A)

recovered = torch.einsum("af,ayx->fyx", basis.to(torch.complex64), coefficients)

# %%
#
# The frames reconstructed one at a time are the reference point: the
# density-compensated adjoint of the encoding without the basis, which is the
# gridding reconstruction of each frame from its single spoke.

weights = torch.linalg.norm(trajectory.real[..., :2], dim=-1).clamp(min=0.25)
gridded = frames.H(measured * weights.to(torch.complex64))

for name, estimate in (("frame by frame", gridded), ("subspace", recovered)):
    print(
        f"{name:>14}  NRMSE of the series {bt.nrmse(series.abs(), estimate.abs(), scaled=True):.3f}"
    )

# %%
#
# Parameter fit
# -------------
#
# The recovered coefficients are matched against the dictionary projected onto
# the same subspace, by the normalized inner product, which is dictionary
# matching performed in four dimensions rather than four hundred. Matching in
# the subspace and matching the reconstructed curves differ only by the
# component of the dictionary the basis discards. The frame-by-frame series
# has no subspace, and is matched against the dictionary itself.


def match(voxels, atoms):
    """The T1 of the dictionary atom with the largest normalized inner product."""
    voxels = voxels.reshape(len(atoms), -1)
    voxels = voxels / voxels.norm(dim=0, keepdim=True).clamp(min=1e-12)
    atoms = atoms / atoms.norm(dim=0, keepdim=True)
    return t1_values[(atoms.conj().T @ voxels).abs().argmax(0)].reshape(SIZE, SIZE)


t1_map = match(coefficients, basis.to(torch.complex64) @ dictionary.T.to(torch.complex64))
t1_gridded = match(gridded, dictionary.T.to(torch.complex64))

# %%
#
# The fit is reported where the proton density is high enough for a curve to be
# defined, and separately in the interior of each tissue class: the voxels
# one class dominates, less a one-voxel rim, so that the numbers are not
# those of partial volume.

support = occupancy > 0.2 * float(occupancy.max())
dominant = memberships.argmax(0)
pure = support & (memberships.max(0).values > 0.7)


def erode(mask):
    """The mask less a one-voxel rim."""
    return -torch.nn.functional.max_pool2d(-mask.float()[None], 3, 1, 1)[0] > 0


core = {index: erode(pure & (dominant == index)) for index in CLASS.values()}
interior = torch.stack(list(core.values())).any(0)

print(f"{'':>13}  {'table':>7}  {'frame by frame':>14}  {'subspace':>8}   [ms]")
for name, index in CLASS.items():
    selected = core[index]
    if int(selected.sum()) < 20:
        continue
    print(
        f"{name:>13}  {tissue_t1[index]:7.0f}  {float(t1_gridded[selected].median()):14.0f}"
        f"  {float(t1_map[selected].median()):8.0f}   ({int(selected.sum())} voxels)"
    )
white = core[CLASS["WM"]]
for name, estimate in (("frame by frame", t1_gridded), ("subspace", t1_map)):
    relative = (estimate - T1).abs() / T1.clamp(min=1.0)
    print(
        f"{name:>14}  mean relative T1 error: interior {float(relative[interior].mean()):.3f},"
        f" whole head {float(relative[support].mean()):.3f};"
        f"  white-matter standard deviation {float(estimate[white].std()):.0f} ms"
    )

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, 2, width=0.75 * WIDTH)
for column, axis in enumerate(axes.flat):
    magnitude = coefficients[column].abs()
    show(axis, magnitude, f"$\\alpha_{column + 1}$", vmax=float(magnitude.max()))
figure.suptitle("coefficient maps, each on its own scale")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Each coefficient map is the weight of one singular vector of the dictionary.
# The first resembles a proton-density-weighted image, since the first singular
# vector is close to the mean recovery curve; the later ones encode the
# differences between the curves of short and long :math:`T_1`, and are not
# images of a tissue contrast. The basis is orthonormal, so the noise is
# spread over the four maps alike while the signal falls with the singular
# value: the fourth map has the lowest signal-to-noise ratio of the four.
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


shown = (8, null_frame("WM"), FRAMES - 1)
top = float(series.abs().max())
reference_frames = series.abs()
recovered_frames = scaled(recovered, series)
figure, axes = panels(len(shown), 2)
for column, frame in enumerate(shown):
    show(axes[0, column], reference_frames[frame], f"t = {frame * TR:.0f} ms", vmax=top)
    show(axes[1, column], recovered_frames[frame], vmax=top)
for axis, label in zip(axes[:, 0], ("reference", "subspace")):
    axis.text(-0.04, 0.5, label, rotation=90, va="center", ha="right", transform=axis.transAxes)
plt.show()

frame = null_frame("WM")
figure, axes = panels(3)
show(axes[0, 0], reference_frames[frame], "reference", vmax=top)
show(axes[0, 1], scaled(gridded[frame], series[frame]), "frame by frame", vmax=top)
show(axes[0, 2], recovered_frames[frame], "subspace", vmax=top)
figure.suptitle(f"t = {frame * TR:.0f} ms")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The three frames are shortly after the inversion, when every tissue is
# inverted and bright in magnitude; at the null of white matter, which
# appears dark; and at the end of the train, in the steady state. The
# subspace reconstruction reproduces these contrast changes, each frame from
# its single spoke. The same frame reconstructed on its own is the streak
# pattern of one spoke, with no anatomy left in it.
#
# The curve of a single voxel shows the same with its sign. The complex
# signal is rotated so that its steady state is positive and real, and the
# reconstruction, which determines the series up to a global complex scale
# because the data are normalized before the solve, is scaled to the
# reference in the least-squares sense.

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.8 * WIDTH, 3.8))
time_ms = np.arange(FRAMES) * TR
for (name, label), colour in zip(
    (("white matter", "WM"), ("grey matter", "GM"), ("CSF", "CSF")), ("C0", "C1", "C2")
):
    voxel = torch.nonzero(core[CLASS[label]])
    voxel = voxel[len(voxel) // 2]
    truth = series[:, voxel[0], voxel[1]]
    estimate = recovered[:, voxel[0], voxel[1]]
    estimate = estimate * (estimate.conj() @ truth) / (estimate.conj() @ estimate)
    rotation = truth[-1].conj() / truth[-1].abs()
    axis.plot(time_ms, (truth * rotation).real.numpy(), lw=2.6, color=colour, alpha=0.45)
    axis.plot(
        time_ms, (estimate * rotation).real.numpy(), lw=1.2, ls="--", color=colour, label=name
    )
axis.axhline(0.0, color="#8a8a8a", lw=0.6)
axis.set_xlabel("time after the inversion [ms]")
axis.set_ylabel("signal [a.u.]")
axis.set_title("reference (thick) and subspace (dashed)")
axis.legend(loc="lower right")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The :math:`T_1` map
# -------------------
#
# The maps are drawn with the lipari colormap [#fuderer]_, in a window that
# spans white and grey matter; cerebrospinal fluid, beyond it, saturates. The
# difference maps are relative to the reference :math:`T_1`.

# sphinx_gallery_start_ignore
figure, axes = panels(3)
parameter(axes[0, 0], torch.where(support, T1, torch.zeros(())), "T1", "reference")
parameter(axes[0, 1], torch.where(support, t1_gridded, torch.zeros(())), "T1", "frame by frame")
parameter(axes[0, 2], torch.where(support, t1_map, torch.zeros(())), "T1", "subspace")
scalebar(figure, axes[0], name="T1")
plt.show()

figure, axes = panels(2, width=0.85 * WIDTH)
for axis, name, estimate in (
    (axes[0, 0], "frame by frame", t1_gridded),
    (axes[0, 1], "subspace", t1_map),
):
    difference = torch.where(support, 100 * (estimate - T1).abs() / T1.clamp(min=1.0), 0.0)
    handle = show(axis, difference.numpy(), name, vmax=25.0, cmap="magma")
figure.colorbar(handle, ax=axes[0], fraction=0.046, label="$|\\Delta T_1| / T_1$ [%]")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The printed table compares the median fitted :math:`T_1` with the tabulated
# value in the interior of each tissue class. The frames reconstructed one at
# a time still yield a :math:`T_1` map: the aliasing of each frame differs
# from that of the next, so along the recovery curve it is incoherent, and
# the match to the dictionary rejects much of it, which is the principle of
# MR fingerprinting [#mrf]_. What it does not reject remains as a
# voxel-to-voxel scatter over the whole head, visible in the white matter of
# the difference map and in its standard deviation. The subspace
# reconstruction fits the coefficient maps to all spokes at once under the
# total-variation penalty, and inside each tissue its error is a fraction of
# that of the frame-by-frame match.
#
# Cerebrospinal fluid is the tissue the frame-by-frame match underestimates
# most. Its :math:`T_1` is poorly determined by this acquisition: the
# recovery observed during a gradient-echo train is governed by the apparent
# relaxation time :math:`T_1^* = (1/T_1 - \ln\cos\alpha / T_R)^{-1}`
# [#deichmann]_, which for a flip angle :math:`\alpha` of 6 degrees and
# :math:`T_R` of 4.1 ms is below 750 ms for any :math:`T_1`. The curves of
# long :math:`T_1` therefore differ from each other by little, and noise
# moves the match along the dictionary; a smaller flip angle or a longer
# train increases the sensitivity to long :math:`T_1`.
#
# What both difference maps share is the rim of every tissue. A voxel holding
# two tissues follows the sum of two recovery curves, which is not itself a
# recovery curve, and the dictionary entry that matches it best has a
# :math:`T_1` between the two. This partial-volume bias belongs to any
# voxelwise fit, not to the subspace, and it is why the whole-head error is
# larger than the interior one for both. The scalp fat is a layer one to two
# voxels thick, with no interior at this resolution, and is fitted between
# its own :math:`T_1` and that of its neighbours.
#
# Estimating the parameters directly from k-space, without an intermediate
# series or a subspace, is :doc:`../05-model-based/01-quantitative-models`.

# %%
#
# References
# ----------
#
# .. [#tamir] Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. T2
#    shuffling: sharp, multicontrast, volumetric fast spin-echo imaging.
#    *Magn Reson Med* 77(1):180-195 (2017). https://doi.org/10.1002/mrm.26102
#
# .. [#rof] Rudin LI, Osher S, Fatemi E. Nonlinear total variation based noise removal
#    algorithms. *Physica D* 60(1-4):259-268 (1992).
#    https://doi.org/10.1016/0167-2789(92)90242-F
#
# .. [#llr] Zhang T, Pauly JM, Levesque IR. Accelerating parameter mapping with a
#    locally low rank constraint. *Magn Reson Med* 73(2):655-661 (2015).
#    https://doi.org/10.1002/mrm.25161
#
# .. [#mrf] Ma D, Gulani V, Seiberlich N, Liu K, Sunshine JL, Duerk JL, Griswold MA.
#    Magnetic resonance fingerprinting. *Nature* 495(7440):187-192 (2013).
#    https://doi.org/10.1038/nature11971
#
# .. [#deichmann] Deichmann R, Haase A. Quantification of T1 values by SNAPSHOT-FLASH
#    NMR imaging. *J Magn Reson* 96(3):608-612 (1992).
#    https://doi.org/10.1016/0022-2364(92)90347-A
#
# .. [#fuderer] Fuderer M, Wichtmann B, Crameri F, de Souza NM, Baeßler B, Gulani V,
#    et al. Color-map recommendation for MR relaxometry maps. *Magn Reson Med*
#    93(2):490-506 (2025). https://doi.org/10.1002/mrm.30290
