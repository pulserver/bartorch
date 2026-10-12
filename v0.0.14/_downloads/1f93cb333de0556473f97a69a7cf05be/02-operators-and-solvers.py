"""
=============================
6. Custom encoding and solver
=============================

In lesson 5 you handed ``pics`` the data, the maps and a penalty, and it did
the rest. In this lesson you assemble the same reconstruction from its
parts. You build the SENSE encoding operator, check its adjoint, run the
solver on it, and obtain exactly what ``pics`` returns, bit for bit. Then
you extend the operator with one written in Python, and differentiate
through it.

The mental model: :func:`bartorch.apps.pics` builds three objects and runs
BART's iteration on them: the encoding operator :math:`A = PFS`
(sensitivities, Fourier transform, sampling), the regularization terms of
:mod:`bartorch.priors`, and an iterative algorithm of :mod:`bartorch.optim`.
You need the parts whenever BART has no application for what you want to
do: an encoding with an extra factor, such as an off-resonance or motion
phase; a solver called from your own outer loop; an operator written in
Python; or a gradient with respect to the data, to train a network.

**Learning objectives**

- Build :class:`bartorch.linop.CartesianSense` and read the plan it was
  lowered into.
- Check an operator's adjoint with the dot-product test.
- Prepare the data as ``pics`` does and reproduce
  :func:`bartorch.apps.pics` bit for bit with :class:`bartorch.optim.FISTA`.
- Compose operators, include one defined in Python, and differentiate
  through an application.

Previous: :doc:`01-regularized-reconstruction`. Next:
:doc:`../04-non-cartesian/01-trajectories-and-transforms`, where you replace
the FFT with a NUFFT and leave the Cartesian grid.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

WIDTH = 7.8  # inches, the width of the documentation column at 110 dpi


def panels(columns, rows=1, width=WIDTH, bars=0):
    """A row (or grid) of frameless square image panels, leaving room for
    ``bars`` colorbars in each row."""
    side = (width - 0.9 * bars) / columns
    figure, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(width, rows * (side + 0.35) + 0.2),
    )
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


def scaled(estimate, reference):
    """``estimate`` scaled to ``reference`` in the least-squares sense."""
    a, b = estimate.abs().double(), reference.abs().double()
    return (float((a * b).sum() / (a * a).sum()) * a).float()


# The phantom: the BrainWeb [#brainweb]_ slice of lesson 2, rendered as a
# T1-weighted spin-echo image with a smooth phase, and BART's eight-channel
# head coil.
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

import torch

import bartorch
import bartorch.tools as bt

SIZE = 192
COILS = 8
ACCELERATION = 3
CALIBRATION = 24
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)
MARGIN = 0.25
TR, TE = 600.0, 12.0  # ms

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
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
T1 = (weights * torch.as_tensor(tissue_t1)[:, None, None]).sum(0) / share
T2 = (weights * torch.as_tensor(tissue_t2)[:, None, None]).sum(0) / share
proton_density = weights.sum(0) / weights.sum(0).max()
signal = (
    proton_density * (1 - torch.exp(-TR / T1.clamp(min=1e-3))) * torch.exp(-TE / T2.clamp(min=1e-3))
)
signal = torch.where(T1 > 0, signal, torch.zeros(()))
signal = signal / signal.max()
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE),
    torch.linspace(-1.0, 1.0, SIZE),
    indexing="ij",
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)
peak = float(image.abs().max())

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)

# The acquisition of lesson 2 with a tenth of its noise: a variable-density
# random third of the phase encodes around a fully sampled ACS block.
kspace = bt.noise(bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True), n=1e-5, s=42)
encodes = torch.arange(SIZE) - SIZE // 2
centre = (encodes.abs() < CALIBRATION // 2).to(torch.float32)
drawn = torch.multinomial(
    (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0 * (1.0 - centre),
    SIZE // ACCELERATION - CALIBRATION,
    replacement=False,
    generator=torch.Generator().manual_seed(11),
)
lines = centre.clone()
lines[drawn] = 1.0
kspace = kspace[:, None] * lines.reshape(SIZE, 1).to(torch.complex64)
# sphinx_gallery_end_ignore

# %%
# The encoding operator
# ---------------------
#
# Your scan is the one of lesson 2 [#brainweb]_, with a tenth of its noise:
# eight channels, ``kspace`` of shape ``(coils, 1, y, x)`` with one
# phase-encoding line in three acquired around a 24-line ACS block. ESPIRiT
# gives you the maps, as before.
#
# :class:`bartorch.linop.CartesianSense` is :math:`A = PFS` as one operator.
# You give it the sensitivities, the shape of the image, and the sampling
# pattern; the shape of the k-space it maps to follows from those.
# :func:`bartorch.tools.pattern` reads the pattern off the data, as you would
# for a prospectively undersampled scan.
#
# ``modulated=True`` selects the uncentred sample convention that BART's
# applications iterate in; the default is the centred convention of
# :func:`bartorch.fft`. The two differ by a modulation of the samples and
# give the same image. The choice matters only when you apply the operator to
# data already in one of them, as you do below.
import torch

import bartorch
import bartorch.tools as bt

from bartorch import apps, linop, optim, priors

maps = bt.ecalib(kspace, maps=1, calib_size=CALIBRATION, crop=0.8)
pattern = bt.pattern(kspace)

A = linop.CartesianSense(
    maps.squeeze(1),
    (SIZE, SIZE),
    pattern.squeeze(),
    modulated=True,
)
print(f"{A.ishape} -> {A.oshape}")
print(A.plan)
print(f"fused: {A.plan.fused}")

# %%
# The operator maps an image of ``(y, x)`` to k-space of ``(coils, y, x)``.
# ``A.plan`` reports how it is computed: which transform, what multiplies
# the image and the samples, and how the normal operator :math:`A^H A` is
# applied. It is read back from the operator that was built, not predicted.
# ``plan.fused`` is false where the composition could not be expressed as one
# encoding and fell back to a chain, which gives the same numbers more slowly.
#
# The adjoint
# -----------
#
# A solver needs :math:`A^H`, not :math:`A^T`, and one built on the wrong one
# converges to the wrong image. You can check it with the definition,
# :math:`\langle Ax, y\rangle = \langle x, A^H y\rangle`, which holds for
# any pair of vectors, random ones included:
generator = torch.Generator().manual_seed(0)
probe = torch.randn(A.ishape, dtype=torch.complex64, generator=generator)
samples = torch.randn(A.oshape, dtype=torch.complex64, generator=generator)

forward = (A(probe).conj() * samples).sum()
adjoint = (probe.conj() * A.H(samples)).sum()
print(f"relative difference {abs(forward - adjoint) / abs(forward):.2e}")

# %%
# The two inner products agree to single-precision round-off.
#
# Solving
# -------
#
# A solver is called as ``solver(y, A)``. To get the answer ``pics`` gets,
# you give it the data ``pics`` iterates on rather than the array the scanner
# wrote: the sampling pattern applied, the modulation into the uncentred
# convention, and the data divided by the scale
# :func:`bartorch.optim.data_scaling` estimates.
measured = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
scale = optim.data_scaling(measured)
data = (measured / scale).squeeze(1)

term = priors.Wavelet(axes=(-1, -2), weight=0.002)
assembled = optim.FISTA(term, maxiter=100)(data, A) * scale

# %%
# The solution is multiplied back by the scale, which puts the image into the
# units of the k-space, as ``pics`` does unless given ``rescale=False``.
# Now run ``pics`` on the raw ``kspace`` with the same term and solver, and
# compare:
tool = apps.pics(kspace, maps, regularizers=term, solver="fista", maxiter=100)
same = torch.equal(assembled.squeeze(), tool.squeeze())
print(f"identical to pics: {same}")


def nrmse(estimate):
    return bt.nrmse(image.abs(), estimate.abs(), scaled=True)


print(f"NRMSE, adjoint {nrmse(A.H(data)):.3f}")
print(f"NRMSE, FISTA   {nrmse(assembled):.3f}")

# sphinx_gallery_start_ignore
figure, axes = panels(2, rows=2, bars=1)
show(axes[0, 0], image, "the object", vmax=peak)
show(axes[0, 1], scaled(A.H(data), image), "adjoint, $A^H y$", vmax=peak)
show(axes[1, 0], scaled(assembled, image), "FISTA, wavelet", vmax=peak)
handle = show(
    axes[1, 1],
    (scaled(assembled, image) - image.abs()).abs() / peak,
    "|error|, FISTA",
    cmap="magma",
    vmax=0.1,
)
figure.colorbar(handle, ax=axes[1, 1], fraction=0.046, label="|error| / peak")
# A colorbar of the same width keeps the upper panels aligned with the lower.
figure.colorbar(handle, ax=axes[0, 1], fraction=0.046).ax.set_visible(False)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The two are the same iteration over the same operator, so they return the
# same bits, not merely close ones. In the figure, the adjoint :math:`A^H y`
# is not the inverse: it is the sensitivity-weighted coil combination of the
# zero-filled k-space, and it carries the aliasing of the undersampling and
# the shading of :math:`\sum_c |S_c|^2`. The solve removes both, as the NRMSE
# printed above shows, and its remaining error sits at the tissue boundaries.
#
# Operator algebra
# ----------------
#
# ``@`` composes, ``+`` adds, ``A.H`` is the adjoint and ``A.gram()`` the
# normal operator :math:`A^H A`. A composition builds one BART operator
# rather than a Python chain, so a solver iterating on it does not return to
# Python between applications. :func:`bartorch.optim.maxeigen` runs the power
# iteration on an operator; the largest eigenvalue of :math:`A^H A` is the
# Lipschitz constant of the least-squares gradient, which bounds the
# gradient step size.
print(f"largest eigenvalue of A^H A: {optim.maxeigen(A.gram()):.3f}")

# %%
# An operator of your own is composed with BART's operators through
# :meth:`~bartorch.linop.LinearOperator.from_callbacks`: you give it the input
# and output shapes, a forward and an adjoint, and BART calls them back. Here
# it is a spatially varying phase, as an off-resonance or eddy-current phase
# would be, placed between the image and the encoding:
field = torch.exp(1j * 0.4 * torch.pi * grid_x).to(torch.complex64)
phase = linop.LinearOperator.from_callbacks(
    (SIZE, SIZE),
    (SIZE, SIZE),
    lambda u: field * u,
    lambda u: field.conj() * u,
)
composed = A @ phase
print(f"{composed.ishape} -> {composed.oshape}, fused: {composed.plan.fused}")

# %%
# ``composed`` has the same interface as ``A`` (forward, adjoint, normal
# operator) and is passed to a solver in the same way.
#
# Differentiation
# ---------------
#
# Apply an operator to a tensor that requires a gradient, and autograd
# records it. For :math:`y = Ax` the gradient torch propagates back is
# :math:`A^H g`, not :math:`A^T g`, which is torch's conjugate Wirtinger
# convention for complex tensors. A real :math:`A` has :math:`A^H = A^T`, so
# only a complex check tells the two apart. Here the gradient of
# :math:`\|Ax - y\|^2` at :math:`x = 0` should be :math:`-2A^H y`:
variable = data.new_zeros(A.ishape).requires_grad_(True)
residual = A(variable) - data
(residual.abs() ** 2).sum().backward()

expected = 2 * A.H(-data)
difference = (variable.grad - expected).abs().max() / expected.abs().max()
print(f"relative difference from 2 A^H (Ax - y): {float(difference):.2e}")

# %%
# :doc:`../../explanation/learned-reconstruction` describes the backward
# passes of the solvers, and :doc:`../../explanation/solvers` which algorithm
# applies to which problem.
#
# Exercise: FISTA convergence
# ---------------------------
#
# ``optim.FISTA`` is a loop over :class:`bartorch.optim.FISTABlock`, one
# iteration per call. Run the loop yourself for 200 iterations and record the
# error of the image after each one:
block = optim.FISTABlock(term)
state = block.start(data, A)
trace = []
for _ in range(200):
    state = block(state, A)
    trace.append(nrmse(block.output(state, A)))

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 0.42 * WIDTH))
axis.plot(range(1, len(trace) + 1), trace, lw=1.8, color="C0", label="FISTA")
axis.axvline(100, color="C1", lw=1.2, ls="--", label="maxiter=100")
axis.set_xlabel("iteration")
axis.set_ylabel("NRMSE")
axis.set_ylim(0, None)
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The error falls steeply over the first few tens of iterations, reaches its
# minimum before iteration 50, and then stays close to that level with small
# fluctuations up to iteration 200. For this data and weight, the 100
# iterations used above are past the point where more iterations reduce the
# error. Try ``step=0.5`` in :class:`~bartorch.optim.FISTABlock`, or a larger
# weight, and compare the curves.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With bartorch, build linop.CartesianSense from ESPIRiT maps, the image
#    shape and the sampling pattern of a 2D Cartesian 8-channel k-space
#    (modulated=True), print its plan, and check its adjoint with a random
#    dot-product test. Prepare the data as pics does (pattern, fftmod
#    inverse, divided by optim.data_scaling), solve with optim.FISTA and
#    priors.Wavelet over the last two axes (weight 0.002, 100 iterations),
#    and check torch.equal against apps.pics with the same term. Compose the
#    operator with a Python phase operator from from_callbacks, and check
#    that autograd's gradient of ||Ax - y||^2 at zero is -2 A^H y.

# %%
#
# References
# ----------
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
