"""
===================================
EPI Nyquist ghost and ramp sampling
===================================

An EPI readout acquires k-space as a train of lines with alternating readout
gradient polarity, and before those lines make a Cartesian k-space you
correct two things. A timing error between the readout gradient and the ADC,
and eddy currents, shift the echoes of the reversed lines against the forward
ones; the odd/even phase this leaves gives you a Nyquist ghost, a copy of the
object shifted by half the field of view along the phase-encoding direction.
And if you sample on the ramps of the readout gradient, to shorten the echo
spacing, your samples are uniform in time but not in :math:`k_x`.

In this tour you simulate an eight-channel EPI acquisition of a Shepp-Logan
phantom with a known ADC delay and constant phase error, estimate the
odd/even phase from a three-line navigator, and remove the ghost. Then you
resample a ramp-sampled readout onto a uniform :math:`k_x` grid.

Prerequisites: lesson 1, :doc:`../01-basics/01-tensors-and-commands`.

**Learning objectives**

- Relate an ADC delay to a linear phase in hybrid space, and that phase to
  the ghost at half the field of view.
- Estimate the odd/even phase from a three-line navigator with
  :func:`~bartorch.tools.estimate_epi_phase`, and apply it to the reversed
  lines with :func:`~bartorch.tools.correct_lines`.
- Measure the ghost by its ghost-to-signal ratio.
- Resample a ramp-sampled readout with
  :func:`~bartorch.tools.epi_ramp_operator`, and check the sampling condition
  under which that resampling is exact.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt


def show(axis, values, title, vmin=0.0, vmax=1.0, cmap="gray"):
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_axis_off()
    return handle


def nrmse(estimate, target):
    return float((estimate - target).norm() / target.norm())


# sphinx_gallery_end_ignore
import math

import torch

import bartorch
import bartorch.tools as bt

# %%
# The Nyquist ghost
# -----------------
#
# Here is the mental model. Say the ADC starts :math:`\delta` dwell times late
# relative to the readout gradient. Every echo then moves by the same time:
# towards positive :math:`k_x` on a forward line and towards negative
# :math:`k_x` on a reversed one. In hybrid space, after the inverse Fourier
# transform along the readout, a shift of :math:`\delta` dwell times is a
# linear phase :math:`\pi \delta u` over the readout coordinate
# :math:`u \in [-1, 1]`. A constant phase :math:`\phi_0`, from eddy currents
# or a :math:`B_0` offset during the readout, adds to it with the same
# alternating sign. So forward lines carry :math:`+(\pi \delta u + \phi_0)`
# and reversed lines :math:`-(\pi \delta u + \phi_0)`. That difference between
# odd and even lines modulates k-space at the Nyquist frequency of the
# phase-encoding direction, and the image picks up a ghost at half the field
# of view.
#
# You simulate it with a delay of a fifth of a dwell time and a constant
# phase of a quarter radian:
SIZE = 128
DELAY = 0.2  # dwell times
PHASE_0 = 0.25  # rad
SLOPE = math.pi * DELAY * (SIZE - 1) / SIZE  # rad over u in [-1, 1]

kspace = bt.phantom(SIZE, kspace=True, coils=8)[:, 0]  # (coils, ky, kx)

# sphinx_gallery_start_ignore
hybrid = bartorch.fft(kspace, axes=(-1,), inverse=True, unitary=True)
u = torch.linspace(-1.0, 1.0, SIZE)


def acquire(row, polarity):
    """One readout of polarity +1 or -1, digitised in the order it was played."""
    phase = polarity * (SLOPE * u + PHASE_0)
    delayed = row * torch.polar(torch.ones(SIZE), phase)
    line = bartorch.fft(delayed, axes=(-1,), unitary=True)
    return torch.flip(line, [-1]) if polarity < 0 else line


# sphinx_gallery_end_ignore

# %%
# ``acquire(row, polarity)`` plays one line of the hybrid-space k-space with
# the delay and the phase above, and returns it in the order the ADC
# digitised it: a reversed line comes out flipped along the readout. Your EPI
# train is a list of ``(line, reversed)`` pairs, which is the form
# :func:`~bartorch.tools.correct_lines` takes:
train = []
for ky in range(SIZE):
    reversed_line = ky % 2 == 1
    polarity = -1 if reversed_line else 1
    train.append((acquire(hybrid[:, ky], polarity), reversed_line))

# %%
# The navigator
# -------------
#
# Before the train, the scanner plays three lines of alternating polarity with
# no phase-encoding blips. They all sample the same line of k-space, so the
# phase between the reversed line and the mean of its two neighbours is the
# odd/even phase and nothing else. :func:`~bartorch.tools.estimate_epi_phase`
# fits a polynomial to it, weighted by the signal magnitude and summed over
# the coils. You pass the reversed navigator line already flipped back into
# readout order.
centre = hybrid[:, SIZE // 2]
navigator = [
    acquire(centre, 1),
    torch.flip(acquire(centre, -1), [-1]),
    acquire(centre, 1),
]

fit = bt.estimate_epi_phase(navigator)
constant, slope = float(fit[0]), float(fit[1])
print(f"fitted     constant {constant:+.3f}  linear {slope:+.3f} rad")
print(f"impressed  constant {2 * PHASE_0:+.3f}  linear {2 * SLOPE:+.3f} rad")

# %%
# The fitted phase is twice the one you put in, because the navigator
# measures the difference between a forward and a reversed line, and they
# carry it with opposite signs.
#
# The correction
# --------------
#
# :func:`~bartorch.tools.correct_lines` flips the reversed lines back into
# readout order and, given the fit, removes the odd/even phase from them. That
# brings them into phase with the forward lines. The phase the forward lines
# keep is common to every line, so it does not change the magnitude image.
# Without a fit it only flips:
ghosted_lines = bt.correct_lines(train)
corrected_lines = bt.correct_lines(train, fit)


def reconstruct(lines):
    stacked = torch.stack(lines, dim=1)  # (coils, ky, kx)
    coil_images = bartorch.ifft(stacked, axes=(-2, -1), unitary=True)
    return bartorch.rss(coil_images, axes=(0,)).abs()


ideal = reconstruct([kspace[:, ky] for ky in range(SIZE)])
ghosted = reconstruct(ghosted_lines)
corrected = reconstruct(corrected_lines)

# %%
# You measure the ghost by the ghost-to-signal ratio: the mean signal outside
# the object divided by the mean signal inside it.

# sphinx_gallery_start_ignore
inside = ideal > 0.05 * ideal.max()
outside = ideal < 0.01 * ideal.max()


def ghost_to_signal(image):
    return float(image[outside].mean() / image[inside].mean())


for name, image in (
    ("delay-free", ideal),
    ("flipped only", ghosted),
    ("corrected", corrected),
):
    print(
        f"{name:>12}: ghost-to-signal {100 * ghost_to_signal(image):5.2f} %, "
        f"NRMSE {nrmse(image, ideal):.1e}"
    )

peak = float(ideal.max())
figure, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
for axis, (values, title) in zip(
    axes,
    ((ideal, "delay-free"), (ghosted, "flipped only"), (corrected, "corrected")),
):
    show(axis, values / peak, title, 0, 0.3)
plt.show()

readout = bartorch.fft(torch.stack(navigator), axes=(-1,), inverse=True, unitary=True)
cross = (0.5 * (readout[0] + readout[2]) * readout[1].conj()).sum(0)
keep = (cross.abs() > 0.1 * cross.abs().max()).numpy()
fitted = (fit[0] + fit[1] * u.double()).numpy()
figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
axes[0].plot(u[keep], torch.angle(cross)[keep], ".", color="C7")
axes[0].plot(u, fitted, color="C0")
axes[0].set_xlabel("readout coordinate u")
axes[0].set_ylabel("odd/even phase [rad]")
axes[0].set_title("navigator phase: dots, fit: line", fontsize=11)
column = SIZE // 2
axes[1].semilogy(ideal[:, column] / peak + 1e-6, color="C7", linewidth=3.0, label="delay-free")
axes[1].semilogy(ghosted[:, column] / peak + 1e-6, color="C1", label="flipped only")
axes[1].semilogy(corrected[:, column] / peak + 1e-6, "--", color="C0", label="corrected")
axes[1].set_ylim(1e-3, 3)
axes[1].set_xlabel("phase-encoding index")
axes[1].set_title("central column")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The images run from zero to 30 % of the peak, with phase encoding vertical.
# With the lines only flipped, you see the ghost at the top and bottom of the
# field of view, overlapping the object where it wraps. On the left, the
# navigator phase falls on a straight line, and the first-order fit goes
# through it. With that fit applied, the ghost-to-signal ratio is back to that
# of the delay-free image, whose signal outside the object is the truncation
# ringing of the phantom; in the central column the corrected profile lies on
# the delay-free one. It comes back exactly because the phase error you
# simulated is exactly first order. On measured data, higher orders of the
# phase, and phase errors that differ between lines of the same polarity,
# leave a residual ghost.
#
# Ramp sampling
# -------------
#
# If you sample on the ramps of the trapezoidal readout gradient, the samples
# land at :math:`k_x(t) = \gamma \int_0^t G_x(\tau)\, d\tau`: denser on the
# ramps than on the plateau. The readout is the Fourier transform of an object
# that spans ``support`` pixels, so samples at any positions determine it, as
# long as no two neighbouring samples are further apart than one over the
# support, the Nyquist spacing of that object.
# :func:`~bartorch.tools.epi_ramp_operator` [#bruder]_ is a matrix: the
# regularized least-squares inverse of the transform at the sampled positions,
# followed by the transform at the uniform ones.
#
# Your readout gradient is a trapezoid whose ramps each take 30 % of the ADC
# window, sampled with 160 samples, and the object is one-dimensional, 64
# pixels wide. Positions are in cycles per pixel.
SUPPORT, SAMPLES = 64, 160
RAMP = 0.3

time = torch.linspace(0.0, 1.0, SAMPLES, dtype=torch.float64)
gradient = torch.clamp(torch.minimum(time / RAMP, (1 - time) / RAMP), 0, 1)
sampled_at = torch.cumsum(gradient, 0)
sampled_at = sampled_at - sampled_at.mean()
sampled_at = 0.5 * sampled_at / sampled_at.abs().max()
uniform_at = torch.arange(SAMPLES, dtype=torch.float64) / SAMPLES - 0.5

# sphinx_gallery_start_ignore
import numpy as np

profile = torch.zeros(SUPPORT, dtype=torch.complex128)
profile[16:48] = torch.linspace(0.3, 1.0, 32) * torch.exp(1j * torch.linspace(0.0, 2.0, 32))
pixels = (torch.arange(SUPPORT) - SUPPORT // 2).double()


def encode(positions):
    return torch.exp(-2j * math.pi * torch.outer(positions, pixels)) @ profile


figure, axis = plt.subplots(figsize=(7.2, 3.2))
axis.plot(time, gradient, color="C7", label="$G_x / G_{max}$")
axis.plot(time, 2 * sampled_at, color="C0", label="$k_x / k_{max}$, sampled")
axis.plot(time, 2 * uniform_at, "--", color="C1", label="$k_x / k_{max}$, uniform")
axis.set_xlabel("time in the ADC window")
axis.set_title("trapezoidal readout gradient")
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The sampled :math:`k_x` curves on the ramps and is straight on the plateau,
# where it is steeper than the uniform grid you want to end on.
#
# ``measured`` is the readout of a 64-pixel profile at the sampled positions
# and ``truth`` the same readout on the uniform grid; how they are simulated
# is hidden. You build the operator once from the trajectory and apply it to
# every line of the train:
measured, truth = encode(sampled_at), encode(uniform_at)

operator = bt.epi_ramp_operator(sampled_at, uniform_at, SUPPORT)
resampled = measured.to(torch.complex64) @ operator.T

# %%
# Linear interpolation between neighbouring samples is the comparison. You
# judge both on the image profile, the inverse transform of the uniform
# samples, and you check the sampling condition: the largest step between
# samples, times the support, has to stay below one.
step = float(torch.diff(sampled_at).max()) * SUPPORT
print(f"largest step x support: {step:.2f}")

# sphinx_gallery_start_ignore
resampled = resampled.to(torch.complex128)
linear = torch.complex(
    torch.from_numpy(np.interp(uniform_at, sampled_at, measured.real)),
    torch.from_numpy(np.interp(uniform_at, sampled_at, measured.imag)),
)


def image_profile(samples):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(samples))).abs()


for name, estimate in (
    ("band-limited resampling", resampled),
    ("linear interpolation", linear),
):
    error = nrmse(image_profile(estimate), image_profile(truth))
    print(f"{name:>24}: image NRMSE {error:.1e}")

figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
position = torch.arange(SAMPLES) - SAMPLES // 2
reference = image_profile(truth)
axes[0].plot(position, reference, color="C7", linewidth=3.0, label="uniform")
axes[0].plot(position, image_profile(resampled), color="C0", label="resampled")
axes[0].plot(position, image_profile(linear), "--", color="C1", label="linear")
axes[0].set_xlim(-50, 50)
axes[0].set_xlabel("pixel")
axes[0].set_title("image profile")
for estimate, colour, style in ((resampled, "C0", "-"), (linear, "C1", "--")):
    axes[1].semilogy(
        position,
        (image_profile(estimate) - reference).abs() / reference.max() + 1e-9,
        style,
        color=colour,
    )
axes[1].set_xlim(-80, 80)
axes[1].set_ylim(1e-8, 1)
axes[1].set_xlabel("pixel")
axes[1].set_title("|error| / peak")
figure.legend(*axes[0].get_legend_handles_labels(), loc="outside right upper")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The step stays below one, so the resampling is exact up to the precision
# of the operator, which is returned in single precision: its error sits near
# :math:`10^{-6}` of the peak and below. Linear interpolation errs most on
# the plateau, where the samples are furthest apart, and in the image its
# error spreads over the whole field of view, inside and outside the object,
# at up to a few per cent of the peak. On a measured trajectory the printed
# ratio is the thing to check: where a step exceeds one over the support, the
# readout is aliased and no resampling recovers it.
#
# Your turn: a longer delay
# -------------------------
#
# Change the ADC delay, acquire the train and the navigator again, and compare
# the ghost-to-signal ratio with the lines only flipped and with the fit
# applied:
for delay in (0.1, 0.5, 1.0):
    SLOPE = math.pi * delay * (SIZE - 1) / SIZE
    lines = []
    for ky in range(SIZE):
        polarity = -1 if ky % 2 else 1
        lines.append((acquire(hybrid[:, ky], polarity), polarity < 0))
    navigator = [
        acquire(centre, 1),
        torch.flip(acquire(centre, -1), [-1]),
        acquire(centre, 1),
    ]
    fit = bt.estimate_epi_phase(navigator)
    before = ghost_to_signal(reconstruct(bt.correct_lines(lines)))
    after = ghost_to_signal(reconstruct(bt.correct_lines(lines, fit)))
    print(f"delay {delay:.1f}: {100 * before:5.2f} % -> {100 * after:5.2f} %")

# %%
# The ghost grows with the delay, and the correction takes it back to the
# delay-free level every time: the phase you simulate stays first order
# whatever its slope, so the navigator fit describes it exactly. Try adding a
# quadratic term to the phase in ``acquire`` and see what is left.

# %%
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Given an EPI train as a list of (line, reversed) pairs, each line a
#    (coils, readout) tensor in acquisition order, and a three-line
#    navigator (forward, reversed flipped into readout order, forward),
#    estimate the first-order odd/even phase with bt.estimate_epi_phase and
#    apply it with bt.correct_lines(train, fit). Report the ghost-to-signal
#    ratio before and after. For a ramp-sampled readout with known sample
#    positions in cycles per pixel, build bt.epi_ramp_operator(sampled,
#    uniform, support), apply it to every line, and check that the largest
#    sample step times the support is below 1.

# %%
#
# References
# ----------
#
# .. [#bruder] Bruder H, Fischer H, Reinfelder HE, Schmitt F. Image
#    reconstruction for echo planar imaging with nonequidistant k-space
#    sampling. *Magn Reson Med* 23(2):311-323 (1992).
#    https://doi.org/10.1002/mrm.1910230211
