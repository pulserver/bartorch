"""
===================================
EPI Nyquist ghost and ramp sampling
===================================

Two corrections an echo-planar readout needs before its lines form a
Cartesian k-space: the odd/even phase that produces the Nyquist ghost,
estimated from a three-line navigator, and the resampling of samples taken on
the gradient ramps onto the uniform grid.

The data is BART's analytical Shepp-Logan phantom with eight simulated coils,
into which a known readout delay is introduced, so the estimate can be
compared with the delay.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 10,
        "figure.constrained_layout.use": True,
    }
)
# sphinx_gallery_end_ignore
import math

import numpy as np
import torch

import bartorch
import bartorch.tools as bt

# %%
#
# The Nyquist ghost
# -----------------
#
# An EPI train reads alternate lines with readout gradients of opposite
# polarity. A delay between the gradient and the ADC shifts the echo of every
# line by the same time, which is a shift towards positive :math:`k_x` on a
# forward line and towards negative :math:`k_x` on a reversed one. In hybrid
# space -- after the inverse transform along the readout -- a shift in
# :math:`k_x` is a linear phase in :math:`x`, so the reversed lines carry a
# phase the forward ones do not. A phase that alternates from line to line
# modulates k-space at the Nyquist frequency of the phase-encoding direction,
# and the image acquires a copy of the object displaced by half the field of
# view.
#
# Below, forward lines carry :math:`+(a u + b)` and reversed lines
# :math:`-(a u + b)` in hybrid space, with :math:`u \in [-1, 1]` the readout
# coordinate. A reversed line is stored in the order it was digitised, that is
# flipped along the readout.

SIZE = 128
SLOPE, OFFSET = 0.6, 0.25  # rad

kspace = bt.phantom(SIZE, kspace=True, coils=8)[:, 0]  # (coils, ky, kx)
hybrid = bartorch.fft(kspace, axes=(-1,), inverse=True, unitary=True)
u = torch.linspace(-1.0, 1.0, SIZE)


def acquire(row, polarity):
    """One readout of polarity +1 or -1, digitised in the order it was played."""
    delayed = row * torch.polar(torch.ones(SIZE), polarity * (SLOPE * u + OFFSET))
    line = bartorch.fft(delayed, axes=(-1,), unitary=True)
    return torch.flip(line, [-1]) if polarity < 0 else line


train = [(acquire(hybrid[:, ky], 1 - 2 * (ky % 2)), ky % 2 == 1) for ky in range(SIZE)]

# %%
#
# The navigator
# -------------
#
# Three lines of alternating polarity acquired without a phase-encoding blip
# sample the same line of k-space, so the phase between the reversed line and
# the mean of its two neighbours is the odd/even phase alone.
# :func:`~bartorch.tools.estimate_epi_phase` fits a polynomial to it,
# weighted by the signal magnitude and summed over coils. The reversed
# navigator line is passed already flipped into readout order.

centre = hybrid[:, SIZE // 2]
navigator = [acquire(centre, 1), torch.flip(acquire(centre, -1), [-1]), acquire(centre, 1)]

fit = bt.estimate_epi_phase(navigator)
print(f"fitted     constant {float(fit[0]):+.3f}  linear {float(fit[1]):+.3f} rad")
print(f"impressed  constant {2 * OFFSET:+.3f}  linear {2 * SLOPE:+.3f} rad")

# %%
#
# The fitted phase is twice the impressed one, because the navigator measures
# the difference between a forward and a reversed line. The correction is
# applied to the reversed lines only, which rotates them onto the forward ones
# and leaves the image in place.
#
# :func:`~bartorch.tools.correct_lines` flips the reversed lines back
# and, given the fit, removes the phase.


def reconstruct(lines):
    coil_images = bartorch.fft(torch.stack(lines, dim=1), axes=(-2, -1), inverse=True, unitary=True)
    return bartorch.rss(coil_images, axes=(0,)).abs()


ideal = reconstruct([kspace[:, ky] for ky in range(SIZE)])
ghosted = reconstruct(bt.correct_lines(train))
corrected = reconstruct(bt.correct_lines(train, fit))

for name, estimate in (("flipped only", ghosted), ("phase-corrected", corrected)):
    error = float((estimate - ideal).norm() / ideal.norm())
    print(f"{name:>16}: difference from the delay-free image {error:.1e}")

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(8.0, 2.8), width_ratios=(1, 1, 1.3))
peak = float(ideal.max())
for axis, values, title in ((axes[0], ghosted, "flipped only"), (axes[1], corrected, "corrected")):
    axis.imshow(values, cmap="gray", vmin=0, vmax=0.3 * peak)
    axis.set_title(f"{title}, window 0-30 %")
    axis.set_xticks([])
    axis.set_yticks([])
axes[2].semilogy(ghosted[:, SIZE // 2] / peak + 1e-9, label="flipped only")
axes[2].semilogy(corrected[:, SIZE // 2] / peak + 1e-9, label="corrected")
axes[2].set_xlabel("phase-encoding index")
axes[2].set_title("central column")
axes[2].legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The images are windowed to 30 % of the peak, where the ghost at half the
# field of view is visible. What remains after correction is the difference
# between the first-order phase model and the impressed phase, which here is
# zero up to single-precision round-off.
#
# Ramp sampling
# -------------
#
# Sampling during the gradient ramps shortens the echo spacing, but the
# samples are not uniformly spaced in :math:`k_x`. The readout is the
# transform of an object that spans ``support`` pixels, so samples at any
# positions determine it, provided no two neighbouring samples are further
# apart than one over the support, the Nyquist spacing of that object.
# :func:`~bartorch.tools.epi_ramp_operator` [#bruder]_ is the regularized
# least-squares inverse of the transform at the sampled positions followed by
# the transform at the uniform ones.
#
# A trapezoidal readout gradient of 160 samples with ramps of 30 % of its
# duration is simulated for a one-dimensional object of 64 pixels, positions
# in cycles per pixel.

SUPPORT, SAMPLES = 64, 160
profile = torch.zeros(SUPPORT, dtype=torch.complex128)
profile[16:48] = torch.linspace(0.3, 1.0, 32) * torch.exp(1j * torch.linspace(0.0, 2.0, 32))
pixels = (torch.arange(SUPPORT) - SUPPORT // 2).double()

time = torch.linspace(0.0, 1.0, SAMPLES, dtype=torch.float64)
RAMP = 0.3
gradient = torch.clamp(torch.minimum(time / RAMP, (1 - time) / RAMP), 0, 1)
sampled_at = torch.cumsum(gradient, 0)
sampled_at = sampled_at - sampled_at.mean()
sampled_at = 0.5 * sampled_at / sampled_at.abs().max()
uniform_at = torch.arange(SAMPLES, dtype=torch.float64) / SAMPLES - 0.5


def encode(positions):
    return torch.exp(-2j * math.pi * torch.outer(positions, pixels)) @ profile


measured, truth = encode(sampled_at), encode(uniform_at)

operator = bt.epi_ramp_operator(sampled_at, uniform_at, SUPPORT)
resampled = (measured.to(torch.complex64)[None] @ operator.T)[0]

# %%
#
# Linear interpolation between neighbouring samples is the comparison.

linear = torch.complex(
    torch.from_numpy(np.interp(uniform_at, sampled_at, measured.real)),
    torch.from_numpy(np.interp(uniform_at, sampled_at, measured.imag)),
)


def error(estimate):
    return float((estimate.to(torch.complex128) - truth).norm() / truth.norm())


step = float(torch.diff(sampled_at).max()) * SUPPORT
print(f"largest step x support: {step:.2f}  (the samples determine the object below 1)")
print(f"band-limited resampling {error(resampled):.1e}   linear interpolation {error(linear):.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
axes[0].plot(time, sampled_at, label="sampled")
axes[0].plot(time, uniform_at, "--", label="uniform")
axes[0].set_xlabel("readout time")
axes[0].set_ylabel("$k_x$ [cycles/pixel]")
axes[0].legend(fontsize=8)
axes[1].plot(uniform_at, truth.abs(), color="k", label="uniform samples")
axes[1].plot(uniform_at, resampled.abs(), label="resampled")
axes[1].plot(uniform_at, linear.abs(), "--", label="linear")
axes[1].set_xlabel("$k_x$")
axes[1].legend(fontsize=8)
axes[2].semilogy(
    uniform_at, (resampled.to(torch.complex128) - truth).abs() + 1e-18, label="resampled"
)
axes[2].semilogy(uniform_at, (linear - truth).abs() + 1e-18, "--", label="linear")
axes[2].set_xlabel("$k_x$")
axes[2].set_title("error")
axes[2].legend(fontsize=8)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The resampling is exact to the precision of the operator, which is returned
# in single precision; linear interpolation errs most on the plateau, where
# the samples are furthest apart. The ratio printed above is the condition to
# check on a measured trajectory: where a step exceeds one over the support,
# the readout is aliased and no resampling recovers it.
#
# References
# ----------
#
# .. [#bruder] Bruder H, Fischer H, Reinfelder HE, Schmitt F. Image
#    reconstruction for echo planar imaging with nonequidistant k-space
#    sampling. *Magn Reson Med* 23(2):311-323 (1992).
#    https://doi.org/10.1002/mrm.1910230211
