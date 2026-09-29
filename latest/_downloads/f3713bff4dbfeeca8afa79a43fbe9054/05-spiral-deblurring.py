"""
=================================
Off-resonance blurring in spirals
=================================

The blurring that off-resonance produces in a spiral image, simulated for a
known field map and removed with :func:`bartorch.tools.deblur`.

A spin precessing at an offset :math:`f` from the reference frequency
accumulates the phase :math:`2\\pi f t` during the readout. A spiral acquires
each k-space radius at its own time :math:`t(k)`, so the phase is a function
of :math:`|k|`, and a voxel off resonance is convolved with a kernel whose
width grows with :math:`f` and with the readout duration. Conjugate-phase
reconstruction removes the phase at each voxel's own frequency [#noll]_; the
correction here applies it as a sum of a few image-domain convolutions whose
per-voxel weights depend on the field map, in the manner of multifrequency
interpolation [#man]_.
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

SIZE = 128

# %%
#
# Object and field
# ----------------
#
# BART's brain phantom in a field map made of a smooth second-order variation
# and a localized offset near the frontal pole, of the kind an air-tissue
# interface produces, in Hz.

image = bt.phantom(SIZE, geometry="brain").abs()
image = (image / image.max()).to(torch.complex64)
inside = image.abs() > 0

y, x = torch.meshgrid(torch.linspace(-1, 1, SIZE), torch.linspace(-1, 1, SIZE), indexing="ij")
field = 40 * x - 30 * y + 60 * (x**2 + y**2) + 120 * torch.exp(-((y + 0.6) ** 2 + x**2) / 0.05)
field = field - field[inside].mean()
low, high = float(field[inside].min()), float(field[inside].max())
print(f"field over the object: {low:+.0f} to {high:+.0f} Hz")

# %%
#
# The spiral
# ----------
#
# Sixteen interleaves of a variable-density spiral, each a readout of 20 ms
# whose radius grows as :math:`t^{0.6}` to the edge of the 128 grid, in grid
# units. The density compensation is :func:`bartorch.estimate_density`.

INTERLEAVES, SAMPLES, READOUT_S = 16, 2000, 20e-3
progress = np.linspace(0.0, 1.0, SAMPLES)
radius = progress**0.6 * SIZE / 2
angle = 2 * np.pi * SIZE / (2 * INTERLEAVES) * progress
arms = np.stack(
    [
        np.stack(
            [
                radius * np.cos(angle + 2 * np.pi * arm / INTERLEAVES),
                radius * np.sin(angle + 2 * np.pi * arm / INTERLEAVES),
                np.zeros(SAMPLES),
            ],
            axis=-1,
        )
        for arm in range(INTERLEAVES)
    ]
)
trajectory = torch.tensor(arms, dtype=torch.float32)
density = bartorch.estimate_density(trajectory[..., :2].reshape(-1, 2), (SIZE, SIZE))
density = density.reshape(INTERLEAVES, SAMPLES, 1)

# %%
#
# The acquisition is simulated exactly for a field quantized to 128 levels:
# each level's part of the object is transformed with :func:`bartorch.nufft`
# and given the phase its frequency accumulates at each sample time.

sample_time = torch.tensor(progress * READOUT_S, dtype=torch.float32)
levels = torch.linspace(float(field.min()), float(field.max()), 128)
field = levels[(field[..., None] - levels).abs().argmin(-1)]


def acquire(field):
    samples = torch.zeros(INTERLEAVES, SAMPLES, 1, dtype=torch.complex64)
    for frequency in torch.unique(field):
        part = image * (field == frequency)
        accrued = torch.polar(torch.ones(SAMPLES), 2 * math.pi * float(frequency) * sample_time)
        samples += bartorch.nufft(part, trajectory) * accrued[:, None]
    return samples


def grid(samples):
    return bartorch.nufft_adjoint(samples * density, trajectory, image_shape=(SIZE, SIZE))


on_resonance = grid(bartorch.nufft(image, trajectory))
blurred = grid(acquire(field))

# %%
#
# The transfer
# ------------
#
# :class:`~bartorch.tools.ReadoutTiming` tabulates the readout time as
# a function of :math:`|k|^2` from one arm; the interleaves are rotations of
# it, so it serves all of them. :func:`~bartorch.tools.fit_transfer`
# approximates :math:`e^{-2\pi i f t(k)}` over a band of frequencies by a sum
# of terms, each a separable function of k-space times a weight that depends
# on :math:`f` alone. More terms reduce the error of the approximation and
# cost one convolution each.

timing = bt.ReadoutTiming.from_trajectory(arms[0][:, :2], duration=READOUT_S)
band = float(field.abs().max()) + 5.0


def error(estimate):
    return float((estimate - on_resonance).norm() / on_resonance.norm())


print(f"blurred        NRMSE {error(blurred):.3f}")
for terms in (4, 8, 16):
    transfer = bt.fit_transfer(timing, band=band, terms=terms)
    deblurred = bt.deblur(blurred, field, transfer)
    print(
        f"{terms:2d} terms       NRMSE {error(deblurred):.3f}   fit error "
        f"{transfer.error(timing):.1e}   amplification {transfer.amplification:.0f}"
    )

# %%
#
# The error is measured against the gridding reconstruction of the same
# trajectory on resonance. Once the fit error is small, adding terms no longer
# changes the result, and the residual is the error of the conjugate-phase
# approximation itself, which treats the field as constant over the extent of
# each voxel's blurring kernel. A uniform field satisfies that assumption:

uniform = torch.full_like(field, 50.0)
transfer = bt.fit_transfer(timing, band=band, terms=16)
flat = grid(acquire(uniform))
deblurred_flat = bt.deblur(flat, uniform, transfer)
print(f"uniform 50 Hz: blurred {error(flat):.3f}, deblurred {error(deblurred_flat):.3f}")

# %%

# sphinx_gallery_start_ignore
deblurred = bt.deblur(blurred, field, transfer)
peak = float(on_resonance.abs().max())
figure, axes = plt.subplots(1, 4, figsize=(8.0, 2.4), width_ratios=(1, 1, 1, 1.25))
for axis, values, title in (
    (axes[0], on_resonance, "on resonance"),
    (axes[1], blurred, "off resonance"),
    (axes[2], deblurred, "deblurred"),
):
    axis.imshow(values.abs(), cmap="gray", vmin=0, vmax=peak)
    axis.set_title(title)
handle = axes[3].imshow(
    torch.where(inside, field, torch.tensor(float("nan"))), cmap="RdBu_r", vmin=-150, vmax=150
)
axes[3].set_title("field map")
figure.colorbar(handle, ax=axes[3], fraction=0.046, label="Hz")
for axis in axes:
    axis.set_xticks([])
    axis.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The remaining error for the uniform field is the difference between the
# deblurred gridding reconstruction and the on-resonance one, which the
# approximation does not introduce. Where the field varies within the kernel,
# as at the localized offset, the conjugate-phase residual adds to it. An
# iterative reconstruction that includes the field in the encoding operator
# removes that residual at the cost of one transform per time segment.
#
# References
# ----------
#
# .. [#noll] Noll DC, Meyer CH, Pauly JM, Nishimura DG, Macovski A. A homogeneity
#    correction method for magnetic resonance imaging with time-varying
#    gradients. *IEEE Trans Med Imaging* 10(4):629-637 (1991).
#    https://doi.org/10.1109/42.108599
#
# .. [#man] Man LC, Pauly JM, Macovski A. Multifrequency interpolation for fast
#    off-resonance correction. *Magn Reson Med* 37(5):785-792 (1997).
#    https://doi.org/10.1002/mrm.1910370523
