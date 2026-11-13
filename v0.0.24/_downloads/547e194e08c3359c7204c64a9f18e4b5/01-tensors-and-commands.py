"""
=======================
1. Your data as tensors
=======================

If you have used BART before, you know its arrays as CFL files on disk and its
commands as programs you call from a shell. bartorch runs the same BART inside
your Python process: every command is a Python function that takes and returns
:class:`torch.Tensor` objects, with no files and no subprocess in between. In
this lesson you see what your data look like on that side: how a multichannel
image and its k-space are laid out as tensors, which Fourier transform takes
one to the other, and how a BART command line maps onto a function call.

Every later lesson relies on these conventions. A reconstruction that is off
by a factor of :math:`\\sqrt{N}`, or shifted by half a voxel because the
k-space centre is misplaced, usually results from misreading one of them.

**Learning objectives**

- Read a tensor shape as BART's dimension vector, and find the receive-channel
  axis in it.
- Make a phantom as an image, as coil images and as analytical k-space.
- Know where :func:`bartorch.fft` puts the k-space centre and how it
  normalizes, and check both against :func:`numpy.fft.fftn`.
- See why analytical k-space, cut to the acquired matrix, reconstructs with
  Gibbs ringing.
- Call a BART command from Python, and the same command from its command line.

Next: :doc:`02-from-kspace-to-image`, where you reconstruct an undersampled
scan with the functions you meet here.
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


# sphinx_gallery_end_ignore

# %%
# A multichannel image
# --------------------
#
# BART stores an array in Fortran order and describes it by a dimension vector
# whose first entry varies fastest: readout (``x``), then the phase-encoding
# directions (``y``, ``z``), then the receive channels in BART's dimension 3,
# then further dimensions for sets of maps, echoes, frames and so on. A torch
# tensor holding the same bytes in C order has the reversed shape, so no copy
# is made between the two. A 2D multichannel image is therefore
# ``(coils, 1, y, x)``: the 1 is BART's ``z``, and the readout is the last
# axis. :doc:`/explanation/data-layout` lists the layouts you meet in the rest
# of the course.
#
# :func:`bartorch.tools.phantom` is BART's ``phantom`` command. On its own it
# gives you the Shepp-Logan image; with ``coils`` it gives you that image as
# seen by each channel of BART's analytical eight-channel head coil.
import numpy as np
import torch

import bartorch
import bartorch.tools as bt

SIZE = 128
COILS = 8

image = bt.phantom(SIZE)
coil_images = bt.phantom(SIZE, coils=COILS)

print(f"image {tuple(image.shape)}, {image.dtype}")
print(f"coil images {tuple(coil_images.shape)}")
print(f"as BART sees them {list(reversed(coil_images.shape))}")

# sphinx_gallery_start_ignore
figure, axes = panels(2, rows=2, width=0.8 * WIDTH)
# The skull is the brightest structure; a window at a third of the peak shows
# the brain, where the sensitivity shading is read.
window = float(coil_images.abs().max()) / 3
for axis, channel in zip(axes.flat, (0, 2, 4, 6)):
    axis.imshow(coil_images[channel, 0].abs(), vmin=0, vmax=window)
    axis.set_title(f"channel {channel}")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each channel sees the object through its own sensitivity: bright next to its
# coil element, dark on the far side of the head. The channel axis, first in
# the tensor shape, is BART's dimension 3, the fourth entry of the reversed
# shape.
#
# Combining the channels
# ----------------------
#
# The usual way to make one magnitude image from a receive array is the root
# sum of squares (RSS) over the channels, :math:`\sqrt{\sum_c |x_c|^2}`.
# :func:`bartorch.rss` takes the channel axis as a tensor axis, as every
# function in bartorch does, rather than as BART's bitmask:
combined = bartorch.rss(coil_images, axes=(0,))
print(f"combined {tuple(combined.shape)}")

# %%
# Fourier transform
# -----------------
#
# What a channel measures is the Fourier transform of its coil image, sampled
# on the k-space grid. :func:`bartorch.fft` is BART's *centred* transform: the
# k-space centre (DC) and the image centre both sit at index ``n // 2``, which
# is how scanners deliver k-space and how you display it. It is unnormalized
# unless you ask for ``unitary=True``.
kspace = bartorch.fft(coil_images, axes=(-2, -1), unitary=True)

# %%
# You can check this against NumPy. For an even matrix size, the centred
# transform is NumPy's transform between an ``ifftshift`` and an ``fftshift``:
shifted = np.fft.ifftshift(coil_images.numpy(), axes=(-2, -1))
reference = np.fft.fftshift(
    np.fft.fft2(shifted, norm="ortho"),
    axes=(-2, -1),
)

difference = np.abs(kspace.numpy() - reference).max() / np.abs(reference).max()
ratio = float(coil_images.norm() / kspace.norm())
print(f"largest difference, relative to the peak: {difference:.1e}")
print(f"norm ratio image / k-space: {ratio:.6f}")

# sphinx_gallery_start_ignore
figure, axes = panels(2, bars=1)
axes[0, 0].imshow(combined[0].abs(), vmin=0, vmax=float(combined.abs().max()))
axes[0, 0].set_title("root sum of squares")
log_k = torch.log10(kspace[0, 0].abs() / kspace.abs().max()).clamp(min=-5)
handle = axes[0, 1].imshow(log_k, cmap="magma", vmin=-5, vmax=0)
axes[0, 1].set_title("k-space of channel 0")
figure.colorbar(handle, ax=axes[0, 1], fraction=0.046, label="$\\log_{10}$ (|signal| / peak)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The two transforms differ by single-precision round-off. The unitary
# transform keeps the :math:`\ell_2` norm (Parseval's theorem), which is why
# the norm ratio is one: white noise has the same standard deviation in
# k-space as in the image.
#
# On the left is the RSS image, with the shading the summed coil
# sensitivities leave on it. On the right is the k-space of one channel, on a
# logarithmic scale. It spans five orders of magnitude: most of the energy is
# at the centre, which sets the contrast, while the periphery carries the
# edges and the fine detail. This is what variable-density undersampling
# exploits in lesson 2.
#
# Analytical k-space
# ------------------
#
# The k-space above is the discrete transform of a sampled image, so
# transforming it back gives you that image exactly. A scanner does not
# measure that. It measures the continuous object, whose spectrum never ends,
# and keeps only the part inside :math:`\pm k_{\max}`. ``kspace=True`` asks
# ``phantom`` for exactly that: the Fourier transform of its ellipses,
# evaluated analytically at the sample positions. Simulating data with the
# same discrete model you then invert is the *inverse crime* [#guerquin]_;
# analytical k-space avoids it, and is the more realistic test data.
analytical = bt.phantom(SIZE, kspace=True)
truncated = bartorch.ifft(analytical, axes=(-2, -1))

error = float((truncated - image).norm() / image.norm())
print(f"relative difference from the sampled phantom: {error:.3f}")

# sphinx_gallery_start_ignore
figure, axes = panels(3, bars=1)
axes[0, 0].imshow(image.abs(), vmin=0, vmax=1)
axes[0, 0].set_title("sampled phantom")
axes[0, 1].imshow(truncated.abs(), vmin=0, vmax=1)
axes[0, 1].set_title("analytical k-space")
handle = axes[0, 2].imshow((truncated - image).abs(), cmap="magma", vmin=0, vmax=0.3)
axes[0, 2].set_title("|difference|")
figure.colorbar(handle, ax=axes[0, 2], fraction=0.046, label="fraction of peak")
plt.show()

row = SIZE // 2
figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.9), width_ratios=(2, 1))
for axis, limits in zip(axes, ((0, SIZE - 1), (4, 30))):
    axis.plot(image[row].abs(), color="C0", lw=1.8, label="sampled phantom")
    axis.plot(truncated[row].abs(), color="C1", lw=1.2, label="from analytical k-space")
    axis.set_xlim(*limits)
    axis.set_ylim(-0.05, 1.5)
    axis.set_xlabel("x [voxel]")
axes[0].set_ylabel("magnitude")
axes[0].set_title(f"profile along row {row}")
axes[1].set_title("skull edge, enlarged")
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc="outside lower center", ncols=2)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The difference is largest at the edges of the ellipses and fades away from
# them. The profile through the centre shows the ringing itself: the image
# from analytical k-space overshoots at every step in intensity and oscillates
# next to it with a period of about two voxels. That is the truncation
# artefact you see next to sharp edges on a scanner.
#
# Here you use the unnormalized inverse, because ``phantom`` scales its
# k-space so that the unnormalized inverse returns the image's intensities.
#
# The command line
# ----------------
#
# A function of :mod:`bartorch.tools` takes BART's options as keyword
# arguments, under their long names or their single-letter flags, and reverses
# the shape of every tensor it hands to BART. The ``bartorch`` console command
# takes BART's own command lines and works on CFL files, so a script written
# for the ``bart`` executable runs against the same library.
# :func:`bartorch.cli.main` runs one such command line in your process.
# :func:`bartorch.io.writecfl` wants the array in BART's order, which is the
# transpose of a C-order array:
import tempfile

from pathlib import Path

with tempfile.TemporaryDirectory() as directory:
    source = str(Path(directory) / "coils")
    target = str(Path(directory) / "kspace")
    bartorch.io.writecfl(source, coil_images.numpy().T)
    status = bartorch.cli.main(["fft", "-u", "3", source, target])
    from_files = torch.as_tensor(bartorch.cli.read(target))

print(f"exit status {status}")
print(f"identical to bartorch.fft: {torch.equal(from_files, kspace)}")

# %%
# ``-u`` asks for the unitary transform, and ``3`` is BART's bitmask for its
# dimensions 0 and 1 (readout and the first phase encoding), which are the last
# two tensor axes. Both routes run the same BART function on the same bytes, so
# the results are identical, bit for bit.
#
# One difference matters later. A BART command is not recorded by autograd, so
# its result has no gradient with respect to its inputs. The operators of
# :mod:`bartorch.linop` and the solvers of :mod:`bartorch.optim`, which you
# meet in lesson 6 (:doc:`../03-regularization/02-operators-and-solvers`), are
# differentiable. :doc:`../../explanation/execution-model` describes both
# routes.
#
# Exercise: matrix size
# ---------------------
#
# Does acquiring more of k-space get rid of the ringing? Repeat the comparison
# with analytical k-space at a few matrix sizes:
sizes = (64, 128, 256, 512)
errors, overshoots = [], []
for n in sizes:
    sampled = bt.phantom(n)
    from_kspace = bartorch.ifft(bt.phantom(n, kspace=True), axes=(-2, -1))
    errors.append(float((from_kspace - sampled).norm() / sampled.norm()))
    overshoots.append(float(from_kspace.abs().max() / sampled.abs().max()))

for n, e, o in zip(sizes, errors, overshoots):
    print(f"{n:>4}: relative difference {e:.3f}, peak {o:.2f} x the phantom's")

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(0.75 * WIDTH, 0.42 * WIDTH))
axis.plot(sizes, errors, "o-", lw=1.8, label="relative difference")
axis.plot(sizes, [o - 1 for o in overshoots], "o-", lw=1.8, label="overshoot")
axis.set_xscale("log", base=2)
axis.set_xticks(sizes, [str(n) for n in sizes])
axis.set_xlabel("matrix size")
axis.set_ylim(0, None)
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The relative difference falls by about :math:`\sqrt{2}` for every doubling
# of the matrix: the ringing occupies a band of a few voxels along each edge,
# a shrinking fraction of the image. The overshoot does not decrease: the
# brightest voxel stays more than a tenth above the phantom's peak at every
# size. This is the Gibbs phenomenon: truncating the spectrum of an object
# with discontinuities produces an overshoot at each of them, whatever the
# matrix size. Try multiplying the analytical k-space
# by a Hann window along both axes before the inverse, and see which of the
# two numbers changes.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With bartorch, make BART's 128x128 Shepp-Logan phantom as an image and as
#    8 coil images (bt.phantom, shape (coils, 1, y, x)), combine the channels
#    with bartorch.rss over axis 0, and transform them to k-space with
#    bartorch.fft over the last two axes, unitary. Check the transform against
#    numpy's fft2 between ifftshift and fftshift (norm="ortho"). Reconstruct
#    analytical k-space (bt.phantom(kspace=True)) with the unnormalized
#    bartorch.ifft and report its relative difference from the sampled
#    phantom. Finally run "fft -u 3" through bartorch.cli.main on CFL files
#    and check the result equals bartorch.fft's.

# %%
#
# References
# ----------
#
# .. [#guerquin] Guerquin-Kern M, Lejeune L, Pruessmann KP, Unser M. Realistic analytical
#    phantoms for parallel magnetic resonance imaging. *IEEE Trans Med Imaging*
#    31(3):626-636 (2012). https://doi.org/10.1109/TMI.2011.2174158
