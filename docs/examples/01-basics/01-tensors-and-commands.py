"""
====================
Tensors and commands
====================

This first lesson establishes how MR data from BART appear in Python: how a
multichannel image and its k-space are laid out as tensors, which Fourier
convention BART uses to go from k-space to the image, and how a BART command
is called. Every later lesson relies on these conventions. A reconstruction
that is off by a factor of :math:`\\sqrt{N}`, or by a half-voxel shift from a
misplaced k-space centre, is a consequence of misreading one of them.

bartorch runs BART inside the Python process. Every BART command is a function
of :mod:`bartorch.tools` or of the ``bartorch`` namespace that takes and returns
:class:`torch.Tensor` objects, with no files and no subprocess in between. The
lesson simulates a receive-array acquisition of BART's analytical Shepp-Logan
phantom, transforms it between image space and k-space, checks the transform
against NumPy, and runs the same command once more through BART's command line.

**Learning objectives**

- Relate a C-order tensor shape to BART's dimension vector, and locate the
  receive-channel axis.
- Generate a phantom as an image, as coil images and as analytical k-space.
- State the centring and normalization of :func:`bartorch.fft`, and verify them
  against :func:`numpy.fft.fftn`.
- Explain why analytically simulated k-space, truncated to the acquired
  matrix, reconstructs with Gibbs ringing.
- Call a BART command from Python, and the same command from its command line.

The next lesson, :doc:`02-from-kspace-to-image`, reconstructs an undersampled
acquisition with the functions introduced here.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

WIDTH = 8.0  # inches, the width of the documentation column


def panels(columns, rows=1, width=WIDTH):
    """A row (or grid) of frameless square image panels."""
    side = width / columns
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(width, rows * side + 0.5))
    for axis in axes.flat:
        axis.set_axis_off()
    return figure, axes


# sphinx_gallery_end_ignore
import tempfile
from pathlib import Path

import numpy as np
import torch

import bartorch
import bartorch.tools as bt

SIZE = 128
COILS = 8

# %%
#
# Array layout
# ------------
#
# BART stores an array in Fortran order and describes it by a dimension vector
# whose first entry varies fastest: readout (``x``), then the phase-encoding
# directions (``y``, ``z``), then the receive channels in BART's dimension 3,
# and further dimensions for sets of sensitivity maps, echoes, frames and so
# on. A tensor holding the same bytes in C order has the reversed shape, so no
# copy is needed between the two: a two-dimensional multichannel image is
# ``(coils, 1, y, x)``, with the singleton standing for BART's ``z``, and the
# readout direction is the last tensor axis.
# :doc:`/explanation/data-layout` tabulates the layouts used throughout.
#
# :func:`bartorch.tools.phantom` is BART's ``phantom`` command. Without
# ``coils`` it returns the Shepp-Logan image; with ``coils`` it returns that
# image weighted by the complex receive sensitivities of BART's analytical
# eight-channel head array, one image per channel.

image = bt.phantom(SIZE)
coil_images = bt.phantom(SIZE, coils=COILS)

print(f"image {tuple(image.shape)}, {image.dtype}")
print(f"coil images {tuple(coil_images.shape)}")
print(f"BART dimensions of the coil images {list(reversed(coil_images.shape))}")

# %%
#
# Each channel sees the object through its own sensitivity profile: bright
# near the coil element, dark on the opposite side of the head. The root sum
# of squares (RSS) over the channel axis, :math:`\sqrt{\sum_c |x_c|^2}`, is
# the standard magnitude combination of a receive array; it recovers the
# object with the residual shading of the summed sensitivity magnitudes.
# :func:`bartorch.rss` takes the channel axis as a tensor index, as every
# function of this package does, rather than as BART's bitmask.

combined = bartorch.rss(coil_images, axes=(0,))
print(f"combined {tuple(combined.shape)}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(4)
# The skull is the brightest structure; a window at half the peak shows the
# brain, where the sensitivity shading is read.
peak = 0.5 * float(coil_images.abs().max())
for column, channel in enumerate((0, 2, 4, 6)):
    axes[0, column].imshow(coil_images[channel, 0].abs(), vmin=0, vmax=peak)
    axes[0, column].set_title(f"channel {channel}")
figure.suptitle("single-channel magnitude images")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Fourier transform
# -----------------
#
# The measured signal of a channel is the Fourier transform of its coil image,
# sampled on the k-space grid. :func:`bartorch.fft` is BART's *centred*
# discrete transform: the k-space centre (DC) and the image centre are both at
# index ``n // 2``, which is how k-space is displayed and how scanners
# deliver it. The transform is unnormalized unless ``unitary=True``. For an
# even matrix size the centred transform equals NumPy's transform between
# ``ifftshift`` and ``fftshift``, which gives a reference computed outside
# BART.

kspace = bartorch.fft(coil_images, axes=(-2, -1), unitary=True)

shifted = np.fft.ifftshift(coil_images.numpy(), axes=(-2, -1))
reference = np.fft.fftshift(np.fft.fft2(shifted, norm="ortho"), axes=(-2, -1))

difference = np.abs(kspace.numpy() - reference).max() / np.abs(reference).max()
print(f"largest difference from numpy, relative to the peak: {difference:.1e}")

# %%
#
# The difference is single-precision round-off. The unitary transform
# preserves the :math:`\ell_2` norm (Parseval's theorem), so white noise has
# the same standard deviation in k-space and in the image, a property the
# noise and SNR lessons rely on.

print(f"norm ratio image/k-space: {float(coil_images.norm() / kspace.norm()):.6f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, width=6.4)
axes[0, 0].imshow(combined[0].abs(), vmin=0, vmax=float(combined.abs().max()))
axes[0, 0].set_title("root sum of squares")
log_k = torch.log10(kspace[0, 0].abs() / kspace.abs().max()).clamp(min=-5)
handle = axes[0, 1].imshow(log_k, cmap="magma", vmin=-5, vmax=0)
axes[0, 1].set_title("k-space of channel 0")
figure.colorbar(handle, ax=axes[0, 1], fraction=0.046, label="$\\log_{10}$ (|signal| / peak)")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The k-space magnitude, drawn on a logarithmic scale, spans five orders of
# magnitude: the signal energy is concentrated at the centre, which sets the
# image contrast, while the periphery carries the edges and fine detail. This
# distribution is what variable-density undersampling exploits in the next
# lessons.
#
# Analytical k-space
# ------------------
#
# ``kspace=True`` evaluates the Fourier transform of the phantom's ellipses
# analytically at the k-space sample positions, rather than transforming the
# sampled image. The two are different data. The continuous object has
# infinite spatial-frequency extent, and an acquisition of a finite matrix
# truncates it at :math:`\pm k_{\max}`; its inverse transform therefore
# carries Gibbs ringing, the oscillation next to every sharp edge that
# truncation artefacts produce on a scanner. The DFT of the sampled image,
# by contrast, reproduces that image exactly. Simulating data with the same
# discrete model the reconstruction inverts, the *inverse crime*
# [#guerquin]_, removes this discretization error from a simulation;
# analytical k-space retains it and is therefore the more realistic test
# data.

analytical = bt.phantom(SIZE, kspace=True)
truncated = bartorch.ifft(analytical, axes=(-2, -1))

error = float((truncated - image).norm() / image.norm())
print(f"relative difference from the sampled phantom: {error:.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(3)
axes[0, 0].imshow(image.abs(), vmin=0, vmax=1)
axes[0, 0].set_title("sampled phantom")
axes[0, 1].imshow(truncated.abs(), vmin=0, vmax=1)
axes[0, 1].set_title("from analytical k-space")
handle = axes[0, 2].imshow((truncated - image).abs(), cmap="magma", vmin=0, vmax=0.3)
axes[0, 2].set_title("|difference|")
figure.colorbar(handle, ax=axes[0, 2], fraction=0.046, label="fraction of peak")

row = SIZE // 2
figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), width_ratios=(2, 1))
for axis, limits in zip(axes, ((0, SIZE - 1), (4, 30))):
    axis.plot(image[row].abs(), color="C0", lw=1.8, label="sampled phantom")
    axis.plot(truncated[row].abs(), color="C1", lw=1.2, label="from analytical k-space")
    axis.set_xlim(*limits)
    axis.set_xlabel("x [voxel]")
axes[0].set_ylabel("magnitude")
axes[0].set_title(f"profile along row {row}")
axes[1].set_title("skull edge, enlarged")
axes[0].legend(loc="upper center", ncols=2)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The difference map is largest at the ellipse boundaries and decays with
# distance from them. The profile through the centre of the phantom shows the
# ringing directly: the image from analytical k-space overshoots at every
# intensity step and oscillates with a period of about two voxels next to it. The
# unnormalized inverse transform is used here because ``phantom`` scales its
# k-space so that the unnormalized inverse returns the image's intensities.
#
# Commands and the command line
# -----------------------------
#
# A function of :mod:`bartorch.tools` passes BART's options as keyword
# arguments, under their long names or their single-letter flags, and reverses
# the dimension vector of each tensor it hands over. The ``bartorch`` console
# command takes BART's own command lines and operates on CFL files, so a script
# written for the ``bart`` executable runs against the same library;
# :func:`bartorch.cli.main` runs one such command line in the current process.
# :func:`bartorch.io.writecfl` takes an array in BART's order, which the
# transpose of a C-order array is.

with tempfile.TemporaryDirectory() as directory:
    source, target = str(Path(directory) / "coils"), str(Path(directory) / "kspace")
    bartorch.io.writecfl(source, coil_images.numpy().T)
    status = bartorch.cli.main(["fft", "-u", "3", source, target])
    from_files = torch.as_tensor(bartorch.cli.read(target))

print(f"exit status {status}, identical to bartorch.fft: {torch.equal(from_files, kspace)}")

# %%
#
# ``-u`` requests the unitary transform and ``3`` is BART's bitmask for its
# dimensions 0 and 1 (readout and first phase-encoding direction), the last
# two tensor axes. The two routes run the same BART function on the same
# bytes, so the results are identical.
#
# A BART command is not recorded by autograd: its result has no gradient with
# respect to its inputs. The operators of :mod:`bartorch.linop` and the solvers
# of :mod:`bartorch.optim`, introduced from
# :doc:`../03-regularization/02-operators-and-solvers` on, are differentiable.
# :doc:`../../explanation/execution-model` describes both routes.

# %%
#
# References
# ----------
#
# .. [#guerquin] Guerquin-Kern M, Lejeune L, Pruessmann KP, Unser M. Realistic analytical
#    phantoms for parallel magnetic resonance imaging. *IEEE Trans Med Imaging*
#    31(3):626-636 (2012). https://doi.org/10.1109/TMI.2011.2174158
