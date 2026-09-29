"""
====================
Tensors and commands
====================

The first lesson of the course: how BART's arrays, commands and Fourier
conventions appear in Python.

bartorch runs BART inside the Python process. Every BART command is a function
of :mod:`bartorch.tools` or of the ``bartorch`` namespace that takes and returns
:class:`torch.Tensor` objects, with no files and no subprocess in between. This
lesson simulates a multichannel acquisition with BART's analytical phantom,
transforms it with BART's FFT, checks the transform against NumPy, and runs the
same command once more from a CFL file through BART's command line.

**Learning objectives**

- Relate a C-order tensor shape to BART's dimension vector.
- Generate an analytical phantom as an image, as coil images and as k-space.
- State the centring and normalization of :func:`bartorch.fft`, and verify them
  against :func:`numpy.fft.fftn`.
- Explain why k-space simulated analytically differs from the DFT of a sampled
  image.
- Call a BART command from Python, and the same command from its command line.

The next lesson, :doc:`02-from-kspace-to-image`, reconstructs an undersampled
acquisition with the functions introduced here.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 11,
        "figure.constrained_layout.use": True,
    }
)


def panels(rows, columns):
    """A grid of square image panels filling the documentation column."""
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(8.0, rows * 8.0 / columns + 0.4)
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
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
# whose first entry varies fastest. A tensor holding the same bytes in C order
# has the reversed shape, so no copy is needed between the two. BART reserves
# its dimension 3 for receive channels, which in a tensor is the fourth axis
# from the end: a two-dimensional multichannel image is
# ``(coils, 1, y, x)``, with the singleton standing for BART's ``z``.
# :doc:`../../guides/user/conventions` tabulates the layouts used throughout.
#
# :func:`bartorch.tools.phantom` is BART's ``phantom`` command. Without
# ``coils`` it returns the Shepp-Logan image; with ``coils`` it returns that
# image multiplied by the sensitivities of BART's analytical head coil.

image = bt.phantom(SIZE)
coil_images = bt.phantom(SIZE, coils=COILS)

print(f"image {tuple(image.shape)}, {image.dtype}")
print(f"coil images {tuple(coil_images.shape)}")
print(f"BART dimensions of the coil images {list(reversed(coil_images.shape))}")

# %%
#
# The root sum of squares over the channel axis combines the coil images into
# a magnitude image. :func:`bartorch.rss` takes the axis as a tensor index, as
# every function of this package does, rather than as BART's bitmask.

combined = bartorch.rss(coil_images, axes=(0,))
print(f"combined {tuple(combined.shape)}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 5)
axes[0, 0].imshow(image.abs(), cmap="gray")
axes[0, 0].set_title("phantom")
for column in range(3):
    axes[0, column + 1].imshow(coil_images[column, 0].abs(), cmap="gray")
    axes[0, column + 1].set_title(f"channel {column}")
axes[0, 4].imshow(combined[0].abs(), cmap="gray")
axes[0, 4].set_title("root sum of squares")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Fourier transform
# -----------------
#
# :func:`bartorch.fft` is BART's centred transform: the zero frequency and the
# image centre are both at index ``n // 2``. It is unnormalized unless
# ``unitary=True``. For an even number of samples the centred transform equals
# NumPy's transform between ``ifftshift`` and ``fftshift``, which gives a
# reference computed outside BART.

kspace = bartorch.fft(coil_images, axes=(-2, -1), unitary=True)

shifted = np.fft.ifftshift(coil_images.numpy(), axes=(-2, -1))
reference = np.fft.fftshift(np.fft.fft2(shifted, norm="ortho"), axes=(-2, -1))

difference = np.abs(kspace.numpy() - reference).max() / np.abs(reference).max()
print(f"largest difference from numpy, relative to the peak: {difference:.1e}")

# %%
#
# The difference is single-precision round-off. The unitary transform
# preserves the norm, so white noise has the same standard deviation in
# k-space and in the image.

print(f"norm ratio image/k-space: {float(coil_images.norm() / kspace.norm()):.6f}")

# %%
#
# Analytical k-space
# ------------------
#
# ``kspace=True`` evaluates the Fourier transform of the phantom's ellipses
# analytically at the k-space sample positions, rather than transforming the
# sampled image. The two are different data: the analytical k-space has
# infinite extent and is truncated at the edge of the matrix, so its inverse
# transform carries Gibbs ringing at every edge, whereas the DFT of the
# sampled image reproduces that image exactly. Simulating data with the same
# discrete model the reconstruction inverts, the *inverse crime* [#guerquin]_,
# removes this discretization error from a simulation; the analytical k-space
# retains it.

analytical = bt.phantom(SIZE, kspace=True)
truncated = bartorch.ifft(analytical, axes=(-2, -1))

error = float((truncated - image).norm() / image.norm())
print(f"relative difference from the sampled phantom: {error:.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 3)
axes[0, 0].imshow(image.abs(), cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title("sampled phantom")
axes[0, 1].imshow(truncated.abs(), cmap="gray", vmin=0, vmax=1)
axes[0, 1].set_title("from analytical k-space")
handle = axes[0, 2].imshow((truncated - image).abs(), cmap="magma", vmin=0, vmax=0.3)
axes[0, 2].set_title("|difference|")
figure.colorbar(handle, ax=axes[0, 2], fraction=0.046)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The difference is largest at the ellipse boundaries, where the truncated
# spectrum rings, and decays with distance from them. The
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
# dimensions 0 and 1, the last two tensor axes. The two routes run the same
# BART function on the same bytes, so the results are identical.
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
