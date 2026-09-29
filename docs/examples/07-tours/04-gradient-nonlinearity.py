"""
=====================
Gradient nonlinearity
=====================

The geometric distortion a gradient coil's nonlinearity produces over a large
field of view, simulated from a spherical-harmonic description of the coil and
corrected with :class:`bartorch.tools.Gradunwarp`.

The spatial encoding assumes gradient fields that vary linearly with
position. The field of a real coil departs from linearity with distance from
isocentre, so a voxel is encoded at a position displaced from its true one;
the displacement is a property of the coil, stated by its manufacturer as the
coefficients of a spherical-harmonic expansion [#janke]_. The correction
evaluates that expansion at every voxel of the image and resamples the image
at the displaced positions.

The coefficients here describe a generic coil with third-order terms only,
defined in the code; no manufacturer's table is used.
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
import numpy as np
import torch
from scipy import ndimage

import bartorch.tools as bt

# %%
#
# The coil
# --------
#
# :class:`~bartorch.tools.GradientCoefficients` holds the cosine and
# sine coefficients :math:`\alpha_{nm}` and :math:`\beta_{nm}` of the
# departure of each gradient's field from linearity, in the Siemens
# convention: harmonics normalized over a reference radius :math:`R_0`, and
# positions in scanner coordinates. An all-zero table is a linear coil.
#
# Each gradient field is odd along its own axis, so its lowest-order
# nonlinear terms are of third order: :math:`\alpha_{31}` for the x gradient,
# :math:`\beta_{31}` for the y gradient and :math:`\alpha_{30}` for the z
# gradient. Each is given the same magnitude here.

ORDER = 3
alpha = np.zeros((3, ORDER + 1, ORDER + 1))
beta = np.zeros_like(alpha)
alpha[0, 3, 1] = -0.03  # x gradient
beta[1, 3, 1] = -0.03  # y gradient
alpha[2, 3, 0] = -0.03  # z gradient
coil = bt.GradientCoefficients(
    basis="normalized", alpha=alpha, beta=beta, reference_radius_mm=250.0
)

# %%
#
# An axial slice through isocentre, 256 voxels over a 400 mm field of view.
# :class:`~bartorch.tools.Gradunwarp` evaluates the expansion at every
# voxel of the corrected grid; its ``source_grid`` is the index into the
# acquired image at which each corrected voxel was encoded.

SIZE, FOV_MM = 256, 400.0
unwarp = bt.Gradunwarp(coil, shape=(SIZE, SIZE), fov_mm=(FOV_MM, FOV_MM))

index = np.stack(np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij"), axis=-1)
displacement = unwarp.source_grid - index  # voxels
distance_mm = np.linalg.norm(displacement, axis=-1) * FOV_MM / SIZE

radius_mm = np.linalg.norm(index - (SIZE - 1) / 2, axis=-1) * FOV_MM / SIZE
for r in (100, 150, 200):
    ring = np.abs(radius_mm - r) < 1.0
    print(f"displacement at {r} mm from isocentre: {distance_mm[ring].mean():.2f} mm")

# %%
#
# The acquisition
# ---------------
#
# The object is a disc of 360 mm diameter carrying a lattice of lines 25 mm
# apart, defined in closed form so it can be evaluated at any position. The
# acquired image at index :math:`p` holds the object at the position
# :math:`r` encoded there, the solution of :math:`r + d(r) = p`, found by
# fixed-point iteration. Its intensity is divided by the Jacobian determinant
# of the mapping, since a voxel whose volume the nonlinearity enlarges
# collects the signal of a larger region.


def lattice(position):
    """Disc with a lattice of lines, at positions in voxel units."""
    y, x = position[..., 0] - (SIZE - 1) / 2, position[..., 1] - (SIZE - 1) / 2
    spacing = 25.0 * SIZE / FOV_MM
    lines = np.maximum(
        np.exp(-0.5 * (((y % spacing) - spacing / 2) / 0.8) ** 2),
        np.exp(-0.5 * (((x % spacing) - spacing / 2) / 0.8) ** 2),
    )
    disc = 1.0 / (1.0 + np.exp((np.hypot(x, y) - 180.0 * SIZE / FOV_MM) / 0.8))
    return disc * (0.3 + 0.7 * lines)


def at(values, position):
    return ndimage.map_coordinates(values, np.moveaxis(position, -1, 0), order=3, mode="nearest")


encoded = index.astype(float)
for _ in range(20):
    encoded = index - np.stack([at(displacement[..., c], encoded) for c in range(2)], axis=-1)

acquired = lattice(encoded) / at(unwarp.jacobian_grid, encoded)
truth = lattice(index.astype(float))

# %%
#
# Correction
# ----------
#
# The correction resamples the acquired image at the source grid by cubic
# B-spline interpolation and multiplies it by the Jacobian determinant, which
# restores the intensity. ``jacobian=False`` corrects the geometry only.

corrected = unwarp(torch.as_tensor(acquired, dtype=torch.float32)).numpy()
geometry_only = bt.Gradunwarp(coil, shape=(SIZE, SIZE), fov_mm=(FOV_MM, FOV_MM), jacobian=False)(
    torch.as_tensor(acquired, dtype=torch.float32)
).numpy()


def error(image):
    return float(np.linalg.norm(image - truth) / np.linalg.norm(truth))


print(f"acquired       NRMSE {error(acquired):.3f}")
print(f"geometry only  NRMSE {error(geometry_only):.3f}")
print(f"corrected      NRMSE {error(corrected):.4f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 4, figsize=(8.0, 2.4), width_ratios=(1, 1, 1, 1.25))
for axis, values, title in (
    (axes[0], acquired, "acquired"),
    (axes[1], corrected, "corrected"),
    (axes[2], acquired - truth, "acquired - object"),
):
    if title.endswith("object"):
        axis.imshow(values, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    else:
        axis.imshow(values, cmap="gray", vmin=0, vmax=1)
    axis.set_title(title)
handle = axes[3].imshow(distance_mm, cmap="magma")
axes[3].set_title("displacement")
figure.colorbar(handle, ax=axes[3], fraction=0.046, label="mm")
for axis in axes:
    axis.set_xticks([])
    axis.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The displacement grows with the cube of the distance from isocentre, so the
# centre of the field of view is unaffected and the periphery is displaced by
# several millimetres. The residual after correction is the error of
# interpolating the acquired image; the geometry-only correction leaves the
# intensity error of the Jacobian, largest where the displacement varies
# fastest.
#
# A manufacturer's table is read with
# :meth:`~bartorch.tools.GradientCoefficients.from_file`, and
# :meth:`~bartorch.tools.Gradunwarp.from_mrd` and
# :meth:`~bartorch.tools.Gradunwarp.from_affine` take the geometry of
# the image from an MRD header or an affine.
#
# References
# ----------
#
# .. [#janke] Janke A, Zhao H, Cowin GJ, Galloway GJ, Doddrell DM. Use of
#    spherical harmonic deconvolution methods to compensate for nonlinear
#    gradient effects on MRI images. *Magn Reson Med* 52(1):115-122 (2004).
#    https://doi.org/10.1002/mrm.20122
