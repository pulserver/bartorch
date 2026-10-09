"""
=====================
Gradient nonlinearity
=====================

Spatial encoding assumes that every gradient field changes linearly with
position. A real gradient coil departs from that more and more with the
distance from isocentre, so each spin is encoded at a position displaced from
its true one. The image is warped, by a few millimetres at the edge of a
head-sized field of view and by centimetres at the edge of a body-sized one,
and the voxel volume changes with the warp. The displacement is a property of
the coil: the manufacturer states it as the coefficients of a
spherical-harmonic expansion of each gradient field [#janke]_. To correct it
you evaluate that expansion at every voxel and resample the image at the
positions where the voxels were actually encoded.

In this tour you warp a grid phantom in a coronal slice over a 450 mm field
of view with a coil described by third-order harmonics, and correct it with
:class:`bartorch.tools.Gradunwarp`. The coil is a generic one, defined in the
code; no manufacturer's table is used.

Prerequisites: lesson 1, :doc:`../01-basics/01-tensors-and-commands`.

**Learning objectives**

- Describe a gradient coil's nonlinearity by its spherical-harmonic
  coefficients with :class:`~bartorch.tools.GradientCoefficients`.
- Relate the displacement to the distance from isocentre and to the gradient
  axis.
- Correct the geometry and the intensity of an image with
  :class:`~bartorch.tools.Gradunwarp`, and tell the two apart.
- Place an image in scanner coordinates by its orientation and field of
  view.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage


def show(axis, values, title, vmin=0.0, vmax=1.0, cmap="gray"):
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_axis_off()
    return handle


def displacement_at(displacement_mm, offset_mm, r):
    """Mean |displacement| at ``r`` mm from isocentre on the z and x axes."""
    along_z = np.abs(offset_mm[..., 0]) - r
    along_x = np.abs(offset_mm[..., 1]) - r
    on_z = (np.abs(along_z) < VOXEL_MM / 2) & (np.abs(offset_mm[..., 1]) < VOXEL_MM)
    on_x = (np.abs(along_x) < VOXEL_MM / 2) & (np.abs(offset_mm[..., 0]) < VOXEL_MM)
    return (
        np.abs(displacement_mm[on_z, 0]).mean(),
        np.abs(displacement_mm[on_x, 1]).mean(),
    )


# sphinx_gallery_end_ignore
import numpy as np
import torch

import bartorch.tools as bt

# %%
# The coil
# --------
#
# :class:`~bartorch.tools.GradientCoefficients` holds the cosine and sine
# coefficients :math:`\alpha_{nm}` and :math:`\beta_{nm}` of each gradient
# field's departure from linearity, in the Siemens convention: harmonics
# normalized over a reference radius :math:`R_0`, and positions in scanner
# coordinates, with :math:`z` along the bore. An all-zero table is a perfectly
# linear coil.
#
# Each gradient field is odd along its own axis, so its lowest-order
# nonlinear terms are of third order: :math:`\alpha_{31}` for the :math:`x`
# gradient, :math:`\beta_{31}` for the :math:`y` gradient and
# :math:`\alpha_{30}` for the :math:`z` gradient, whose harmonic is
# proportional to :math:`z\,(2z^2 - 3x^2 - 3y^2)`. The signs and sizes of the
# terms decide where the image is compressed and where it is stretched. On a
# short whole-body coil the :math:`z` gradient is usually the least linear,
# so you give it the largest term:
ORDER = 3
alpha = np.zeros((3, ORDER + 1, ORDER + 1))
beta = np.zeros_like(alpha)
alpha[0, 3, 1] = -0.04  # x gradient
beta[1, 3, 1] = -0.04  # y gradient
alpha[2, 3, 0] = -0.06  # z gradient
coil = bt.GradientCoefficients(
    basis="normalized", alpha=alpha, beta=beta, reference_radius_mm=250.0
)

# %%
# The slice
# ---------
#
# The correction needs to know where each voxel sits in the scanner. Your
# slice is coronal through isocentre, 256 x 256 over a 450 mm field of view
# (1.8 mm voxels): rows run from superior to inferior along :math:`-z`, and
# columns from right to left along :math:`x`. The orientation matrix says, for
# each array axis, along which scanner direction it increases: its columns
# are the row axis and the column axis, its rows are :math:`x`, :math:`y` and
# :math:`z`.
SIZE, FOV_MM = 256, 450.0
VOXEL_MM = FOV_MM / SIZE
coronal = np.array([[0.0, 1.0], [0.0, 0.0], [-1.0, 0.0]])
unwarp = bt.Gradunwarp(
    coil,
    shape=(SIZE, SIZE),
    fov_mm=(FOV_MM, FOV_MM),
    orientation=coronal,
)

# %%
# :class:`~bartorch.tools.Gradunwarp` evaluates the expansion at every voxel
# of the corrected grid. Its ``source_grid`` is the index into the acquired
# image at which each voxel was encoded, so the displacement is the
# difference from the voxel's own index:
grid = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
index = np.stack(grid, axis=-1)
displacement_mm = (unwarp.source_grid - index) * VOXEL_MM
offset_mm = (index - (SIZE - 1) / 2) * VOXEL_MM

# sphinx_gallery_start_ignore
for r in (100, 150, 200):
    along_z, along_x = displacement_at(displacement_mm, offset_mm, r)
    print(
        f"{r} mm from isocentre: displacement {along_z:4.1f} mm along z, {along_x:4.1f} mm along x"
    )
# sphinx_gallery_end_ignore

# %%
# The displacement grows with the cube of the distance from isocentre: under
# a millimetre at 100 mm, several at 200 mm, and about twice as large along
# :math:`z` as along :math:`x`, as the coefficients say it should be.
#
# The acquisition
# ---------------
#
# The object is a grid phantom: a disc 420 mm across carrying lines 30 mm
# apart, written in closed form so that you can evaluate it at any position.
# The acquired image at index :math:`p` holds the object at the position
# :math:`r` encoded there, the solution of :math:`r + d(r) = p`, found by
# fixed-point iteration. Its intensity is divided by the Jacobian
# determinant of the mapping, because a voxel whose volume the nonlinearity
# enlarges collects the signal of a larger region. That code simulates the
# artefact rather than correcting it, so it is hidden; ``acquired`` is the
# warped image and ``truth`` the object on the true grid.

# sphinx_gallery_start_ignore


def grid_phantom(position):
    """Disc with a grid of lines, at positions in voxel units."""
    y, x = position[..., 0] - (SIZE - 1) / 2, position[..., 1] - (SIZE - 1) / 2
    spacing = 30.0 / VOXEL_MM
    lines = np.maximum(
        np.exp(-0.5 * (((y % spacing) - spacing / 2) / 0.8) ** 2),
        np.exp(-0.5 * (((x % spacing) - spacing / 2) / 0.8) ** 2),
    )
    disc = 1.0 / (1.0 + np.exp((np.hypot(x, y) - 210.0 / VOXEL_MM) / 0.8))
    return disc * (0.3 + 0.7 * lines)


def at(values, position):
    return ndimage.map_coordinates(values, np.moveaxis(position, -1, 0), order=3, mode="nearest")


shift = unwarp.source_grid - index  # voxels
encoded = index.astype(float)
for _ in range(30):
    encoded = index - np.stack([at(shift[..., c], encoded) for c in range(2)], axis=-1)

acquired = grid_phantom(encoded) / at(unwarp.jacobian_grid, encoded)
truth = grid_phantom(index.astype(float))

ZOOM = (slice(8, 88), slice(8, 88))
grid_lines = grid_phantom(index.astype(float)) > 0.65


def boxed(axis):
    axis.add_patch(
        Rectangle(
            (ZOOM[1].start, ZOOM[0].start),
            ZOOM[1].stop - ZOOM[1].start,
            ZOOM[0].stop - ZOOM[0].start,
            fill=False,
            edgecolor="C0",
            linewidth=1.2,
        )
    )


nan = np.nan
inside = truth > 0.1
figure, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), width_ratios=(1.15, 1.2, 1.2))
show(axes[0], acquired, "acquired")
boxed(axes[0])
handle = show(
    axes[1],
    np.where(inside, np.linalg.norm(displacement_mm, axis=-1), nan),
    "displacement",
    0,
    10,
    "magma",
)
figure.colorbar(handle, ax=axes[1], fraction=0.046, label="mm")
handle = show(
    axes[2],
    np.where(inside, unwarp.jacobian_grid, nan),
    "Jacobian determinant",
    0.7,
    1.3,
    "RdBu_r",
)
figure.colorbar(handle, ax=axes[2], fraction=0.046)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The centre of the field of view is untouched, and the displacement rises
# towards the edge, fastest at the top and bottom, along :math:`z`. There the
# Jacobian determinant is below one: the image is compressed towards
# isocentre, each voxel collects the signal of a larger region, and the
# periphery comes out brighter. Along :math:`x` the image is stretched and
# darker. Off the axes the :math:`z` displacement follows its harmonic and
# changes sign where :math:`2z^2 = 3x^2`, which is the cross you see in the
# Jacobian determinant.
#
# The correction
# --------------
#
# Calling the :class:`~bartorch.tools.Gradunwarp` on an image resamples it at
# the source grid by cubic B-spline interpolation and multiplies it by the
# Jacobian determinant, which restores the intensity. With ``jacobian=False``
# it corrects the geometry only, which lets you see the two parts separately:
image = torch.as_tensor(acquired, dtype=torch.float32)
corrected = unwarp(image).numpy()

geometry = bt.Gradunwarp(
    coil,
    shape=(SIZE, SIZE),
    fov_mm=(FOV_MM, FOV_MM),
    orientation=coronal,
    jacobian=False,
)
geometry_only = geometry(image).numpy()

# sphinx_gallery_start_ignore


def nrmse(image):
    return float(np.linalg.norm(image - truth) / np.linalg.norm(truth))


print(f"acquired       NRMSE {nrmse(acquired):.3f}")
print(f"geometry only  NRMSE {nrmse(geometry_only):.3f}")
print(f"corrected      NRMSE {nrmse(corrected):.3f}")

figure, axes = plt.subplots(1, 2, figsize=(7.8, 4.1))
for axis, values, title in ((axes[0], acquired, "acquired"), (axes[1], corrected, "corrected")):
    show(axis, values, title)
    axis.contour(grid_lines, levels=[0.5], colors="C1", linewidths=0.9)
    boxed(axis)
figure.suptitle("orange: grid of the object")
plt.show()

figure, axes = plt.subplots(2, 2, figsize=(7.0, 7.4))
for axis, values, title in zip(
    axes.flat,
    (truth, acquired, geometry_only, corrected),
    ("object", "acquired", "geometry only", "corrected"),
):
    show(axis, values[ZOOM], title)
figure.suptitle("superior right corner (blue box)")
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
for axis, values, title in zip(axes, (geometry_only, corrected), ("geometry only", "corrected")):
    handle = show(
        axis,
        np.where(inside, values - truth, nan),
        f"{title} - object",
        -0.3,
        0.3,
        "RdBu_r",
    )
figure.colorbar(handle, ax=axes, shrink=0.9, label="difference")
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), sharey=True)
centre = SIZE // 2
for axis, profile, position, label in (
    (axes[0], lambda v: v[:centre, centre], -offset_mm[:centre, 0, 0], "z [mm], superior"),
    (axes[1], lambda v: v[centre, centre:], offset_mm[0, centre:, 1], "x [mm]"),
):
    for values, title, style, width in (
        (truth, "object", "-", 3.0),
        (acquired, "acquired", "-", 1.2),
        (corrected, "corrected", "--", 1.2),
    ):
        axis.plot(position, profile(values), style, label=title, linewidth=width)
    axis.set_xlim(0, 225)
    axis.set_xlabel(label)
axes[0].set_ylabel("signal")
axes[0].set_title("along z")
axes[1].set_title("along x")
figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
plt.show()
# sphinx_gallery_end_ignore

# %%
# In the acquired image the grid lines bend away from the grid of the object
# (orange) towards the edge; after correction they lie on it. In the enlarged
# corner the geometry-only correction already puts the lines back in place,
# but leaves the intensity error of the Jacobian: its difference from the
# object is positive (red) at the top and bottom and negative (blue) at the
# sides. The full correction
# removes both, and the NRMSE against the object drops from the acquired
# image's to a few per cent. What is left is the error of interpolating the
# acquired image, largest on the thin lines where compression has
# undersampled them. The profiles show the same thing on the two axes: the
# acquired lines move inwards and brighten along :math:`z`, move outwards
# along :math:`x`, and the corrected lines sit on the object's.
#
# For a real coil, :meth:`~bartorch.tools.GradientCoefficients.from_file`
# reads the manufacturer's table, and
# :meth:`~bartorch.tools.Gradunwarp.from_mrd` and
# :meth:`~bartorch.tools.Gradunwarp.from_affine` take the geometry of the
# image from an MRD header or an affine. Scanners apply a two-dimensional
# correction in the plane of the slice by default; the through-plane
# displacement moves signal between slices, and correcting it needs the whole
# volume.
#
# Your turn: a less linear coil
# -----------------------------
#
# The displacement is linear in the coefficients. Scale the :math:`z`
# gradient's term and see how far a voxel 200 mm above isocentre moves:
for term in (-0.03, -0.06, -0.12):
    stronger = alpha.copy()
    stronger[2, 3, 0] = term
    trial = bt.Gradunwarp(
        bt.GradientCoefficients(
            basis="normalized",
            alpha=stronger,
            beta=beta,
            reference_radius_mm=250.0,
        ),
        shape=(SIZE, SIZE),
        fov_mm=(FOV_MM, FOV_MM),
        orientation=coronal,
    )
    moved_mm = (trial.source_grid - index) * VOXEL_MM
    along_z, _ = displacement_at(moved_mm, offset_mm, 200)
    print(f"alpha_30 = {term:+.2f}: {along_z:4.1f} mm at z = 200 mm")

# %%
# Doubling the coefficient doubles the displacement. Try a positive
# sign as well, and look at which way the grid lines move.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Correct the gradient-nonlinearity distortion of a 2D coronal slice
#    (256 x 256, 450 mm field of view, rows superior to inferior along -z,
#    columns along +x) with bt.Gradunwarp. Describe the coil with
#    bt.GradientCoefficients (normalized basis, R0 250 mm, third-order
#    alpha_31 for x, beta_31 for y, alpha_30 for z). Apply the full
#    correction and a geometry-only one (jacobian=False), report the NRMSE of
#    each against the object, and the displacement in mm at 100, 150 and
#    200 mm from isocentre along z and x.

# %%
#
# References
# ----------
#
# .. [#janke] Janke A, Zhao H, Cowin GJ, Galloway GJ, Doddrell DM. Use of
#    spherical harmonic deconvolution methods to compensate for nonlinear
#    gradient effects on MRI images. *Magn Reson Med* 52(1):115-122 (2004).
#    https://doi.org/10.1002/mrm.20122
