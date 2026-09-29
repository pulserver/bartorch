"""
===================================
Susceptibility distortion in EPI
===================================

In an echo-planar image, the phase-encoding direction is sampled at the echo
spacing rather than the dwell time, so its bandwidth per pixel is a few tens of
hertz. A spin off resonance by :math:`\\Delta f` is displaced along the
phase-encoding axis by :math:`\\Delta f` divided by that bandwidth, which near
the frontal sinus and the petrous bone amounts to several voxels at 3 T: the
orbitofrontal cortex is compressed or stretched, and signal piles up where
neighbouring voxels are displaced onto the same location.

The displacement changes sign with the direction in which k-space is
traversed. Two acquisitions with opposite phase-encoding polarity,
*blip-up* and *blip-down*, are distorted in opposite directions, and the
displacement field that brings them into register is the correction
[#andersson]_. This example simulates such a pair of a head at 3 T in a
:math:`B_0` field computed from the magnetic susceptibility of the head, and
corrects it with :func:`bartorch.tools.correct_susceptibility`, which runs
PyHySCO [#pyhysco]_.

**Learning objectives**

* Compute the displacement of an EPI voxel from its off-resonance frequency,
  the echo spacing and the number of phase-encoding lines.
* Recognise the compression, stretching and signal pile-up of susceptibility
  distortion, and their reversal between the two phase-encoding polarities.
* Estimate the displacement field from a reversed phase-encoding pair and
  correct both images, including their intensity.
* Assess the estimated displacement against the field map it originates
  from.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
import numpy as np
from brainweb_dl import get_mri
from matplotlib.patches import Rectangle
from scipy import ndimage

HZ_PER_PPM_3T = 127.74  # 42.577 MHz/T x 3 T x 1e-6


def brainweb_slab(size, fov_mm, slices_mm):
    """T2-weighted BrainWeb axial slices, their 3 T field map in Hz, and head and brain masks.

    The field is the dipole field of the head's susceptibility distribution
    (air 9.4 ppm above tissue), computed in 3D on a 2 mm grid with B0 along
    the inferior-superior axis, less a second-order shim fitted over the brain.
    """
    fuzzy = get_mri(sub_id=0, contrast="fuzzy")
    coarse = fuzzy[::2, ::2, ::2]
    chi = 9.4 * coarse[..., 0]
    shape = [2 * n for n in chi.shape]
    k = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    k2 = sum(c**2 for c in k)
    k2[0, 0, 0] = 1.0
    kernel = 1 / 3 - k[0] ** 2 / k2
    kernel[0, 0, 0] = 0.0
    inside = tuple(slice(0, n) for n in chi.shape)
    padded = np.zeros(shape)
    padded[inside] = chi
    ppm = np.real(np.fft.ifftn(kernel * np.fft.fftn(padded)))[inside]
    z, y, x = np.meshgrid(*[np.arange(n) - n / 2 for n in chi.shape], indexing="ij")
    shim = [np.ones_like(x), x, y, z, x * y, x * z, y * z, x**2 - y**2, 2 * z**2 - x**2 - y**2]
    brain3d = coarse[..., 1:4].sum(-1) > 0.5
    design = np.stack([term[brain3d] for term in shim], 1)
    coefficients = np.linalg.lstsq(design, ppm[brain3d], rcond=None)[0]
    ppm = ppm - sum(c * term for c, term in zip(coefficients, shim, strict=True))

    t2 = get_mri(sub_id=0, contrast="T2").astype(np.float32)
    t2 /= t2.max()
    side = int(round(fov_mm))  # BrainWeb is at 1 mm
    zoom = size / side
    layers = []
    for s in slices_mm:
        rows, cols = t2[s].shape
        head = ndimage.binary_fill_holes(ndimage.binary_closing(t2[s] > 0.03, iterations=3))
        stack = np.stack(
            [
                t2[s],
                ndimage.zoom(ppm[s // 2], 2, order=1)[:rows, :cols] * HZ_PER_PPM_3T,
                head.astype(np.float32),
                (fuzzy[s, ..., 1:4].sum(-1) > 0.5).astype(np.float32),
            ]
        )
        canvas = np.zeros((4, side, side), dtype=np.float32)
        top, left = (side - rows) // 2, (side - cols) // 2
        canvas[:, top : top + rows, left : left + cols] = stack
        canvas = np.flip(canvas, axis=1)  # anterior at the top
        layers.append(
            [ndimage.zoom(layer, zoom, order=1) for layer in canvas[:2]]
            + [ndimage.zoom(layer, zoom, order=0) > 0.5 for layer in canvas[2:]]
        )
    image, field, head, brain = (np.stack(part) for part in zip(*layers, strict=True))
    field = np.where(head, np.round(field.clip(-150, 150)), 0.0)
    return image.clip(0), field, head, brain & head


def show(axis, values, title, vmin=0.0, vmax=1.0, cmap="gray"):
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_axis_off()
    return handle


# sphinx_gallery_end_ignore
import torch

import bartorch
import bartorch.tools as bt

# %%
#
# Object and field map
# --------------------
#
# The object is a slab of eleven axial slices of the BrainWeb T2-weighted head
# [#brainweb]_, 3 mm apart, through the orbitofrontal cortex and the temporal
# lobes, on a 96 x 96 matrix over a 220 mm field of view (2.3 mm in-plane).
# T2 weighting stands in for the :math:`b = 0` image of a diffusion
# acquisition. The field map is the :math:`B_0` offset at 3 T produced by the susceptibility
# difference between air and tissue, :math:`\Delta\chi = 9.4` ppm, computed in
# 3D with the dipole kernel and less a second-order shim fitted over the brain,
# rounded to 1 Hz and limited to :math:`\pm 150` Hz.

SIZE, FOV_MM, SLICE_MM = 96, 220.0, 3.0
VOXEL_MM = (SLICE_MM, FOV_MM / SIZE, FOV_MM / SIZE)
# sphinx_gallery_start_ignore
image, field_map, head, brain = brainweb_slab(SIZE, FOV_MM, range(37, 70, 3))
image, field_map = torch.as_tensor(image), torch.as_tensor(field_map, dtype=torch.float32)
head, brain = torch.as_tensor(head), torch.as_tensor(brain)
# sphinx_gallery_end_ignore
print(f"slab {tuple(image.shape)}, voxel {VOXEL_MM[1]:.1f} x {VOXEL_MM[2]:.1f} x {SLICE_MM} mm")
for name, region in (("head", head), ("brain", brain)):
    values = field_map[region]
    print(f"field over the {name}: {float(values.min()):+.0f} to {float(values.max()):+.0f} Hz")

# %%
#
# Displacement along the phase-encoding axis
# ------------------------------------------
#
# The phase-encoding axis is anterior-posterior and the readout is
# left-right. With :math:`N` phase-encoding lines acquired at echo spacing
# :math:`\Delta t_{esp}`, line :math:`n` is sampled at
# :math:`t_n = n\, \Delta t_{esp}` relative to the echo time, with
# :math:`k_y = n` in grid units from :math:`-N/2` to :math:`N/2 - 1`. The
# phase accrued off resonance, :math:`2\pi \Delta f\, t_n`, is then linear in
# :math:`k_y`, which is a displacement by
#
# .. math::
#
#    d = \frac{\Delta f}{\mathrm{BW}_{PE}}\ \text{voxels}, \qquad
#    \mathrm{BW}_{PE} = \frac{1}{N\, \Delta t_{esp}},
#
# with :math:`\mathrm{BW}_{PE}` the bandwidth per pixel along the
# phase-encoding axis. Reversing the phase-encoding blips traverses
# :math:`k_y` in the opposite order, :math:`t_n = -n\, \Delta t_{esp}`, and
# reverses the displacement. The readout, sampled at a dwell time of a few
# microseconds, has a bandwidth per pixel over a kilohertz and its
# displacement is neglected. An echo spacing of 0.6 ms without parallel
# imaging is typical of a single-shot diffusion acquisition at this
# resolution.

ECHO_SPACING_S = 0.6e-3
bandwidth_pe = 1 / (SIZE * ECHO_SPACING_S)
displacement_mm = field_map / bandwidth_pe * VOXEL_MM[1]
print(f"bandwidth per pixel along phase encoding: {bandwidth_pe:.1f} Hz")
print(
    f"displacement over the brain: {float(displacement_mm[brain].min()):+.1f} to "
    f"{float(displacement_mm[brain].max()):+.1f} mm"
)

# %%
#
# The acquisition
# ---------------
#
# Each column of each slice is simulated exactly for the field map: the
# phase-encoding line :math:`n` is the Fourier sum over the column,
# :math:`s_n = \sum_y x(y)\, e^{-2\pi i k_n y / N}\, e^{2\pi i \Delta f(y)\,
# t_n}`, and the image is its inverse transform along the phase-encoding axis
# with :func:`bartorch.ifft`. The blip-up image is the one acquired with
# :math:`t_n = -n\, \Delta t_{esp}`, whose voxels are displaced by
# :math:`+d`, towards posterior for a positive offset.

k_y = torch.arange(SIZE, dtype=torch.float64) - SIZE // 2
encoding = torch.exp(-2j * torch.pi * torch.outer(k_y, k_y) / SIZE) / SIZE**0.5  # (n, y)


def acquire(polarity):
    line_time = polarity * k_y * ECHO_SPACING_S
    accrued = torch.polar(
        torch.ones((), dtype=torch.float64),
        2 * torch.pi * line_time[:, None, None, None] * field_map.double(),
    )
    lines = torch.einsum("ny,nzyx,zyx->znx", encoding, accrued, image.to(torch.complex128))
    return bartorch.ifft(lines.to(torch.complex64), axes=1, unitary=True).abs()


blip_up, blip_down = acquire(-1), acquire(+1)

# %%
#
# Correction from the reversed pair
# ---------------------------------
#
# PyHySCO estimates the displacement field :math:`b` for which the blip-up
# image sampled at :math:`y + b` and the blip-down image sampled at
# :math:`y - b`, each multiplied by the Jacobian determinant of its
# transformation, agree. The Jacobian factor restores the intensity of voxels
# compressed into a pile-up or stretched over several voxels. A smoothness
# penalty on :math:`b` and a constraint that keeps both transformations
# invertible regularize the problem; the estimation is three-dimensional, and
# :func:`~bartorch.tools.correct_susceptibility` takes the slab with its voxel
# size and returns :math:`b` in millimetres.

result = bt.correct_susceptibility(blip_up, blip_down, voxel_size=VOXEL_MM, phase_encoding_axis=1)
corrected = 0.5 * (result.blip_up + result.blip_down).float()
estimated_mm = 0.5 * (result.field_map[:, 1:] + result.field_map[:, :-1]).float()  # voxel centres

# %%
#
# Results
# -------
#
# Each image is compared with the undistorted object as the normalized
# root-mean-square error (NRMSE) over the brain; the estimated displacement is
# compared with :math:`d` over the brain.


def nrmse(estimate, truth, region=brain):
    return float((estimate - truth)[region].norm() / truth[region].norm())


print(f"blip-up                NRMSE {nrmse(blip_up, image):.3f}")
print(f"blip-down              NRMSE {nrmse(blip_down, image):.3f}")
print(f"mean of the pair       NRMSE {nrmse(0.5 * (blip_up + blip_down), image):.3f}")
print(f"corrected              NRMSE {nrmse(corrected, image):.3f}")
error_mm = (estimated_mm - displacement_mm)[brain]
print(
    f"displacement over the brain: RMS {float(displacement_mm[brain].square().mean().sqrt()):.1f}"
    f" mm, RMS error of the estimate {float(error_mm.square().mean().sqrt()):.1f} mm"
)

# %%

# sphinx_gallery_start_ignore
SLICE = 5
ZOOM = (slice(4, 44), slice(20, 76))
COLUMN = 40
peak = float(image[SLICE][head[SLICE]].quantile(0.99))
nan = torch.tensor(float("nan"))

figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
show(axes[0], image[SLICE] / peak, "object, T2-weighted")
handle = show(
    axes[1],
    torch.where(head[SLICE], field_map[SLICE], nan),
    "field map at 3 T",
    -120,
    120,
    "RdBu_r",
)
axes[1].contour(brain[SLICE], levels=[0.5], colors="0.35", linewidths=0.6)
figure.colorbar(handle, ax=axes[1], fraction=0.046, label="off-resonance [Hz]")
plt.show()

panels = (
    (image, "undistorted"),
    (blip_up, "blip-up"),
    (blip_down, "blip-down"),
    (corrected, "corrected"),
)
figure, axes = plt.subplots(1, 4, figsize=(10.4, 3.0))
for axis, (values, title) in zip(axes, panels, strict=True):
    show(axis, values[SLICE] / peak, title)
    axis.add_patch(
        Rectangle(
            (ZOOM[1].start, ZOOM[0].start),
            ZOOM[1].stop - ZOOM[1].start,
            ZOOM[0].stop - ZOOM[0].start,
            fill=False,
            edgecolor="#e8a33d",
            linewidth=1.2,
        )
    )
    axis.axvline(COLUMN, color="#3dbde8", linewidth=0.8, linestyle="--")
plt.show()

figure, axes = plt.subplots(1, 4, figsize=(10.4, 2.4))
for axis, (values, title) in zip(axes, panels, strict=True):
    show(axis, values[SLICE][ZOOM] / peak, title)
figure.suptitle("orbitofrontal region", color="#e8a33d")
plt.show()

figure, axis = plt.subplots(figsize=(7.2, 3.0))
rows = torch.arange(SIZE) * VOXEL_MM[1]
for (values, title), style in zip(panels, ("-", "--", "--", "-"), strict=True):
    axis.plot(rows, values[SLICE, :, COLUMN] / peak, style, label=title, linewidth=1.2)
axis.set_xlabel("anterior to posterior [mm]")
axis.set_ylabel("signal / peak")
axis.set_xlim(0, 120)
axis.legend(frameon=False, ncol=4, loc="upper right")
axis.set_title("profile along the phase-encoding axis, dashed line above")
plt.show()

figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
for axis, (values, title, limit) in zip(
    axes,
    (
        (displacement_mm, "displacement d", 10),
        (estimated_mm, "estimated b", 10),
        (estimated_mm - displacement_mm, "b - d", 10),
    ),
    strict=True,
):
    handle = show(
        axis, torch.where(head[SLICE], values[SLICE], nan), title, -limit, limit, "PuOr_r"
    )
    axis.contour(brain[SLICE], levels=[0.5], colors="0.35", linewidths=0.6)
figure.colorbar(handle, ax=axes, fraction=0.03, label="along phase encoding [mm]")
plt.show()

figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
for axis, (values, title) in zip(axes, panels[1:], strict=True):
    difference = torch.where(head[SLICE], (values[SLICE] - image[SLICE]) / peak, nan)
    handle = show(axis, difference, f"{title} - undistorted", -0.3, 0.3, "RdBu_r")
    axis.contour(brain[SLICE], levels=[0.5], colors="0.35", linewidths=0.6)
figure.colorbar(handle, ax=axes, fraction=0.03, label="difference / peak")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Where the offset is positive -- the scalp and the fat anterior to the
# frontal lobes, and the orbitofrontal cortex -- the blip-up image displaces
# signal towards posterior and the blip-down image towards anterior; in the
# lateral temporal lobes, where it is negative, the directions are exchanged.
# Where the displacement decreases along its own direction, neighbouring
# voxels converge onto one location and form a bright pile-up, as the frontal
# scalp does in the blip-up image; where it increases, their signal is
# stretched over more voxels and darkened. The mean of the pair superimposes
# both distortions. The corrected image restores the position and the
# intensity of the cortex. Its residual is largest in the scalp, where the
# displacement reaches several voxels and changes within a few, and where a
# pile-up has summed the signal of separate voxels into one, which no
# transformation of the pair separates.
#
# The estimate of :math:`b` follows :math:`d` over the brain; it is smoother,
# as the regularization requires, and has no information where the images
# carry no signal. In practice the pair is acquired as two short
# :math:`b = 0` series, the displacement field is estimated once, and it is
# applied to every diffusion-weighted volume acquired with one of the two
# polarities.
#
# References
# ----------
#
# .. [#andersson] Andersson JLR, Skare S, Ashburner J. How to correct
#    susceptibility distortions in spin-echo echo-planar images: application
#    to diffusion tensor imaging. *NeuroImage* 20(2):870-888 (2003).
#    https://doi.org/10.1016/S1053-8119(03)00336-7
#
# .. [#pyhysco] Julian A, Ruthotto L. PyHySCO: GPU-enabled susceptibility
#    artifact distortion correction in seconds. *Front Neurosci* (2024).
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ,
#    Holmes CJ, Evans AC. Design and construction of a realistic digital brain
#    phantom. *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
