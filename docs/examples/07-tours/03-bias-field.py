"""
==================
Receive bias field
==================

A root-sum-of-squares combination of the images of a receive array is the
object weighted by the root sum of squares of the coil sensitivities. That
weighting is smooth, multiplicative and largest near the elements, and it
shades the image: the same tissue appears brighter near the array than far
from it, which biases segmentation, intensity-based registration and any
quantitative comparison across the field of view.

This example shades a T1-weighted head with the posterior elements of a head
array, estimates the field with N4 [#tustison]_ through
:func:`bartorch.tools.bias_field_correct`, and compares the corrected image and
the estimated field with the object and the true field.

**Learning objectives**

* Relate the shading of a root-sum-of-squares image to the receive
  sensitivities.
* Estimate and remove the bias field with N4, and assess the result by the
  uniformity of a tissue class and by the intensity histogram.
* Choose the mask and the fitting grid of the estimate.
* Recognise what N4 cannot recover: the scale of the field, and a smooth
  intensity variation that belongs to the object.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
import numpy as np
from brainweb_dl import get_mri


def brainweb_slice(size, fov_mm, slice_mm):
    """A T1-weighted BrainWeb axial slice, and its brain, white- and grey-matter masks."""
    t1 = get_mri(sub_id=0, contrast="T1")[slice_mm].astype(np.float32)
    fuzzy = get_mri(sub_id=0, contrast="fuzzy")[slice_mm]
    rows, cols = t1.shape
    side = int(round(fov_mm))  # BrainWeb is at 1 mm
    canvas = np.zeros((4, side, side), dtype=np.float32)
    top, left = (side - rows) // 2, (side - cols) // 2
    canvas[:, top : top + rows, left : left + cols] = [
        t1 / t1.max(),
        fuzzy[..., 1:4].sum(-1),
        fuzzy[..., 3],
        fuzzy[..., 2],
    ]
    canvas = np.flip(canvas, axis=1)  # anterior at the top
    image, brain, white, grey = (ndimage.zoom(c, size / side, order=1) for c in canvas)
    return image.clip(0), brain > 0.5, white > 0.9, grey > 0.9


def show(axis, values, title, vmin=0.0, vmax=1.0, cmap="gray"):
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_axis_off()
    return handle


# sphinx_gallery_end_ignore
import torch
from scipy import ndimage

import bartorch
import bartorch.tools as bt

# %%
#
# Object
# ------
#
# The object is an axial slice of the BrainWeb T1-weighted head [#brainweb]_
# through the lateral ventricles, 128 x 128 over a 220 mm field of view. The
# BrainWeb tissue model also gives the voxels that are at least 90 % white or
# grey matter, over which the uniformity of each class is measured.

SIZE, FOV_MM = 128, 220.0
# sphinx_gallery_start_ignore
image, brain, white, grey = (torch.as_tensor(a) for a in brainweb_slice(SIZE, FOV_MM, 90))
# sphinx_gallery_end_ignore

# %%
#
# The shaded image
# ----------------
#
# The array is the posterior half of BART's analytical eight-element head
# coil, four elements. Each coil image is the object weighted by one
# element's sensitivity :math:`S_c`, with independent complex Gaussian noise;
# their root sum of squares is the object weighted by the bias field
# :math:`B = \sqrt{\sum_c |S_c|^2}`, and the noise adds a Rician floor in the
# background.

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=8)[[0, 1, 6, 7], 0]
bias = bartorch.rss(sensitivities, axes=(0,)).abs()
sensitivities = sensitivities / bias[brain].mean()
bias = bias / bias[brain].mean()

coil_images = bt.noise(sensitivities * image, n=1e-4, s=3)
shaded = bartorch.rss(coil_images, axes=(0,)).abs()

low, high = bias[brain].quantile(0.02), bias[brain].quantile(0.98)
print(f"bias field over the brain: {float(low):.2f} to {float(high):.2f} (2nd to 98th percentile)")

# %%
#
# N4 correction
# -------------
#
# N4 models the logarithm of the image as the logarithm of the object plus a
# smooth field, represented by cubic B-splines, and estimates the field by
# alternately sharpening the histogram of the log intensities and fitting the
# splines to what the sharpening removed, over a hierarchy of control-point
# grids. It uses no model of the coil, only an object whose intensities form
# classes.
#
# The uniformity of a tissue class is its coefficient of variation, the
# standard deviation over the mean, which the object has too through partial
# volume at the class boundaries.

corrected, estimate = bt.bias_field_correct(shaded, return_field=True)


def variation(values, region):
    return float(values[region].std() / values[region].mean())


for name, values in (("object", image), ("shaded", shaded), ("N4-corrected", corrected)):
    contrast = values[white].mean() / values[grey].mean()
    print(
        f"{name:13s} coefficient of variation: white matter {variation(values, white):.3f}, "
        f"grey matter {variation(values, grey):.3f}; white/grey {float(contrast):.2f}"
    )

# %%
#
# A multiplicative field is determined up to a constant factor, which the
# correction leaves in the image. The estimate is therefore compared with the
# true field after scaling both to unit mean over the brain.

estimate = estimate / estimate[brain].mean()
ratio = (estimate / bias)[brain]
within = float(((ratio - 1).abs() < 0.1).float().mean())
print(
    f"estimated / true field over the brain: median {float(ratio.median()):.3f}, "
    f"within 10 % in {100 * within:.0f} % of the voxels"
)

# %%

# sphinx_gallery_start_ignore
COLUMN = 64
filled = torch.as_tensor(ndimage.binary_fill_holes(brain.numpy()))
nan = torch.tensor(float("nan"))


def normalised(values):
    return values / values[white].mean()


panels = ((image, "object"), (shaded, "shaded"), (corrected, "N4-corrected"))
figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
for axis, (values, title) in zip(axes, panels, strict=True):
    show(axis, normalised(values), title, 0, 1.4)
    axis.axvline(COLUMN, color="#3dbde8", linewidth=0.8, linestyle="--")
plt.show()

figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
for axis, (values, title) in zip(
    axes, ((bias, "true field"), (estimate, "N4 estimate")), strict=False
):
    handle = show(axis, torch.where(filled, values, nan), title, 0, 2, "magma")
figure.colorbar(handle, ax=axes[:2], fraction=0.046, label="field / mean over brain")
handle = show(
    axes[2], torch.where(filled, estimate / bias, nan), "estimate / true", 0.8, 1.2, "RdBu_r"
)
figure.colorbar(handle, ax=axes[2], fraction=0.046)
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.2))
rows = torch.arange(SIZE) * FOV_MM / SIZE
for (values, title), style in zip(panels, ("-", "--", "-"), strict=True):
    axes[0].plot(rows, normalised(values)[:, COLUMN], style, label=title, linewidth=1.2)
axes[0].set_xlabel("anterior to posterior [mm]")
axes[0].set_ylabel("signal / white-matter mean")
axes[0].set_title("profile along the dashed line")
axes[0].legend(frameon=False)
bins = torch.linspace(0, 1.6, 81)
for (values, title), style in zip(panels, ("-", "--", "-"), strict=True):
    counts = torch.histc(normalised(values)[brain], bins=80, min=0, max=1.6)
    axes[1].step(bins[:-1], counts, style, where="post", label=title, linewidth=1.2)
axes[1].set_xlabel("signal / white-matter mean")
axes[1].set_ylabel("voxels in the brain")
axes[1].set_title("intensity histogram")
axes[1].legend(frameon=False)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The field varies by a factor of four over the brain: in the shaded image
# the occipital lobes are bright and the frontal lobes dark, and the
# histogram of the brain has no separate grey- and white-matter peaks. After
# correction the variation of the white matter falls by more than half and
# the two peaks separate. The
# estimated field is within 10 % of the true one over most of the brain; it
# overestimates the field at the frontal pole, where the true field falls
# steeply to a third of its mean and below, so the corrected frontal cortex
# remains darker than the object. A residual of this kind is the reason a
# corrected image is still compared across regions with care.
#
# The mask and the fitting grid
# -----------------------------
#
# The field is fitted only over a mask, by default Otsu's threshold of the
# image, which under strong shading can exclude the darkest tissue. The
# alternative compared here is a mask of the whole head.

head = ndimage.binary_fill_holes(shaded > 0.02 * shaded.max())
with_head = bt.bias_field_correct(shaded, mask=torch.as_tensor(head, dtype=torch.uint8))
print(f"Otsu mask: white-matter variation {variation(corrected, white):.3f}")
print(f"head mask: white-matter variation {variation(with_head, white):.3f}")

# %%
#
# The field is fitted on the image shrunk by ``shrink_factor`` and evaluated
# on the full grid. A field that is smooth on the scale of the head needs few
# grid points, and the cost of each N4 iteration falls with their number.

for shrink in (1, 2, 4, 8):
    trial = bt.bias_field_correct(shaded, shrink_factor=shrink)
    print(f"shrink_factor {shrink}: white-matter variation {variation(trial, white):.3f}")

# %%
#
# N4 removes any smooth intensity variation, whatever its origin: a receive
# field, a transmit field in a gradient-echo image, or a genuine slow change
# of the tissue signal. In a SENSE reconstruction the sensitivities are
# normalized to unit root sum of squares, which removes the receive shading
# within the reconstruction; N4 serves images for which the sensitivities are
# not available, such as a root-sum-of-squares combination, and the transmit
# field, which no receive calibration measures.
#
# References
# ----------
#
# .. [#tustison] Tustison NJ, Avants BB, Cook PA, Zheng Y, Egan A, Yushkevich PA,
#    Gee JC. N4ITK: improved N3 bias correction. *IEEE Trans Med Imaging*
#    29(6):1310-1320 (2010). https://doi.org/10.1109/TMI.2010.2046908
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ,
#    Holmes CJ, Evans AC. Design and construction of a realistic digital brain
#    phantom. *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
