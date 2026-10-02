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

This example shades a T1-weighted head with a head array that has no anterior
elements, estimates the field with N4 [#tustison]_ through
:func:`bartorch.tools.bias_field_correct`, and compares the corrected image and
the estimated field with the object and the true field.

**Learning objectives**

* Relate the shading of a root-sum-of-squares image to the receive
  sensitivities.
* Estimate and remove the bias field with N4, and assess the result by the
  uniformity of a tissue class and by the intensity histogram.
* Choose the mask and the fitting grid of the estimate.
* Recognise what N4 does not recover: the scale of the field, the steep part
  of a field of large range, and a smooth intensity variation that belongs to
  the object.
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
# The array is BART's analytical eight-element head coil without its two
# anterior elements, as in an open-face head coil: six elements. Each coil
# image is the object weighted by one element's sensitivity :math:`S_c`, with
# independent complex Gaussian noise; their root sum of squares is the object
# weighted by the bias field :math:`B = \sqrt{\sum_c |S_c|^2}`, and the noise
# adds a Rician floor in the background.


def shaded_image(elements):
    sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=8)[elements, 0]
    bias = bartorch.rss(sensitivities, axes=(0,)).abs()
    sensitivities = sensitivities / bias[brain].mean()
    coil_images = bt.noise(sensitivities * image, n=1e-4, s=3)
    return bartorch.rss(coil_images, axes=(0,)).abs(), bias / bias[brain].mean()


shaded, bias = shaded_image([0, 1, 2, 5, 6, 7])

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
# The field is fitted on the image shrunk by ``shrink_factor``, whose default
# of 4 would leave a 32 x 32 grid of this 128 matrix; a factor of 2 keeps
# 64 x 64. The uniformity of a tissue class is its coefficient of variation,
# the standard deviation over the mean, which the object has too through
# partial volume at the class boundaries.

corrected, estimate = bt.bias_field_correct(shaded, shrink_factor=2, return_field=True)


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


def agreement(estimate, bias):
    ratio = (estimate / estimate[brain].mean() / bias)[brain]
    return float(ratio.median()), float(((ratio - 1).abs() < 0.1).float().mean())


median, within = agreement(estimate, bias)
estimate = estimate / estimate[brain].mean()
print(
    f"estimated / true field over the brain: median {median:.3f}, "
    f"within 10 % in {100 * within:.0f} % of the voxels"
)

# %%

# sphinx_gallery_start_ignore
COLUMN = 64
filled = torch.as_tensor(ndimage.binary_fill_holes(brain.numpy()))
nan = torch.tensor(float("nan"))
rows = torch.arange(SIZE) * FOV_MM / SIZE
COLOURS = {"object": "#8a8a8a", "shaded": "#e8a33d", "N4-corrected": "#3dbde8"}


def normalised(values):
    return values / values[white].mean()


WINDOW = (0.2, 1.3)
tissue = filled & (image > 0.2)


def relative_profile(axis, values, **style):
    ratio = normalised(values) / normalised(image)
    axis.plot(rows, torch.where(tissue, ratio, nan)[:, COLUMN], **style)


def field_profile(axis, true, estimated):
    axis.plot(
        rows,
        torch.where(filled, true, nan)[:, COLUMN],
        color="#8a8a8a",
        linewidth=3.0,
        label="true",
    )
    axis.plot(
        rows,
        torch.where(filled, estimated, nan)[:, COLUMN],
        color="#3dbde8",
        linewidth=1.4,
        label="N4",
    )
    axis.set_xlabel("anterior to posterior [mm]")
    axis.set_ylabel("field / brain mean")
    axis.set_title("field on the line")
    axis.legend(loc="upper left")


panels = ((image, "object"), (shaded, "shaded"), (corrected, "N4-corrected"))
figure, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
for axis, (values, title) in zip(axes, panels, strict=True):
    show(axis, normalised(values), title, *WINDOW)
    axis.axvline(COLUMN, color="#3dbde8", linewidth=1.0, linestyle="--")
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for axis, (values, title) in zip(
    axes, ((bias, "true field"), (estimate, "N4 estimate")), strict=True
):
    handle = show(axis, torch.where(filled, values, nan), title, 0.4, 1.4, "magma")
figure.colorbar(handle, ax=axes, shrink=0.9, label="field / brain mean")
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
for values, title in panels[1:]:
    relative_profile(axes[0], values, color=COLOURS[title], label=title, linewidth=1.4)
axes[0].axhline(1.0, color="#8a8a8a", linewidth=1.0, linestyle=":")
axes[0].set_xlabel("anterior to posterior [mm]")
axes[0].set_ylabel("image / object")
axes[0].set_title("image / object on the line")
axes[0].legend(loc="upper left")
field_profile(axes[1], bias, estimate)
plt.show()

figure, axis = plt.subplots(figsize=(7.2, 3.2))
bins = torch.linspace(0, 1.6, 81)
for values, title in panels:
    counts = torch.histc(normalised(values)[brain], bins=80, min=0, max=1.6)
    axis.step(bins[:-1], counts, where="post", color=COLOURS[title], label=title, linewidth=1.4)
axis.set_xlabel("signal / white-matter mean")
axis.set_ylabel("voxels in the brain")
axis.set_title("intensity histogram of the brain")
axis.legend(loc="upper left")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The images share one window, each scaled to its white-matter mean. In the
# shaded image the frontal lobes are darker than the occipital lobes: along
# the dashed line the ratio of the shaded image to the object follows the
# field, and the histogram of the brain has no separate grey- and
# white-matter peaks. After correction the ratio stays close to one except at
# the frontal pole, where the estimated field does not fall as far as the true
# one, and the two peaks separate.
#
# The mask and the fitting grid
# -----------------------------
#
# The field is fitted only over a mask, by default Otsu's threshold of the
# image. The alternatives compared here are a mask of the whole head, which
# adds the scalp and the skull, and a brain mask, such as a skull-stripping
# tool provides; the BrainWeb tissue model gives it here.

head = ndimage.binary_fill_holes(shaded > 0.02 * shaded.max())
masks = {"Otsu": None, "head": torch.as_tensor(head), "brain": brain}
for name, mask in masks.items():
    trial = bt.bias_field_correct(shaded, mask=mask, shrink_factor=2)
    print(f"{name:5s} mask: white-matter variation {variation(trial, white):.3f}")

# %%
#
# N4 sharpens one histogram over the whole mask. The scalp and the skull add
# intensity classes of their own, and the fit over the head mask is the least
# uniform; the brain mask, which holds only the classes the field is judged
# by, is the most.
#
# The field is fitted on the image shrunk by ``shrink_factor`` and evaluated
# on the full grid. A field that is smooth on the scale of the head needs few
# grid points, and the cost of each N4 iteration falls with their number,
# until the shrunk image holds too few voxels of each tissue class.

for shrink in (1, 2, 4, 8):
    trial = bt.bias_field_correct(shaded, shrink_factor=shrink)
    print(f"shrink_factor {shrink}: white-matter variation {variation(trial, white):.3f}")

# %%
#
# What N4 does not recover
# ------------------------
#
# The estimate degrades as the range of the field grows. With the four
# posterior elements alone the field falls steeply towards the frontal pole,
# and N4 overestimates it there.

steep, steep_bias = shaded_image([0, 1, 6, 7])
steep_corrected, steep_estimate = bt.bias_field_correct(steep, shrink_factor=2, return_field=True)
low, high = steep_bias[brain].quantile(0.02), steep_bias[brain].quantile(0.98)
median, within = agreement(steep_estimate, steep_bias)
print(f"four posterior elements: field {float(low):.2f} to {float(high):.2f}")
print(
    f"white-matter variation: shaded {variation(steep, white):.3f}, "
    f"N4-corrected {variation(steep_corrected, white):.3f}; "
    f"estimate within 10 % in {100 * within:.0f} % of the voxels"
)

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), width_ratios=(1, 1, 1.25))
for axis, (values, title) in zip(
    axes, ((steep, "shaded"), (steep_corrected, "N4-corrected")), strict=False
):
    show(axis, normalised(values), title, *WINDOW)
    axis.axvline(COLUMN, color="#3dbde8", linewidth=1.0, linestyle="--")
field_profile(axes[2], steep_bias, steep_estimate / steep_estimate[brain].mean())
axes[2].set_xlabel("A to P [mm]")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The corrected frontal cortex remains darker than the occipital cortex,
# because the estimated field does not fall as far as the true one at the frontal pole. A
# residual of this kind is the reason a corrected image is still compared
# across regions with care.
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
