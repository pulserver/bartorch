"""
==================
Receive bias field
==================

When you combine the images of a receive array by root sum of squares, what
you get is the object weighted by the root sum of squares of the coil
sensitivities. That weight is smooth, multiplicative and largest near the
elements, so it shades the image: the same tissue looks brighter near the
array than far from it. The shading biases segmentation, intensity-based
registration and any quantitative comparison across the field of view.

In this tour you shade a T1-weighted head with a head array that has no
anterior elements, estimate the field with N4 [#tustison]_ through
:func:`bartorch.tools.bias_field_correct`, and compare the corrected image
and the estimated field with the object and the true field.

Prerequisites: lesson 1, :doc:`../01-basics/01-tensors-and-commands`.

**Learning objectives**

- Relate the shading of a root-sum-of-squares image to the receive
  sensitivities.
- Estimate and remove the bias field with N4, and judge the result by the
  uniformity of a tissue class and by the intensity histogram.
- Choose the mask and the fitting grid of the estimate.
- Recognise what N4 does not recover: the scale of the field, the steep part
  of a field of large range, and a smooth intensity variation that belongs to
  the object.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
import numpy as np
from brainweb_dl import get_mri
from scipy import ndimage


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


def variation(values, region):
    """Coefficient of variation over ``region``."""
    return float(values[region].std() / values[region].mean())


def agreement(estimate, bias):
    """Median of estimate / truth over the brain, and the fraction within 10 %."""
    ratio = (estimate / estimate[brain].mean() / bias)[brain]
    return float(ratio.median()), float(((ratio - 1).abs() < 0.1).float().mean())


# sphinx_gallery_end_ignore
import torch

import bartorch
import bartorch.tools as bt

# %%
# The object
# ----------
#
# Your object is an axial slice of the BrainWeb T1-weighted head [#brainweb]_
# through the lateral ventricles, 128 x 128 over a 220 mm field of view. The
# BrainWeb tissue model also tells you which voxels are at least 90 % white or
# grey matter, and you measure the uniformity of each class over those.
SIZE, FOV_MM = 128, 220.0

# sphinx_gallery_start_ignore
image, brain, white, grey = (torch.as_tensor(a) for a in brainweb_slice(SIZE, FOV_MM, 90))
# sphinx_gallery_end_ignore

# %%
# The shaded image
# ----------------
#
# The array is BART's analytical eight-element head coil with its two anterior
# elements removed, as in an open-face head coil: six elements. Each coil
# image is the object weighted by one element's sensitivity :math:`S_c`, plus
# independent complex Gaussian noise. Their root sum of squares is the object
# weighted by the bias field :math:`B = \sqrt{\sum_c |S_c|^2}`, and the noise
# adds a Rician floor in the background.


def shaded_image(elements):
    grid = bt.grid(D=(SIZE, SIZE, 1))
    sensitivities = bt.coils(t=grid, n=8)[elements, 0]
    bias = bartorch.rss(sensitivities, axes=(0,)).abs()
    sensitivities = sensitivities / bias[brain].mean()
    coil_images = bt.noise(sensitivities * image, n=1e-4, s=3)
    shaded = bartorch.rss(coil_images, axes=(0,)).abs()
    return shaded, bias / bias[brain].mean()


shaded, bias = shaded_image([0, 1, 2, 5, 6, 7])

# sphinx_gallery_start_ignore
low, high = bias[brain].quantile(0.02), bias[brain].quantile(0.98)
print(f"bias field over the brain: {float(low):.2f} to {float(high):.2f} (2nd to 98th percentile)")
# sphinx_gallery_end_ignore

# %%
# N4 correction
# -------------
#
# N4 works on the logarithm of the image, where the field is no longer a
# factor but an additive term: the log of the object plus a smooth field,
# represented by cubic B-splines. It estimates the field by alternately
# sharpening the histogram of the log intensities and fitting the splines to
# what the sharpening removed, over a hierarchy of control-point grids. It
# knows nothing about the coil; all it needs is an object whose intensities
# form classes.
#
# The field is fitted on the image shrunk by ``shrink_factor``. The default
# of 4 would leave a 32 x 32 grid of this 128 matrix, so you pass 2, which
# keeps 64 x 64. Ask for the field as well as the corrected image:
corrected, estimate = bt.bias_field_correct(
    shaded,
    shrink_factor=2,
    return_field=True,
)

# %%
# You measure the uniformity of a tissue class by its coefficient of
# variation, the standard deviation over the mean. The object has some too,
# from partial volume at the class boundaries, so that is the floor to
# compare with.

# sphinx_gallery_start_ignore
for name, values in (
    ("object", image),
    ("shaded", shaded),
    ("N4-corrected", corrected),
):
    contrast = values[white].mean() / values[grey].mean()
    print(
        f"{name:13s} coefficient of variation: "
        f"white matter {variation(values, white):.3f}, "
        f"grey matter {variation(values, grey):.3f}; "
        f"white/grey {float(contrast):.2f}"
    )
# sphinx_gallery_end_ignore

# %%
# The correction takes the white-matter variation from more than ten times
# the object's to about three times it, and keeps the white/grey contrast of
# the object.
#
# A multiplicative field is only determined up to a constant factor, and the
# correction leaves that factor in the image. So you compare the estimate with
# the true field after scaling both to unit mean over the brain:
estimate = estimate / estimate[brain].mean()

# sphinx_gallery_start_ignore
median, within = agreement(estimate, bias)
print(
    f"estimated / true field over the brain: median {median:.3f}, "
    f"within 10 % in {100 * within:.0f} % of the voxels"
)

COLUMN = 64
filled = torch.as_tensor(ndimage.binary_fill_holes(brain.numpy()))
nan = torch.tensor(float("nan"))
rows = torch.arange(SIZE) * FOV_MM / SIZE
COLOURS = {"object": "C7", "shaded": "C1", "N4-corrected": "C0"}


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
        color="C7",
        linewidth=3.0,
        label="true field",
    )
    axis.plot(
        rows,
        torch.where(filled, estimated, nan)[:, COLUMN],
        color="C0",
        linewidth=1.4,
        label="N4 field",
    )
    axis.set_xlabel("anterior to posterior [mm]")
    axis.set_ylabel("field / brain mean")
    axis.set_title("field on the line")


panels = ((image, "object"), (shaded, "shaded"), (corrected, "N4-corrected"))
figure, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
for axis, (values, title) in zip(axes, panels):
    show(axis, normalised(values), title, *WINDOW)
    axis.axvline(COLUMN, color="C0", linewidth=1.0, linestyle="--")
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for axis, (values, title) in zip(axes, ((bias, "true field"), (estimate, "N4 estimate"))):
    handle = show(axis, torch.where(filled, values, nan), title, 0.4, 1.4, "magma")
figure.colorbar(handle, ax=axes, shrink=0.9, label="field / brain mean")
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
for values, title in panels[1:]:
    relative_profile(axes[0], values, color=COLOURS[title], label=title, linewidth=1.4)
axes[0].axhline(1.0, color="C7", linewidth=1.0, linestyle=":")
axes[0].set_xlabel("anterior to posterior [mm]")
axes[0].set_ylabel("image / object")
axes[0].set_title("image / object on the line")
field_profile(axes[1], bias, estimate)
for axis in axes:
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
plt.show()

figure, axis = plt.subplots(figsize=(7.2, 3.2))
bins = torch.linspace(0, 1.6, 81)
for values, title in panels:
    counts = torch.histc(normalised(values)[brain], bins=80, min=0, max=1.6)
    axis.step(bins[:-1], counts, where="post", color=COLOURS[title], label=title, linewidth=1.4)
axis.set_xlabel("signal / white-matter mean")
axis.set_ylabel("voxels in the brain")
axis.set_title("intensity histogram of the brain")
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The images share one window, each scaled to its white-matter mean. In the
# shaded image you see the frontal lobes darker than the occipital lobes.
# Along the dashed line the ratio of the shaded image to the object follows
# the field, and the histogram of the brain has no separate grey- and
# white-matter peaks. After correction the ratio stays close to one except at
# the frontal pole, where the estimated field does not fall as far as the
# true one, and the two peaks separate again. Over the brain the estimated
# field is within 10 % of the true one in more than nine voxels out of ten.
#
# The mask
# --------
#
# N4 fits the field only over a mask, by default Otsu's threshold of the
# image. You can give it a mask of the whole head instead, which adds the
# scalp and the skull, or a brain mask such as a skull-stripping tool gives
# you; here the BrainWeb tissue model provides it.

# sphinx_gallery_start_ignore
head = ndimage.binary_fill_holes(shaded > 0.02 * shaded.max())
# sphinx_gallery_end_ignore
masks = {"Otsu": None, "head": torch.as_tensor(head), "brain": brain}
for name, mask in masks.items():
    trial = bt.bias_field_correct(shaded, mask=mask, shrink_factor=2)
    spread = variation(trial, white)
    print(f"{name:5s} mask: white-matter variation {spread:.3f}")

# %%
# N4 sharpens one histogram over the whole mask. The scalp and the skull
# bring intensity classes of their own, so the fit over the head mask is the
# least uniform. The brain mask, which holds only the classes you judge the
# field by, gives the most uniform white matter.
#
# What N4 does not recover
# ------------------------
#
# The estimate gets worse as the range of the field grows. Keep only the four
# posterior elements, and the field falls steeply towards the frontal pole:
steep, steep_bias = shaded_image([0, 1, 6, 7])
steep_corrected, steep_estimate = bt.bias_field_correct(
    steep,
    shrink_factor=2,
    return_field=True,
)

# sphinx_gallery_start_ignore
low, high = steep_bias[brain].quantile(0.02), steep_bias[brain].quantile(0.98)
median, within = agreement(steep_estimate, steep_bias)
print(f"four posterior elements: field {float(low):.2f} to {float(high):.2f}")
print(
    f"white-matter variation: shaded {variation(steep, white):.3f}, "
    f"N4-corrected {variation(steep_corrected, white):.3f}; "
    f"estimate within 10 % in {100 * within:.0f} % of the voxels"
)

figure, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), width_ratios=(1, 1, 1.25))
for axis, (values, title) in zip(axes, ((steep, "shaded"), (steep_corrected, "N4-corrected"))):
    show(axis, normalised(values), title, *WINDOW)
    axis.axvline(COLUMN, color="C0", linewidth=1.0, linestyle="--")
field_profile(axes[2], steep_bias, steep_estimate / steep_estimate[brain].mean())
axes[2].set_xlabel("A to P [mm]")
figure.legend(*axes[2].get_legend_handles_labels(), loc="outside lower center", ncol=2)
plt.show()
# sphinx_gallery_end_ignore

# %%
# N4 overestimates the field at the frontal pole: its estimate does not fall
# as far as the true one, so the corrected frontal cortex stays darker than
# the occipital cortex. A residual like this is why you still compare a
# corrected image across regions with care.
#
# There is a second limit, and it is not a failure of the fit. N4 removes any
# smooth intensity variation, whatever its origin: a receive field, a transmit
# field in a gradient-echo image, or a genuine slow change of the tissue
# signal. In a SENSE reconstruction the sensitivities are normalized to unit
# root sum of squares, which removes the receive shading inside the
# reconstruction. N4 is for images whose sensitivities you do not have, such
# as a root-sum-of-squares combination, and for the transmit field, which no
# receive calibration measures.
#
# Your turn: the fitting grid
# ---------------------------
#
# The field is fitted on the shrunk image and evaluated on the full grid. A
# field that is smooth on the scale of the head needs few grid points, and
# each N4 iteration gets cheaper as their number falls. Vary
# ``shrink_factor`` and watch the white-matter uniformity:
for shrink in (1, 2, 4, 8):
    trial = bt.bias_field_correct(shaded, shrink_factor=shrink)
    spread = variation(trial, white)
    print(f"shrink_factor {shrink}: white-matter variation {spread:.3f}")

# %%
# Factors of 1 to 4 do about equally well here. At 8 the shrunk image is 16 x
# 16 and holds too few voxels of each tissue class, and the variation rises.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Correct the receive bias field of a 2D root-sum-of-squares magnitude
#    image with bt.bias_field_correct (N4), shrink_factor 2, returning the
#    field. Compare a mask of None (Otsu), the whole head and the brain.
#    Report the coefficient of variation over white and grey matter before
#    and after, the white/grey contrast, and the fraction of brain voxels
#    where the estimated field, scaled to unit brain mean, is within 10 % of
#    the true field.

# %%
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
