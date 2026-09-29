"""
==================
Receive bias field
==================

The intensity shading that a surface receive array leaves in a
root-sum-of-squares image, estimated and removed by N4 bias field correction.

The object is BART's brain phantom, whose tissue classes each have one
intensity; the array is BART's analytical eight-element head coil. N4
[#tustison]_ is SimpleITK's implementation, reached through
:func:`bartorch.tools.bias_field_correct`.
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
import torch

import bartorch
import bartorch.tools as bt

SIZE = 128

# %%
#
# The shaded image
# ----------------
#
# The coil images are the object weighted by each element's sensitivity, plus
# independent noise. Their root sum of squares is the object weighted by the
# root sum of squares of the sensitivities, which is smooth, multiplicative and
# largest near the elements.

brain = bt.phantom(SIZE, geometry="brain").abs()
brain = brain / brain.max()
support = brain > 0

head_coil = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=8)[:, 0]


def acquire(elements):
    """Root sum of squares of noisy coil images, and the field it carries."""
    elements = elements / bartorch.rss(elements, axes=(0,)).abs().max()
    coil_images = bt.noise(elements * brain, n=1e-4, s=3)
    return bartorch.rss(coil_images, axes=(0,)).abs(), bartorch.rss(elements, axes=(0,)).abs()


observed, field = acquire(head_coil)
inside = field[support]
print(f"the field spans a factor {float(inside.max() / inside.min()):.1f} over the object")

# %%
#
# N4 correction
# -------------
#
# N4 models the logarithm of the image as the logarithm of the object plus a
# smooth field, represented by B-splines, and estimates the field by
# iteratively sharpening the histogram of the log intensities. It needs no
# model of the coil, only an object whose intensities form classes, and it is
# blind to the origin of the shading: a genuine smooth intensity variation of
# the object is removed as well.
#
# The measure of uniformity is the coefficient of variation over the
# brightest tissue class, which is zero in the object.

corrected, estimated = bt.bias_field_correct(observed, return_field=True)

brightest = brain > 0.99


def variation(image):
    return float(image[brightest].std() / image[brightest].mean())


print(
    f"coefficient of variation: observed {variation(observed):.3f}, "
    f"corrected {variation(corrected):.3f}"
)

# %%
#
# A multiplicative field is determined up to a constant, which the correction
# leaves in the image, so the estimate is compared with the true field after
# matching their means over the object.

scale = field[support].mean() / estimated[support].mean()
residual = ((scale * estimated - field).abs() / field)[support]
print(f"field recovered to {100 * float(residual.median()):.1f} % (median over the object)")

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 4, figsize=(8.0, 2.4))
level = float(brain[brightest].mean() / corrected[brightest].mean())
for axis, values, title in (
    (axes[0], brain, "object"),
    (axes[1], observed / observed[brightest].mean(), "root sum of squares"),
    (axes[2], level * corrected, "N4-corrected"),
):
    axis.imshow(values, cmap="gray", vmin=0, vmax=1.3)
    axis.set_title(title)
nan = torch.tensor(float("nan"))
handle = axes[3].imshow(torch.where(support, scale * estimated / field[support].mean(), nan))
axes[3].set_title("estimated field")
figure.colorbar(handle, ax=axes[3], fraction=0.046)
for axis in axes:
    axis.set_xticks([])
    axis.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The mask and the fitting grid
# -----------------------------
#
# The field is fitted only over a mask, by default Otsu's threshold of the
# image. The object's full support is the alternative compared here.

full = bt.bias_field_correct(observed, mask=support.to(torch.uint8))
print(f"Otsu mask    {variation(corrected):.3f}")
print(f"full support {variation(full):.3f}")

# %%
#
# The field is fitted on the image shrunk by ``shrink_factor`` and evaluated
# on the full grid; a field that is smooth on the scale of the object needs
# few grid points, and the cost of each N4 iteration falls with their number.

for shrink in (1, 2, 4, 8):
    estimate = bt.bias_field_correct(observed, shrink_factor=shrink)
    print(f"shrink_factor {shrink}: coefficient of variation {variation(estimate):.3f}")

# %%
#
# A stronger field
# ----------------
#
# Three elements on one side of the head leave a field that varies by an
# order of magnitude over the object, and the same correction recovers it
# less closely.

strong, strong_field = acquire(head_coil[:3])
inside = strong_field[support]
flattened, strong_estimate = bt.bias_field_correct(strong, return_field=True)
scale = strong_field[support].mean() / strong_estimate[support].mean()
residual = ((scale * strong_estimate - strong_field).abs() / strong_field)[support]
print(f"the field spans a factor {float(inside.max() / inside.min()):.1f} over the object")
print(
    f"coefficient of variation: observed {variation(strong):.3f}, "
    f"corrected {variation(flattened):.3f}"
)
print(f"field recovered to {100 * float(residual.median()):.1f} % (median over the object)")

# %%
#
# The correction applies to a magnitude image after coil combination. In a
# SENSE reconstruction the sensitivities are normalized so that their root sum
# of squares is one, which removes the same shading within the
# reconstruction; N4 is for images whose sensitivities are not available,
# such as a root-sum-of-squares combination.
#
# References
# ----------
#
# .. [#tustison] Tustison NJ, Avants BB, Cook PA, Zheng Y, Egan A, Yushkevich PA,
#    Gee JC. N4ITK: improved N3 bias correction. *IEEE Trans Med Imaging*
#    29(6):1310-1320 (2010). https://doi.org/10.1109/TMI.2010.2046908
