"""The figure style every gallery example is executed under.

Figures are drawn on a transparent canvas with text, ticks and frames in a
mid grey, which reads on the theme's light and dark backgrounds alike; the
images inside them stay opaque.  ``custom.css`` lets these figures through the
dark theme without the white card and dimming it gives other images.

sphinx-gallery calls :func:`reset` before each example, so an example sets
only what is particular to its own figures.  :func:`domain` and
:func:`phase_bar` draw complex images, such as coil sensitivity maps.
"""

from __future__ import annotations

import functools

#: Mid grey: a contrast ratio of 3.45:1 against a white page and above 5:1
#: against the dark theme's background, so that it clears the 3:1 that
#: graphical objects need on both.
FOREGROUND = "#8a8a8a"

STYLE = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "figure.constrained_layout.use": True,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 13,
    "legend.frameon": False,
    "text.color": FOREGROUND,
    "axes.labelcolor": FOREGROUND,
    "axes.titlecolor": FOREGROUND,
    "axes.edgecolor": FOREGROUND,
    "xtick.color": FOREGROUND,
    "ytick.color": FOREGROUND,
    "xtick.labelcolor": FOREGROUND,
    "ytick.labelcolor": FOREGROUND,
    "grid.color": FOREGROUND,
    "grid.alpha": 0.3,
    "image.cmap": "gray",
    "image.interpolation": "nearest",
}


def apply() -> None:
    """Set the gallery's figure style on matplotlib's global parameters."""
    import matplotlib

    matplotlib.rcParams.update(STYLE)


def reset(gallery_conf, fname) -> None:
    """The ``reset_modules`` entry sphinx-gallery calls before each example."""
    import matplotlib

    matplotlib.rcdefaults()
    apply()


@functools.cache
def phase_colormap():
    """The cyclic colormap of phase: yellow at zero, blue at +/- pi."""
    from cmap import Colormap
    from matplotlib.colors import ListedColormap

    cyclic = Colormap("colorcet:CET_C2").to_matplotlib().reversed()
    return ListedColormap(cyclic([((step + 60) % 256) / 255 for step in range(256)]))


def domain(axis, values, title=None, ceiling=None):
    """Draw a complex image with phase as colour and magnitude as brightness.

    ``ceiling`` is the magnitude drawn at full brightness; the image's own
    peak when omitted.  Panels of one figure share a ceiling to be comparable.
    """
    import numpy as np

    values = values.detach().cpu()
    colours = phase_colormap()((values.angle() / (2 * np.pi) + 0.5).numpy())[..., :3]
    magnitude = values.abs().numpy()
    peak = float(magnitude.max()) if ceiling is None else float(ceiling)
    magnitude = np.clip(magnitude / max(peak, 1e-12), 0.0, 1.0)
    axis.imshow(colours * magnitude[..., None])
    if title is not None:
        axis.set_title(title)


def phase_bar(figure, axes):
    """Add the colour-to-phase key for the panels in ``axes``."""
    import matplotlib.pyplot as plt
    import numpy as np

    bar = figure.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(-np.pi, np.pi), phase_colormap()),
        ax=axes,
        fraction=0.046,
        ticks=[-np.pi, 0.0, np.pi],
    )
    bar.ax.set_yticklabels(["$-\\pi$", "0", "$\\pi$"])
    bar.set_label("phase [rad]")
