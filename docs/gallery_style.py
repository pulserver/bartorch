"""The figure style every gallery example is executed under.

Figures are drawn on a transparent canvas with text, ticks and frames in a
mid grey, which reads on the theme's light and dark backgrounds alike; the
images inside them stay opaque.  ``custom.css`` lets these figures through the
dark theme without the white card and dimming it gives other images.

sphinx-gallery calls :func:`reset` before each example, so an example sets
only what is particular to its own figures.
"""

from __future__ import annotations

#: Mid grey: 4.5:1 or better against both the light and the dark page.
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
