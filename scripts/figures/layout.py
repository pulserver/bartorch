"""The figure on the data-layout explanation page."""

import itertools

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def memory() -> plt.Figure:
    """One buffer read as a C-order tensor and as a BART array."""
    coils, ny, nx = 2, 2, 3
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    cells = list(itertools.product(range(coils), range(ny), range(nx)))
    for offset, (c, y, x) in enumerate(cells):
        ax.add_patch(
            Rectangle((offset, 0), 1, 1, facecolor=cycle[c], alpha=0.35, edgecolor=cycle[c], lw=1.2)
        )
        ax.text(offset + 0.5, 0.5, str(offset), ha="center", va="center", fontsize=11)
        ax.text(offset + 0.5, 1.25, f"{c},{y},{x}", ha="center", va="bottom", fontsize=10)
        ax.text(offset + 0.5, -0.25, f"{x},{y},{c}", ha="center", va="top", fontsize=10)
    ax.text(-0.3, 1.25, "tensor[c, y, x]\nshape (2, 2, 3)", ha="right", va="bottom", fontsize=11)
    ax.text(-0.3, -0.25, "BART [x, y, c]\ndims [3, 2, 2]", ha="right", va="top", fontsize=11)
    ax.text(-0.3, 0.5, "memory offset", ha="right", va="center", fontsize=11)
    ax.set_xlim(-4.2, len(cells) + 0.1)
    ax.set_ylim(-1.3, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


FIGURES = {"layout-memory.svg": memory}
