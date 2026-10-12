"""Draw the plots on the explanation pages into ``docs/_static/figures/``.

Each module under ``scripts/figures/`` holds ``FIGURES``, a mapping from a
file name to a function returning a matplotlib figure.  The figures are drawn
in the gallery's style (``docs/gallery_style.py``): a transparent canvas and a
mid-grey foreground, readable on the light and the dark theme alike.

    python scripts/make_figures.py            # every figure
    python scripts/make_figures.py solvers    # one module's
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "_static" / "figures"

sys.path.insert(0, str(ROOT / "docs"))
import gallery_style  # noqa: E402


def _modules(names):
    for path in sorted((ROOT / "scripts" / "figures").glob("*.py")):
        if names and path.stem not in names:
            continue
        spec = importlib.util.spec_from_file_location(f"figures.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


def main() -> None:
    import matplotlib.pyplot as plt

    gallery_style.apply()
    plt.rcParams["svg.hashsalt"] = "bartorch"
    plt.rcParams["svg.fonttype"] = "none"
    OUT.mkdir(parents=True, exist_ok=True)
    for module in _modules(sys.argv[1:]):
        for name, draw in module.FIGURES.items():
            figure = draw()
            figure.savefig(OUT / name, metadata={"Date": None})
            plt.close(figure)
            print(OUT / name)


if __name__ == "__main__":
    main()
