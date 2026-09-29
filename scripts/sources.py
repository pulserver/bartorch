"""The sources the compiled library is built from.

Shared by ``scripts/run_tests.sh``, which decides whether to rebuild, and
``tests/test_build.py``, which fails when the library is older than its sources::

    python scripts/sources.py            # the newest source, and its time
    python scripts/sources.py --newer-than build/local/libbartorch.so
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Everything that ends up in ``libbartorch``.  The BART and FINUFFT submodules
#: are here because their sources are compiled in too, so a bump is a rebuild; the CMake
#: files are here because what they say is compiled is part of the answer.
SOURCES = (
    ROOT / "src" / "csrc",
    ROOT / "cmake",
    ROOT / "external" / "bart" / "src",
    ROOT / "external" / "finufft" / "src",
    ROOT / "external" / "finufft" / "include",
    ROOT / "external" / "finufft" / "cmake",
    ROOT / "external" / "finufft" / "CMakeLists.txt",
    ROOT / "CMakeLists.txt",
)

#: What a source file is.  Anything else under those directories -- a README,
#: a stray object file -- says nothing about whether the library is current.
SUFFIXES = frozenset({".c", ".h", ".hpp", ".cc", ".cpp", ".cu", ".cuh", ".cmake", ".txt"})


def files(roots=SOURCES):
    """Every source file under *roots*, whatever shape the roots are."""
    for root in roots:
        if root.is_file():
            yield root
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in SUFFIXES:
                    yield path


def newest(roots=SOURCES) -> tuple[Path | None, float]:
    """The most recently modified source, and when it was modified.

    ``(None, 0.0)`` where there are no sources to find, which is what an
    installed wheel looks like: the library is there and the checkout is not.
    """
    latest: Path | None = None
    when = 0.0
    for path in files(roots):
        stamp = path.stat().st_mtime
        if stamp > when:
            latest, when = path, stamp
    return latest, when


def built(library: Path) -> float:
    """When the build that produced *library* last wrote any of its libraries.

    The FINUFFT modules are linked beside it, and a source only a module is
    compiled from (``fftw_bind.c``) relinks that module and not the library.
    """
    beside = library.parent.glob(f"libbartorch_finufft_*{library.suffix}")
    return max(path.stat().st_mtime for path in [library, *beside])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--newer-than",
        metavar="FILE",
        help="say nothing and exit 0 if FILE is at least as new as every source, "
        "1 if it is older or missing",
    )
    args = ap.parse_args()

    source, when = newest()
    if source is None:
        print("no sources found beside this checkout", file=sys.stderr)
        return 0 if args.newer_than else 1

    if args.newer_than:
        library = Path(args.newer_than)
        if not library.exists():
            return 1
        if built(library) >= when:
            return 0
        print(source.relative_to(ROOT))
        return 1

    print(f"{source.relative_to(ROOT)} {when:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
