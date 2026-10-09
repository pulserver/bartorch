"""That the library under test was built from the sources under test.

Nothing else here would notice.  The Python side is read from the checkout, so
an edit to it takes effect the moment a test imports it; the C side is a
shared object that was compiled at some point in the past, and a stale one
answers every call happily with the old behaviour.  What that looks like is a
test that passes for a fix that is not in the binary, or fails for a fix that
is -- which is worse than either.

Some of it is caught already.  ``_abi.py`` is generated from the header and
``tests/test_abi.py`` holds the two together, and every symbol the header
declares is looked up at load, so a header that grew an entry point the
library does not have fails at import.  What is left is everything that does
not change the set of symbols: an entry point whose arguments changed, and any
edit at all to a ``.c`` file.  For those there is nothing to compare but the
clock.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# The same definition `scripts/run_tests.sh` builds against, so that what it
# thinks is up to date and what this thinks is up to date cannot disagree.
import sources  # noqa: E402


def test_the_library_is_newer_than_the_sources_it_was_built_from():
    """Otherwise the suite is measuring a build that no longer exists.

    ``scripts/run_tests.sh`` builds before it runs, so the ordinary path never
    reaches this.  What reaches it is ``pytest tests/`` after a change to the C
    side, which is exactly when a green run means nothing.
    """
    from bartorch._lib import library_path

    source, changed = sources.newest()
    if source is None:
        pytest.skip("the C sources are not beside this checkout")

    library = library_path()
    built = sources.built(library)
    assert built >= changed, (
        f"{library} was built before {source.relative_to(ROOT)} was last changed "
        f"({built:.0f} < {changed:.0f}), so these tests are running against a stale "
        "library; rebuild with `./scripts/run_tests.sh` or `cmake --build <dir>`"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="the loader's flags are Linux's")
def test_the_library_loads_alone_in_a_new_process():
    """Every library it calls into is on its own link line.

    One that is not is found only in what the process loaded before: torch
    loads first in ``_lib._load``, and some of its builds put librt in the
    process and others do not, so ``shm_open`` resolved under the CPU torch
    of the wheel's tests and failed to load under a CUDA torch on glibc 2.31.
    Loaded alone with every symbol bound at once, a missing link fails here,
    on whatever glibc the suite runs on: the wheel's own tests run on the
    manylinux floor.
    """
    from bartorch._lib import library_path

    code = (
        f"import ctypes, os; ctypes.CDLL({str(library_path())!r}, mode=os.RTLD_NOW | os.RTLD_LOCAL)"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert loaded.returncode == 0, loaded.stderr
