# Development workflow

A substantial change to the public API is described in an issue before a pull
request is opened; small fixes and documentation corrections go directly to a
pull request.  Work on a topic branch of a fork, one coherent change per
branch.

## Change procedure

1. Read `AGENTS.md`, and the implementation and tests of the code being
   changed.
2. Change the code and the documentation of the public behaviour it affects.
   Generated files are regenerated, not edited.
3. Run the checks:

   | Change | Command |
   | --- | --- |
   | Python | `./scripts/lint.sh`: ruff's format and lint checks and codespell; `--fix` applies ruff's formatting and fixes |
   | Any code | `./scripts/run_tests.sh`, or a subset of the suite during development |
   | C, CUDA or the ABI | The full suite; `python scripts/check_device.py` on a machine with a CUDA device |
   | Documentation | `./scripts/build_docs.sh` (warnings are errors), and `./scripts/build_docs.sh --execute` for changed examples |
   | PDF manual | `./scripts/build_docs_pdf.sh`, after `python -m playwright install chromium` |

4. Commit, push to the fork, and open a pull request ({doc}`pull-requests`).

A topic branch takes upstream changes by rebasing, which rewrites its history:

```bash
git fetch upstream
git rebase upstream/main
git submodule update --init --recursive
```

## Generated files

| File | Generator | Test holding it |
| --- | --- | --- |
| `src/bartorch/_abi.py` | `python scripts/gen_abi.py`, from `src/csrc/include/bartorch.h` | `tests/test_abi.py` |
| `src/bartorch/_catalogue.py` | `python scripts/gen_catalogue.py`, from BART's command declarations | `tests/test_catalogue.py` |
| Logo, mark, architecture and encoding figures in `docs/_static/` | `python scripts/make_artwork.py` | None |
| Plots in `docs/_static/figures/` | `python scripts/make_figures.py [module]`, from `scripts/figures/` | None |

`run_tests.sh` regenerates both Python files before the suite.  The logo's
sources and their terms are listed in {doc}`../../misc/license`.  The
documentation's build products, `docs/generated/`, `docs/api_objects.rst`,
`docs/auto_examples/` and the PDF manual, are not tracked;
`./scripts/build_docs.sh --clean` removes them.

## Published documentation

The docs workflow publishes into the `gh-pages` branch, one directory per
version: `latest` from `main`, and `vX.Y.Z` from a release tag, which
`scripts/publish_docs.py` also copies to `stable` when it is the newest
release.  The site root redirects to `stable`, or to `latest` before the first
release, and the version switcher reads `versions.json` beside the
directories.

## BART submodule updates

`external/bart` pins a revision of the downstream fork `pulserver/bart`
({doc}`bart-fork`).  Moving it is a change of its own: sync the fork, update the
submodule pointer, run `python scripts/gen_catalogue.py`, and review the
resulting API differences.  `tests/test_tools.py` requires every BART command
to be wrapped by hand, generated into {mod}`bartorch.tools`, or listed as
private with a reason in `src/bartorch/_coverage.py`.

## FINUFFT submodule updates

`external/finufft` is upstream FINUFFT at one commit, unmodified.  Moving it is
a change of its own: choose a commit whose upstream CI passed, update the
submodule pointer, record the commit and the reason in the pull request, and
run the suite on every platform.  `tests/test_finufft.py` holds the transform
against explicit discrete Fourier sums and against BART's gridder.  A change
FINUFFT needs goes upstream, not into the checkout.

## Tests

A numerical test compares BART with a reference outside BART: an independent
FFT or DFT, an explicit sum, a closed-form signal, a finite difference or an
adjoint identity.  Comparing two entry points into the same BART
implementation establishes nothing about its correctness.  Tests are pytest
functions whose names state the property tested.
