# Pull requests

A pull request is opened against `pulserver/bartorch:main` from a topic branch,
as a draft while work remains.  Its description follows the repository's
template:

| Section | Contents |
| --- | --- |
| What changed | A concise summary |
| Why | The problem, the issue it addresses and, for a new method, the paper |
| Validation | The commands run and their results; for a numerical change, the reference and the tolerance; hardware or optional dependencies that were not available |
| Public API and documentation | Changes to public behaviour and the documentation updated for them |

CI runs on every pull request:

| Workflow | Checks |
| --- | --- |
| Tests | The suite on Linux with GCC 14, and on macOS and Windows with clang, each on Python 3.10 and 3.14; a CUDA build, tested on the host because the runner has no device |
| Lint, Typos | `./scripts/lint.sh`, and codespell |
| Docs | `./scripts/build_docs.sh`, the reference rendered without the compiled library, when `docs/` or `src/` changes.  The executed gallery and the PDF manual are built when a maintainer comments `/build-docs` on the pull request |
| Publish to PyPI | Every wheel built, repaired and tested as a release would be, when the pull request changes the build (`CMakeLists.txt`, `cmake/`, `pyproject.toml`, `src/cuda/`, `src/bartorch/_lib.py` or the workflow) |

A pull request is merged when CI passes and a maintainer has approved it.
