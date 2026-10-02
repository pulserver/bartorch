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

CI runs the test matrix (Linux with GCC 14, macOS and Windows with clang, each on Python 3.10 and 3.14, and a CUDA build without a device), the lint and spelling checks, and two documentation builds:
the reference without the compiled library, and the executed gallery, which is
published from `main` and from release tags.  A pull request is merged when CI passes and a
maintainer has approved it.
