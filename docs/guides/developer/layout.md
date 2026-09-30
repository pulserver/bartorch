# Repository layout

Code written for bartorch is under `src/`, with its tests under `tests/` and
its scripts under `scripts/`; code from other projects is under `external/`.

| Path | Contents |
| --- | --- |
| `external/bart/` | BART, a Git submodule of the downstream fork `pulserver/bart` ({doc}`bart-fork`) |
| `external/finufft/` | FINUFFT and cuFINUFFT, a Git submodule of upstream `flatironinstitute/finufft` at one reviewed commit, built by its own CMake and linked into the library (`cmake/finufft.cmake`) |
| `external/pocketfft/`, `external/blocksruntime/` | Vendored FFT and Blocks runtime, with their licenses |
| `src/csrc/include/bartorch.h` | The C ABI, the only header the Python side sees |
| `src/csrc/abi/` | Command execution, the in-memory CFL registry, CUDA stream ordering |
| `src/csrc/ops/` | Operators, the encoding executor, and the configuration of BART's iterative solve as `pics` sets it up |
| `src/csrc/substitute/` | Components compiled in place of BART's: the FINUFFT NUFFT, point spread functions, the FFT and BLAS/LAPACK tables |
| `src/bartorch/` | The Python package; private modules begin with an underscore |
| `src/bartorch/_abi.py`, `_catalogue.py` | Generated from the header and from BART's command declarations |
| `scripts/` | Scripts run by hand: tests, lint, documentation, generators, device checks, artwork |
| `cmake/` | Build helpers: FINUFFT's configuration, and the OpenMP runtime the library binds to |
| `tests/` | The test suite |
| `docs/` | Documentation sources, the example gallery under `docs/examples/`, and the Sphinx configuration |
| `docs/design/` | Design records for maintainers, excluded from the built documentation |
| `attic/prototype/` | An earlier pybind11 extension, kept for reference and not built |

The compiled library uses no Python or PyTorch C API.  Python passes data
pointers and reversed dimension vectors through ctypes, which is why one wheel
per platform serves every Python and PyTorch version.  Changes to BART's
behaviour are made by replacing a translation unit or by a compile definition,
never by editing the submodule checkout; `AGENTS.md` lists each replacement.
A change that belongs in BART itself goes to the fork, as {doc}`bart-fork`
describes.

## The Python package

| Path | Contents |
| --- | --- |
| `fourier.py`, `wavelet.py`, `thresh.py`, `util.py`, `interp.py`, `kspace.py`, `_settings.py` | The array functions and runtime settings re-exported as `bartorch.*` |
| `linop/`, `nlop/` | One class per operator; `linop/form.py` is the encoding form and the plan it reports, `linop/plan.py` the matching of a composition against it |
| `optim/`, `priors/` | One class per BART iteration, and BART's regularization terms and denoisers |
| `apps/` | BART's reconstructions assembled from operators and solvers |
| `learning/`, `interop.py` | Unrolled iterations and channel conversions, and the DeepInverse adapter |
| `tools/` | BART's remaining commands in five sections, and `correct.py` and `motion.py`, implemented in `tools/_correct/` and `tools/_motion/` |
| `cli/` | The `bartorch` command line: `_argv.py` reads a `bart` command line, `_apps.py` routes it to an app |
| `io.py` | CFL files |
| `_abi.py`, `_lib.py`, `_marshal.py`, `_buffer.py` | The ctypes signatures, loading the library, the form of an ABI argument, and a tensor over one of BART's buffers |
| `_dispatch.py`, `_operator.py`, `_grid.py` | Running a command on tensors, what every operator shares, and what the operations on a grid share |
| `_backend.py`, `_finufft.py`, `_cuda.py` | The BLAS, LAPACK and FFT sources, and the controls of the NUFFT substitution and of the device |
| `_catalogue.py`, `_options.py`, `_call.py`, `_coverage.py` | BART's command declarations, the name of each option, the wrappers built from them, and where each command is exposed or why it is not |
| `_reference.py` | BART's reconstruction commands, private, against which the apps are tested |

## Documentation sources

| Path | Contents |
| --- | --- |
| `docs/index.md` | Landing page: the README and the top-level toctree |
| `docs/guides/user/`, `docs/guides/developer/` | User and developer guides |
| `docs/explanation/` | Conceptual explanations |
| `docs/examples/` | Example scripts, one directory per section with its header `README.rst`, and `README.rst`, the header of the Examples page |
| `docs/api/` | API category pages; their tables list the objects |
| `docs/api_objects.py` | Collects the objects from the API tables into `docs/api_objects.rst`, which generates one page per object under `docs/generated/` |
| `docs/_templates/autosummary/` | Templates of the generated object pages |
| `docs/misc/` | License, related projects, contributors and citation |
