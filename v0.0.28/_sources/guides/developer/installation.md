# Development installation

(source-builds)=
## Source builds

pip builds the source distribution where no wheel exists, or when you pass
`--no-binary bartorch`.  The build needs:

| Tool | Requirement |
| --- | --- |
| C and C++ compiler | clang, or GCC 14 or later; older GCC is rejected at configuration, because BART's nested functions need clang's Blocks or GCC 14's heap trampolines.  On Windows, clang from MSYS2's CLANG64 environment with the packages `mingw-w64-clang-x86_64-{clang,cmake,ninja,llvm-openmp,llvm-tools}`, `clang64/bin` on `PATH` and `CMAKE_GENERATOR=Ninja` |
| CMake | 3.25 or later |
| Git and network access | FINUFFT's build fetches the xsimd, POET and DUCC0 versions it pins, and CCCL for a CUDA build; `CPM_SOURCE_CACHE` names a directory that keeps them between builds |
| OpenMP | Linux: the compiler's runtime, for example `libomp-dev` with clang on Debian and Ubuntu; without it the library is built single-threaded (`openmp=FALSE` in {func}`bartorch.build_info`).  macOS: `omp.h` from Homebrew's `libomp`, or a directory named by `BARTORCH_OMP_INCLUDE_DIR`.  Windows: MSYS2's `llvm-openmp`.  On macOS and Windows only the header is used: the library links against PyTorch's runtime |
| CUDA toolkit (optional) | `nvcc`, 12.1 or later for compute capability 9.0 |

The compiler is selected with `CC` and `CXX`, and CMake options are passed
with `-C cmake.define.NAME=VALUE`:

```bash
CC=clang CXX=clang++ python -m pip install bartorch --no-binary bartorch
python -m pip install bartorch --no-binary bartorch \
    -C cmake.define.BARTORCH_CUDA=ON \
    -C cmake.define.BARTORCH_CUDA_ARCHITECTURES="80;86"
```

| CMake option | Default | Effect |
| --- | --- | --- |
| `BARTORCH_OPENMP` | `ON` | `OFF` builds a single-threaded library |
| `BARTORCH_CUDA` | `OFF` | Compiles BART's CUDA kernels and cuFINUFFT into the library |
| `BARTORCH_CUDA_ARCHITECTURES` | `75;80;90` | Compute capabilities for BART's kernels and cuFINUFFT; PTX is added for the last |
| `BARTORCH_FINUFFT_SIMD` | `x86-64-v2;x86-64-v3;x86-64-v4` on x86-64 Linux and Windows | Additional x86-64 levels of FINUFFT's CPU transform; empty builds the baseline alone |
| `BARTORCH_FINUFFT_MKL_MODULES` | `AUTO` on x86-64 Linux | Each SIMD level also built on oneMKL's FFT, when oneMKL's headers (`mkl-include`) are present; `OFF` leaves them out |
| `BARTORCH_FINUFFT_FFT` | `AUTO` (DUCC0) | `MKL` links FINUFFT's FFT to oneMKL's FFTW3 interface; x86-64 Linux only, with `pip install mkl mkl-devel` |

## FINUFFT backends

FINUFFT computes every non-uniform FFT, and its CPU transform computes its FFT
with DUCC0.  On x86-64 Linux and Windows the library carries that transform
compiled for any x86-64 processor and, in modules beside it, for the
x86-64-v2 (SSE4.2), x86-64-v3 (AVX2 and FMA) and x86-64-v4 (AVX-512) levels;
the newest level your processor supports is used.  The environment variable
`BARTORCH_FINUFFT_SIMD` selects another level, `x86-64` for the baseline;
`finufft_simd=` in {func}`bartorch.build_info` lists the levels built.

On x86-64 Linux each level is also built on oneMKL's FFT, and that module
replaces the DUCC0 one when oneMKL is installed (the `mkl` extra).
`bartorch.backend_sources()["finufft_fft"]` names the FFT in use.

## Toolchain

A checkout is built with the toolchain of a source build, listed with the
build's CMake options under *Source builds* above, and with Git, for the
submodules `external/bart` and `external/finufft`.  A CUDA build has two
further requirements:

- a host compiler `nvcc` accepts, named by `CUDAHOSTCXX` when the compiler that
  builds BART is newer (CUDA 12.6 accepts GCC 13 at the newest; BART needs
  GCC 14 or clang);
- optionally MathDx, for the paired Toeplitz kernels (cuFFTDx) the CUDA wheels
  carry: `pip install nvidia-mathdx==25.6.0` and
  `-DBARTORCH_MATHDX_DIR=<site-packages>/nvidia/mathdx`.

## Repository clone

```bash
git clone --recurse-submodules https://github.com/YOUR-USERNAME/bartorch.git
cd bartorch
git remote add upstream https://github.com/pulserver/bartorch.git
```

`git submodule update --init --recursive` completes an existing clone.

## Editable installation

Install PyTorch first ({doc}`../user/index`), then:

```bash
CC=clang CXX=clang++ python -m pip install -e '.[dev]'
python -m pip install pre-commit 'codespell[toml]' -r docs/requirements.txt
```

The first command compiles the library through scikit-build-core into
`build/<wheel tag>/` and installs the package with pytest, pytest-xdist,
numpydoc and ruff; with CUDA, add
`--config-settings=cmake.define.BARTORCH_CUDA=ON`.  The second installs the
pre-commit hooks' runner, the spelling check, and the documentation and
gallery requirements.

Some tests skip unless an optional package is present:

```bash
python -m pip install 'bartorch[mkl,io,learning]' deepinv PyHySCO PyWavelets
```

## Pre-commit hooks

`.pre-commit-config.yaml` runs, on the files a commit changes:

| Hook | Check | CI counterpart |
| --- | --- | --- |
| `nbstripout` | Removes outputs from Jupyter notebooks, except the archived ones under `attic/` | None |
| `codespell` | Spelling, configured by `[tool.codespell]` in `pyproject.toml` | The Typos workflow, and the Lint workflow through `./scripts/lint.sh` |
| `ruff` | Lint rules, applying automatic fixes, on `src/`, `tests/` and `docs/examples/` | The Lint workflow (`./scripts/lint.sh`) |
| `ruff-format` | Formatting, on the same paths | The Lint workflow |

Install the hooks once per clone, and run them over the whole repository to
check it independently of a commit:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

A hook that rewrites a file fails the commit; stage the rewritten file and
commit again.  A deliberate exception is made for one commit, or for one hook
by its id:

```bash
git commit --no-verify
SKIP=codespell git commit
```

Bypassing a local hook does not bypass CI: the Lint and Typos workflows run
the same checks on every pull request.  The hook versions are pinned in the
configuration and CI installs the latest releases of ruff and codespell, so a
difference between the two is resolved by updating the pinned `rev`.

## Local builds

`./scripts/run_tests.sh` builds the library into `build/local` (only what
changed), regenerates the generated files, and runs the suite against `src/`
without installing; arguments after its options go to pytest.

```bash
./scripts/run_tests.sh
./scripts/run_tests.sh tests/test_solve.py -k pics
BARTORCH_BUILD_DIR=$PWD/build-clang CC=clang CXX=clang++ ./scripts/run_tests.sh --rebuild
./scripts/run_tests.sh --cuda
```

| Option | Effect |
| --- | --- |
| `--rebuild` | Configure the build directory again before building |
| `--cuda` | Configure with `-DBARTORCH_CUDA=ON`; implies `--rebuild` |
| `--no-build` | Run against the library as it stands |
| `--no-generate` | Skip regenerating `_abi.py` and `_catalogue.py` |

The same steps by hand:

```bash
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build/local -j
BARTORCH_LIBRARY=$PWD/build/local/libbartorch.so PYTHONPATH=src pytest tests/
```

The library is `libbartorch.dylib` on macOS and `libbartorch.dll` on Windows;
`run_tests.sh` handles Linux and macOS.  Keep builds with different compilers
or CUDA settings in separate build directories.  The CUDA tests skip without a
device; on a machine with one, `python scripts/check_device.py` checks the
device path in dependency order.
