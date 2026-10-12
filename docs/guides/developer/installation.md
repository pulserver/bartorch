# Development installation

## Toolchain

A checkout is built with the toolchain of a source build, listed with the
build's CMake options under {ref}`source-builds`, and with Git, for the
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

Install PyTorch first ({doc}`../user/installation`), then:

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
