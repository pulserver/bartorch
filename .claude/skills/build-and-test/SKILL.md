---
name: build-and-test
description: Build bartorch's native library from this checkout and run its linter, test suite and documentation build. Use before reporting any change complete, and whenever a test result or a skip has to be quoted rather than assumed.
---

# Build and test

`libbartorch` is compiled from `src/csrc/`, `external/bart` and
`external/finufft`, so a checkout needs its submodules and a build before the
suite imports anything. Run the whole sequence and report the results it
actually produced.

```bash
git submodule update --init --recursive
./scripts/lint.sh                   # ruff over src/ and tests/; --fix writes
./scripts/run_tests.sh              # builds into build/local, regenerates, runs pytest
./scripts/build_docs.sh             # warnings are errors; --execute runs the examples
```

`run_tests.sh` passes any further argument to pytest
(`./scripts/run_tests.sh tests/test_solve.py -k pics`). After a source file is
added, moved or removed, run it with `--rebuild`: CMake caches the list.

## Interpreting the run

- The compiler has to put BART's nested functions on the heap: GCC 14 or newer
  (`-ftrampoline-impl=heap`), or clang. CMake stops on anything older.
- OpenMP is required unless `-DBARTORCH_OPENMP=OFF` asks for a
  single-threaded build: `libomp-dev` beside clang on Linux, `brew install
  libomp` on macOS.
- The first configure fetches xsimd, POET and DUCC0 for FINUFFT and needs
  network access; `CPM_SOURCE_CACHE` keeps them.
- CUDA tests skip without a card; `python scripts/check_device.py` is the
  check on one. The FFT tests that assert MKL planned need `pip install mkl`,
  and the `LinearPhysics` adapter tests need `deepinv`. Report each skip; it is
  not a pass.
- After changing `src/csrc/include/bartorch.h`, run `python scripts/gen_abi.py`;
  `tests/test_abi.py` fails when `_abi.py` is not what it writes. After a BART
  submodule bump, run `python scripts/gen_catalogue.py`.
- Off a grid nothing is bit-exact: FINUFFT sums over threads in the order they
  finish, so compare to round-off, not with `torch.equal`.

## Documentation

Documentation changes are validated by the documentation build and
`tests/test_docs.py`, not by reading the source. See the `write-documentation`
skill.
