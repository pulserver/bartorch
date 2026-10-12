# Installation

## PyTorch build

Install PyTorch first, with the command the
[PyTorch installation selector](https://pytorch.org/get-started/locally/) gives
for your device: the CPU build, or the build for your CUDA version.  If PyTorch
is missing, pip installs the default build from PyPI with bartorch.  The
version floors are in {doc}`prerequisites`.

## Package and extras

```bash
python -m pip install bartorch
python -m pip install 'bartorch[mkl,io]'   # with extras, comma-separated
```

The wheel contains the compiled library with BART and FINUFFT embedded; you
need no BART or FINUFFT installation and no compiler.  The extras are listed
under {doc}`prerequisites`.

Check the installation by building a phantom:

```python
import bartorch
import bartorch.tools as bt

print(bartorch.__version__, bartorch.bart_version())
print(bartorch.build_info())
print(bartorch.backend_sources())
print(bt.phantom(32).shape)
```

{func}`bartorch.build_info` reports the compiler, the OpenMP and CUDA
configuration of the library, and the FINUFFT version and FFT compiled into it.
{func}`bartorch.backend_sources` reports the library serving each BLAS and
LAPACK routine, BART's FFT and FINUFFT's FFT: MKL when the `mkl` extra is
installed, otherwise the routines PyTorch links, and SciPy's for the rest.

(gpu-builds)=
## GPU builds

A CUDA device needs a CUDA build of PyTorch and bartorch's library compiled for
the same CUDA major version, a separate package installed by an extra.  Both
exist for Linux x86-64 only.

| PyTorch build (`torch.version.cuda`) | bartorch |
| --- | --- |
| 12.x | `python -m pip install 'bartorch[cu12]'` |
| 13.x | `python -m pip install 'bartorch[cu13]'` |

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'bartorch[cu12]'
```

bartorch loads the CUDA build in place of its own library when it matches
PyTorch's CUDA major version and the installed bartorch release, and raises an
`ImportError` naming the extra to install otherwise.
{func}`bartorch.cuda_available` and `cuda=ON` in {func}`bartorch.build_info`
report the library loaded.  The CUDA build contains BART's kernels and
cuFINUFFT for compute capabilities 7.5, 8.0 and 9.0, with the 8.0 code serving
8.6 and 8.9 and PTX for newer devices, and links the CUDA runtime, cuFFT and
cuBLAS installed with PyTorch's CUDA build.

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

A checkout of the repository is built as {doc}`../developer/installation`
describes.

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

## Command-line interface

The `bartorch` command accepts the command lines of BART's `bart` executable
and operates on CFL files:

```sh
bartorch pics -l1 -r0.01 -i30 kspace sensitivities image
bartorch --list
```

A script written for `bart` runs with the command name replaced, or with a
`bart` symbolic link to `bartorch` earlier on the `PATH`.  The commands that
run as {mod}`bartorch.apps` pipelines are listed in {doc}`../../api/cli`.
