# Installation

## PyTorch installation

Install the PyTorch build for the target device first, with the command the
[PyTorch installation selector](https://pytorch.org/get-started/locally/)
gives for the CPU or for a CUDA version.  bartorch requires PyTorch 2.2 or
later (2.3 on macOS, 2.7.1 on Linux); installing bartorch into an environment without PyTorch installs the
default build from PyPI.

## bartorch installation

```bash
python -m pip install bartorch
```

This installs the wheel for the platform where one exists (see
{doc}`prerequisites`) together with NumPy, SciPy, BlochSim and MRI-NUFFT.  A
wheel contains the compiled library with BART and FINUFFT embedded; no BART or
FINUFFT installation and no compiler are needed.  Optional components are installed as
extras:

```bash
python -m pip install 'bartorch[mkl]'        # Linux x86-64: MKL for BLAS, LAPACK and FFT
python -m pip install 'bartorch[deepinv]'    # bartorch.interop.to_deepinv
python -m pip install 'bartorch[learning]'   # Lightning and TorchIO, for bartorch.learning.Reconstruction and RandomGain
python -m pip install 'bartorch[correct]'    # SimpleITK, for bias field and gradient nonlinearity correction
python -m pip install 'bartorch[motion]'     # SimpleITK, for navigator registration
python -m pip install 'bartorch[pyhysco]'    # PyHySCO (GPL-3.0-only), for susceptibility correction
```

The installation is checked by building a phantom:

```python
import bartorch
import bartorch.tools as bt

print(bartorch.__version__, bartorch.bart_version())
print(bartorch.build_info())
print(bartorch.backend_sources())
image = bt.phantom(32)
print(image.shape, image.dtype, image.device)
```

{func}`bartorch.build_info` reports the compiler, the OpenMP and CUDA
configuration of the library and the FINUFFT version compiled into it, and {func}`bartorch.backend_sources` the library
serving each BLAS and LAPACK routine and the FFT: MKL when the `mkl` extra is
installed, otherwise the routines PyTorch links, and SciPy's for the rest.

(source-builds)=
## Source builds

Where no wheel exists, pip builds the source distribution, which needs:

| Tool | Requirement |
| --- | --- |
| C and C++ compiler | clang, or GCC 14 or later; BART's nested functions are compiled as clang Blocks or as GCC heap trampolines, and older GCC is rejected at configuration.  On Windows, clang from MSYS2's CLANG64 environment, as {doc}`../developer/prerequisites` describes |
| CMake | 3.25 or later |
| Git and network access | FINUFFT's build fetches the versions of xsimd, POET and DUCC0 it pins, and CCCL for a CUDA build; `CPM_SOURCE_CACHE` names a directory that keeps them for later builds |
| OpenMP | Linux: the compiler's OpenMP runtime, for example `libomp-dev` with clang on Debian and Ubuntu.  macOS: the header from Homebrew's `libomp`.  Windows: MSYS2's `llvm-openmp`, for the header.  On macOS and Windows the library links against PyTorch's runtime.  Without OpenMP the build fails unless `-C cmake.define.BARTORCH_OPENMP=OFF` asks for a single-threaded library |

The compiler is selected with `CC` and `CXX`:

```bash
CC=clang CXX=clang++ python -m pip install bartorch --no-binary bartorch
```

FINUFFT computes the FFT inside its CPU transform with DUCC0, which every
wheel has compiled into the library.  On x86-64 Linux,
a source build with oneMKL installed (`pip install mkl mkl-devel`) can use
oneMKL's FFT through its FFTW3 interface instead:

```bash
python -m pip install "bartorch[mkl]" --no-binary bartorch \
    -C cmake.define.BARTORCH_FINUFFT_FFT=MKL
```

The library then links the `libmkl_rt` found at build time, and the `mkl`
extra makes the same library serve BART's BLAS, LAPACK and FFT.
`finufft_fft=` in {func}`bartorch.build_info` names the FFT the library was
built with.

On x86-64 Linux and Windows the library carries FINUFFT's CPU transform
compiled for any x86-64 processor, and in modules beside it for the x86-64-v2
(SSE4.2), x86-64-v3 (AVX2 and FMA) and x86-64-v4 (AVX-512) levels; the newest
level the processor supports is used.  The environment variable
`BARTORCH_FINUFFT_SIMD` names another level, `x86-64` for the baseline, and
`finufft_simd=` in {func}`bartorch.build_info` lists the levels the library was
built for.  A source build selects them with
`-C cmake.define.BARTORCH_FINUFFT_SIMD="x86-64-v3"`, or builds the baseline
alone with an empty value.

On x86-64 Linux each of those levels is also built on oneMKL's FFT, and is
used in place of DUCC0 when oneMKL is installed:

```bash
python -m pip install "bartorch[mkl]"
```

`bartorch.backend_sources()["finufft_fft"]` names the FFT in use.  A source
build carries these modules when oneMKL's headers (`mkl-include`) are present,
which the build requires on that platform;
`-C cmake.define.BARTORCH_FINUFFT_MKL_MODULES=OFF` leaves them out.

A checkout of the repository is built as described in the
{doc}`developer guide <../developer/installation>`.

## CUDA

CUDA support requires a CUDA build of both PyTorch and bartorch;
`torch.cuda.is_available()` and {func}`bartorch.cuda_available` report each.
The PyPI wheels are CPU builds, and there is no CUDA build on macOS or
Windows.  The CUDA build of each version, a Linux
x86-64 wheel with the same file name, is attached to the
[GitHub release](https://github.com/pulserver/bartorch/releases) of that
version:

```bash
python -m pip install https://github.com/pulserver/bartorch/releases/download/<tag>/<wheel>
```

It contains cuFINUFFT and device code for compute capabilities 7.5, 8.0 and
9.0, the code for 8.0 serving 8.6 and 8.9 as well, and links the CUDA 12
runtime, cuFFT and cuBLAS dynamically, which a CUDA 12 build of PyTorch
provides.  A source build with CUDA passes
`-C cmake.define.BARTORCH_CUDA=ON` to pip and needs the CUDA toolkit with
`nvcc`, version 12.1 or later for compute capability 9.0;
`-C cmake.define.BARTORCH_CUDA_ARCHITECTURES="80;86"` selects the compute
capabilities for BART's kernels and cuFINUFFT alike.

## Command-line interface

Installation provides the `bartorch` command, which accepts the command lines
of BART's `bart` executable and operates on CFL files:

```sh
bartorch pics -l1 -r0.01 -i30 kspace sensitivities image
bartorch --list
```

A script written for `bart` runs with the command name replaced, or with a
`bart` symbolic link to `bartorch` earlier on the `PATH`.  Which commands run
as {mod}`bartorch.apps` pipelines is stated in {doc}`../../api/cli`.
