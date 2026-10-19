# Performance

```{admonition} TL;DR
:class: tldr

- BLAS, LAPACK and the FFT come from compiled libraries already in your process: MKL, then PyTorch's library, then SciPy's OpenBLAS.
- One call, {func}`bartorch.set_num_threads`, sets the threads of BART, its FFT and FINUFFT.
- Arithmetic is single precision complex (`complex64`); the NUFFT is planned at a tolerance of $10^{-3}$ on a grid oversampled by 1.25.
```

You meet these settings when you time a reconstruction, compare it with BART's
`bart` executable, or share a machine with other work. The library underneath
is BART's, but the linear algebra, the FFT and the non-uniform transform are
served by optimized libraries that bartorch finds at import, and the accuracy
of the NUFFT is a setting. {doc}`execution-model` states which tools run on a
CUDA device and how they are ordered against PyTorch's stream.

## Capabilities

:::{container} capabilities

- **Runs on the device of its tensors: OpenMP threads on the host, cuFFT, cuBLAS and cuFINUFFT on a CUDA card.**

  Code: `src/csrc/abi/cuda.c`. Tests: *a tool on a device tensor returns a device tensor*; *an operator applies on the device* (`test_cuda.py`).
- **Sets one thread count for BART, its FFT and FINUFFT's host transforms.**

  Code: {func}`~bartorch.set_num_threads`. Tests: *one thread count covers bart and the transform* (`test_finufft.py`).
- **Takes every BLAS and LAPACK routine from a compiled library in the process, MKL first, and reports which one serves each.**

  Code: `_backend.py`, {func}`~bartorch.backend_sources`. Tests: *every blas and lapack routine comes from a compiled library*; *lapack is served by a library not by the built in reference*; *an installed mkl serves every routine* (`test_core.py`).
- **Executes BART's FFT with MKL's DFTI where a source has it, and with the transform compiled into the library otherwise.**

  Code: `src/csrc/substitute/fft.cpp`. Tests: *mkl and the built in transform agree*; *every plan a reconstruction makes goes to mkl*; *the fft reports where it came from* (`test_fft.py`).
- **Runs FINUFFT at the newest x86-64 instruction-set level the processor supports, or the one `BARTORCH_FINUFFT_SIMD` names.**

  Code: `_finufft.py`. Tests: *the newest level the processor runs is the default*; *the environment names the level plans start at*; *every simd level matches an explicit dft*; *a plan keeps the level it was made at* (`test_finufft.py`).
- **Computes FINUFFT's FFT with oneMKL where the process has it, loading no second MKL or OpenMP runtime.**

  Code: `src/csrc/substitute/fftw_bind.c`. Tests: *onemkl is the default fft where the process has it*; *finufft on onemkl loads no mkl or openmp runtime of its own* (`test_finufft.py`).
- **Plans the NUFFT at a tolerance of $10^{-3}$ on a grid oversampled by 1.25, and calibrating tools at $10^{-6}$ on a grid oversampled by 2.**

  Code: `_finufft.py`, `_CALIBRATES` in `_dispatch.py`. Tests: *precision can be traded for a transform that fits*; *a tool that calibrates gets the careful transform* (`test_finufft.py`).
- **Runs BART on one to eight CUDA streams, and keeps or returns freed device memory.**

  Code: {func}`~bartorch.set_cuda_streams`, {func}`~bartorch.use_cuda_memcache`. Tests: *more than one stream can be asked for*; *stream count is rejected outside barts range* (`test_cuda.py`).

:::

## Division of responsibility

- **BART**: the parallel loops of its commands and operators, and cuFFT and
  cuBLAS on a device.
- **bartorch**: which library serves each routine, the thread count, the
  FINUFFT instruction-set level and its tolerance and grid.
- **Other libraries**: MKL (the `mkl` extra), PyTorch's library and SciPy's
  OpenBLAS for BLAS, LAPACK and the FFT; FINUFFT and cuFINUFFT for the NUFFT;
  PyTorch's OpenMP runtime for the threads ({ref}`openmp-runtime`).

## Method

```{image} ../_static/architecture.svg
:class: only-light
:alt: PyTorch tensors enter the tools, apps and operators, which call the bartorch C ABI through ctypes; the ABI runs the embedded BART, whose non-uniform transforms and linear algebra come from substituted backends
:width: 100%
```

```{image} ../_static/architecture-dark.svg
:class: only-dark
:alt: PyTorch tensors enter the tools, apps and operators, which call the bartorch C ABI through ctypes; the ABI runs the embedded BART, whose non-uniform transforms and linear algebra come from substituted backends
:width: 100%
```

### Library order

At import, each routine is taken from the first source that exports it: MKL
when the `mkl` extra is installed, then PyTorch's `libtorch_cpu` (with
Accelerate on macOS), then SciPy's `cython_blas` and `cython_lapack`.
`BARTORCH_BLAS_LIBRARY` set to `mkl`, `torch`, `scipy` or a library path puts
that source first, and the others still fill what it lacks. On a GPU, BART calls
cuBLAS; eigendecompositions and SVDs run on the host table, since BART has no
GPU LAPACK. Timings of each source are in
{doc}`../guides/developer/internals/performance`.

### Precision

Every array BART reads or writes is `complex64`, so a `complex128` or real
input is converted, and the result is `complex64`. On the host, a NUFFT
tolerance below what single precision reaches at a given size is planned at the
tolerance it can reach.

### NUFFT accuracy

The default of $10^{-3}$ on a grid oversampled by 1.25 suits a reconstruction,
whose error is set by its data rather than by its transform. `oversampling` and
`width` of {class}`~bartorch.linop.NUFFT`, and `-o` and `-w` of the tools, ask
for another grid or kernel width; on a CUDA device only 2 and 1.25 are
accepted. {doc}`non-cartesian` describes the transform.

## Examples

- {doc}`../auto_examples/04-non-cartesian/02-radial-sense`: a radial SENSE
  reconstruction, where the NUFFT dominates the time.
- {doc}`../api/functions`: threads, backend sources and the CUDA settings.
