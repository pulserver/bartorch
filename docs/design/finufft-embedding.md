# FINUFFT compiled into the library

FINUFFT and cuFINUFFT are built from the submodule `external/finufft` and
linked into `libbartorch`.  This note records the dependency graph, the pin,
the FFT backend and the OpenMP arrangement, with the measurements the choices
rest on.

## Dependency graph

```
libbartorch (shared; exports the bartorch_* C ABI only)
├── BART sources (external/bart)                     C, OpenMP::OpenMP_C
├── src/csrc/substitute/finufft.c ── finufft.h, cufinufft.h
├── finufft (static)            external/finufft/src  C++17, OpenMP::OpenMP_CXX
│   ├── finufft_common (static)
│   ├── ducc0 (static)          FFT; fetched by FINUFFT's CPM at the tag it pins
│   ├── xsimd, POET             header-only; fetched likewise
│   └── OpenMP::OpenMP_CXX ──┐
├── cufinufft (static)          CUDA builds only
│   ├── CCCL                    the toolkit's, or fetched where it has no config
│   └── CUDA::cudart, CUDA::cufft ── dynamic, from the nvidia wheels torch brings
└── OpenMP::OpenMP_C ────────────┤
                                 └── cmake/openmp.cmake
                                     Linux    the toolchain's runtime (libgomp with GCC)
                                     macOS    @rpath/libomp.dylib = torch/lib/libomp.dylib
                                     Windows  libiomp5md.dll, as torch loads it
```

No FINUFFT package is installed at run time.  The static archives'
symbols are hidden (FINUFFT's own visibility, and `--exclude-libs,ALL` on
Linux), so the dynamic table is the C ABI.

## The pin

`b3584138f9795c30bec5510f0c177deeb5aca7ee`, master on 2026-09-25.  The latest
release, 2.5.1, predates the consolidated CPU and CUDA CMake this build relies
on (one source tree, `FINUFFT_USE_CPU` and `FINUFFT_USE_CUDA` together, static
linking and DUCC0 through CMake), and 2.6 is not released.  The commit's
upstream CMake CI run completed green.

## FFT backend

DUCC0.  FFTW is excluded on two counts:

- it is GPL-2.0-or-later, and would be compiled into an MIT wheel;
- this library defines the FFTW guru symbols BART plans with
  (`src/csrc/substitute/fft.cpp`), so a FINUFFT linked against FFTW has its
  FFT bound to those.  Built that way the library crashes on the first
  transform.

The DUCC0 sources FINUFFT compiles are each `BSD-3-Clause OR
GPL-2.0-or-later` and are used under BSD-3-Clause; the notice is written from
their headers at configure time.

## Measurements

4-core Xeon at 2.1 GHz, GCC 14, tolerance 1e-3, upsampling 1.25, best of
five.  2D: a 256² image with 8 coils, 402 radial spokes of 512 samples.  3D: a
128³ image, 2 000 000 uniformly random samples.

bartorch through `linop.NUFFT`, this branch against `main` with the FINUFFT
2.5.1 wheel (FFTW), in ms:

| | threads | 2D forward | 2D adjoint | 3D forward | 3D adjoint |
| --- | --- | --- | --- | --- | --- |
| embedded, DUCC0 | 1 | 64.3 | 63.9 | 332.5 | 293.8 |
| 2.5.1 wheel, FFTW | 1 | 68.8 | 68.4 | 342.3 | 322.0 |
| embedded, DUCC0 | 4 | 22.2 | 19.3 | 86.3 | 83.5 |
| 2.5.1 wheel, FFTW | 4 | 17.8 | 17.9 | 93.9 | 84.7 |

The two agree to 1.9e-05 (2D) and 5.5e-06 (3D) in relative norm, and each is
1.53e-03 (2D) and 3.8e-04 (3D) from an explicit discrete Fourier sum over a
subset of samples.

FINUFFT alone at the pinned commit, both backends built from the same
checkout, same problems, in ms:

| | threads | 2D type 2 | 2D type 1 | 3D type 2 | 3D type 1 |
| --- | --- | --- | --- | --- | --- |
| DUCC0 | 1 | 61.4 | 64.0 | 344.2 | 273.4 |
| FFTW | 1 | 66.4 | 67.7 | 336.3 | 263.2 |
| DUCC0 | 4 | 21.9 | 20.2 | 84.1 | 74.4 |
| FFTW | 4 | 18.2 | 16.9 | 86.3 | 70.2 |

Single-threaded the two backends are level.  On four threads DUCC0 is about
20 per cent slower on the batched 2D problem, whose FFTs are eight 320²
transforms, and level in 3D.  The difference is the FFT's threading: DUCC0
runs its FFT on its own thread pool with the plan's thread count, not on
OpenMP, so it adds no OpenMP runtime.

## OpenMP

`cmake/openmp.cmake` defines `OpenMP::OpenMP_C` and `OpenMP::OpenMP_CXX` once.
FINUFFT's `find_package(OpenMP)` is answered from
`CMAKE_FIND_PACKAGE_REDIRECTS_DIR`, which takes precedence over CMake's Find
module, so on macOS and Windows it gets the targets bound to torch's runtime
rather than whatever the toolchain has.  The entry-point list is
`src/csrc/compat/libiomp5md.def`; clang's code for BART, FINUFFT and DUCC0
calls nothing outside it (checked with `nm -u` on a clang build), except
`__kmpc_dispatch_deinit`, which clang 19 and later emit after a dynamically
scheduled loop and which the libomp of torch 2.3 to 2.5 does not export.
`src/csrc/substitute/openmp.c` defines it inside the library, forwarding to the
runtime's own when the process has one.  torch 2.2 carries no libomp on macOS,
so the macOS floor is 2.3.

`tests/test_openmp.py` is the check: threaded torch and threaded FINUFFT in
one process, both import orders, the loaded OpenMP images listed.  On Linux
the one image found is torch's `libgomp.so.1`, which satisfies the library's
`NEEDED` entry as well.
