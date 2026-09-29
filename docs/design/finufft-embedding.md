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

DUCC0, in every wheel.  FFTW itself is excluded: it is GPL-2.0-or-later, and
would be compiled into an MIT wheel.  `BARTORCH_FINUFFT_FFT=MKL` builds
FINUFFT's FFTW path against oneMKL's FFTW3 interface instead, on x86-64 Linux
only; [the comparison below](#fft-inside-finufft-ducc0-and-onemkl) is why that is an option
and not the default.

The substitution BART plans with (`src/csrc/substitute/fft.cpp`) defines its
FFTW functions as `bartorch_fftwf_*`, through `src/csrc/compat/fftw3.h`, so
FFTW's names in the library resolve to whatever FINUFFT is linked against: a
plan FINUFFT makes with `fftwf_plan_many_dft` is executed and destroyed by the
library that made it.  With both under one name, the static link binds
FINUFFT's `fftwf_execute_dft` to the substitution and the first transform
crashes.

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

## FFT inside FINUFFT: DUCC0 and oneMKL

### Where transforms are computed

| Path | Engine | Linux CPU wheel | macOS CPU wheel | Windows CPU wheel | Linux CUDA wheel |
| --- | --- | --- | --- | --- | --- |
| BART's `fft` and every `md_` FFT on the host | `src/csrc/substitute/fft.cpp`: MKL DFTI from the process, else pocketfft | DFTI from torch's MKL, or from the `mkl` extra | pocketfft | pocketfft | as Linux CPU |
| BART on a device | cuFFT, through BART's `fft-cuda.c` | -- | -- | -- | `libcufft` from the nvidia wheels |
| The grid transforms' normal, and the paired Toeplitz kernels | cuFFT with LTO callbacks; cuFFTDx where built with MathDx | -- | -- | -- | as built |
| FINUFFT on the host | DUCC0, compiled in | DUCC0 | DUCC0 | DUCC0 | DUCC0 |
| cuFINUFFT | cuFFT | -- | -- | -- | `libcufft` |

Read off the built library and from the dependencies it loads: `libbartorch`
imports no FFT library on any platform (`readelf -d`, `otool -L`, the PE import
table), carries DUCC0's and pocketfft's code as hidden symbols, and reaches
DFTI through the table `_backend.py` fills.  torch's `libtorch_cpu.so` exports
DFTI on Linux x86-64; `torch_cpu.dll` exports no MKL symbol at all, and macOS
has no MKL.  No torch library exports MKL's FFTW3 interface: it is in
`libmkl_rt` from the `mkl` package alone.

### What FINUFFT's two paths compute

FINUFFT at the pin calls the FFT once per batch of `batchSize` transforms, from
`execute.hpp` between spreading and deconvolution and outside any OpenMP region
of its own, with `opts.nthreads`.

- `FINUFFT_USE_DUCC0` calls `ducc0::c2c` over the batch.  For a 2D or 3D type
  1 or 2 it transforms one axis in full and only the rows of the others that
  hold modes: roughly `(1 + 1/σ)/2` of the work in 2D and `(1 + 1/σ + 1/σ²)/3`
  in 3D.  Its threads are DUCC0's own pool of `std::thread`s, not OpenMP.
- Otherwise it includes `fftw3.h` and calls `fftw{f}_plan_many_dft` (one
  plan per direction, in place, the whole batch as `howmany`, `FFTW_ESTIMATE`
  by default), `fftw{f}_execute_dft`, `fftw{f}_destroy_plan`,
  `fftw{f}_init_threads` and `fftw{f}_plan_with_nthreads` under `_OPENMP`,
  and `forget_wisdom`, `cleanup` and `cleanup_threads`.  No guru interface, no
  wisdom it depends on, and the full transform on every axis.

oneMKL's `libmkl_rt` exports all sixteen, single and double precision; a
build linked against it leaves each of them undefined in `libbartorch` and
bound to `libmkl_rt` (`nm -D --undefined-only`).  So the MKL path does more
arithmetic than the DUCC0 path and still finishes first where it does.

### Method

`finufftf_*` on single-precision data, as `src/csrc/substitute/finufft.c`
calls it; tolerance 1e-3 and σ = 1.25 (the library's defaults) and σ = 2 (BART's
own, and the calibrating tools'); four threads unless stated.  2D: golden-angle
radial, `π/2 · N` spokes of `2N` samples; 3D: a kooshball of `N³/2` samples.
Batches of 1, 8 and 32 transforms in 2D (coils, and coils times four subspace
coefficients) and 1, 4 and 8 in 3D.  Each case in its own process, the three
builds interleaved; plan, `setpts`, first and best of five (three in 3D)
executions, FINUFFT's own FFT timer, peak RSS, and the error against an
explicit sum on 16 outputs.  FINUFFT from this pin three ways with GCC and
`-march=x86-64`: DUCC0, oneMKL 2026.1 (`mkl-devel`) through its FFTW3
interface, and Ubuntu's FFTW 3.3.10 as a reference.  4-core Xeon at 2.1 GHz
with AVX-512.

### Results

Best execution in ms, forward (type 2) and adjoint (type 1) alike:

| | N | samples | transforms | σ | DUCC0 | oneMKL | FFTW | oneMKL / DUCC0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D | 256 | 0.21 M | 8 | 1.25 | 18.7 | 15.4 | 16.6 | 0.82 |
| 2D | 256 | 0.21 M | 8 | 2 | 21.2 | 13.9 | 22.0 | 0.66 |
| 2D | 256 | 0.21 M | 32 | 1.25 | 83.7 | 68.6 | 71.6 | 0.82 |
| 2D | 512 | 0.82 M | 8 | 1.25 | 81.9 | 68.6 | 72.5 | 0.84 |
| 2D | 512 | 0.82 M | 32 | 2 | 339.8 | 269.0 | 430.1 | 0.79 |
| 3D | 128 | 1.05 M | 8 | 1.25 | 360.6 | 310.7 | 331.8 | 0.86 |
| 3D | 192 | 3.54 M | 4 | 1.25 | 766.8 | 662.8 | 752.0 | 0.86 |
| 3D | 192 | 3.54 M | 4 | 2 | 1682.7 | 1518.8 | 2925.8 | 0.90 |
| 3D | 256 | 8.39 M | 1 | 1.25 | 503.9 | 448.8 | 498.7 | 0.89 |
| 3D | 256 | 8.39 M | 4 | 2 | 3906.8 | 3815.9 | 8244.6 | 0.98 |

Over all 76 cases oneMKL takes 0.63 to 0.86 of DUCC0's time in 2D and 0.8 to
1.0 in 3D.  The FFT is 15 to 40 per cent of a DUCC0 transform and 2 to 25 per
cent of a oneMKL one; the rest is spreading, which the backend does not touch.
Plan creation is 8 ms for oneMKL against under a millisecond for DUCC0, once
per operator; the first execution costs what a steady one does for both.  Peak
memory is the same to within the 10 MB `libmkl_rt` maps.  The error against the
explicit sum agrees to three digits in every case, at tolerances from 1e-3 to
1e-6.

The margin depends on the machine rather than on the transform alone.  With
oneMKL held to AVX2 (`MKL_ENABLE_INSTRUCTIONS=AVX2`), the 3D σ = 2 case above
is level with DUCC0 and the 2D ones keep 0.75 to 0.85.  And the build's own
instruction set costs more than the FFT library: FINUFFT at `-march=x86-64-v3`
takes 0.6 to 0.85 of its `-march=x86-64` time with DUCC0, most of it in
spreading, and oneMKL at v3 still takes 0.7 to 0.9 of DUCC0 at v3.  The wheel
keeps `x86-64` compiled in and chooses a newer level at run time; see
[Instruction-set levels](#instruction-set-levels).

### Numerics

The library built each way, the same inputs: a 128² image with eight coils
over 201 radial spokes, and a 48³ volume.  Relative difference between the
two builds, and between two runs of one build:

| | DUCC0 against oneMKL | DUCC0, run to run | oneMKL, run to run |
| --- | --- | --- | --- |
| forward, 2D and 3D | 1.5e-07 to 5.5e-07 | 0 | 0 |
| adjoint, 2D and 3D | 7.0e-07 to 2.2e-06 | 0 to 6.2e-08 | 0 to 6.0e-08 |
| `A^H A` as the pair | 7.0e-07 | 0 | 0 |
| `A^H A` as a Toeplitz convolution | 9.4e-08 | 5.7e-08 | 5.4e-08 |
| `pics`, 30 iterations, Toeplitz | 7.5e-04 | 1.7e-04 | 1.5e-04 |
| `pics`, 30 iterations, transform pair | 7.0e-05 | 2.3e-05 | 2.7e-05 |

Both are three orders of magnitude inside the 1e-3 tolerance the plans are
made with, and the run-to-run differences are the threaded spreading's
summation order, the same for both.  The whole suite passes on each build
(the explicit-sum, adjoint, Toeplitz, reconstruction and batched tests among
it).

### Threading

The FFT runs outside FINUFFT's parallel regions, so it takes the plan's
threads itself.  Both backends scale with them: in 3D at 192³, 1, 2 and 4
threads give oneMKL's FFT 56, 28 and 13 ms and DUCC0's 136, 89 and 56 ms.
There is no nesting to avoid on this path, and neither backend is held to one
thread.  `libmkl_rt` chooses its threading layer on first use and, with GNU's
runtime already loaded by the library, takes its GNU layer: its FFT runs on the
OpenMP threads that spread, and no `libiomp5` is loaded
(`tests/test_openmp.py`).  Nothing sets `MKL_THREADING_LAYER` or
`MKL_NUM_THREADS`.

DUCC0's threads are a pool of its own beside OpenMP's, and they start while
the OpenMP threads that have just finished spreading are still spinning for
their next region.  On four cores that contention is most of DUCC0's FFT time
in 2D: with `OMP_WAIT_POLICY=passive` its FFT takes a third to a half as long
(256², eight transforms, σ = 2: 9.9 ms against 3.0) and the whole transform 5
to 25 per cent less, which closes about half the gap to oneMKL; oneMKL's times
do not change.  In 3D, where one FFT outlasts the spin, the policy changes
neither.  The library does not set the policy: it is process-wide, and it
applies to torch's and BART's regions too.

Called from inside a parallel region, oneMKL runs its FFT on the one thread it
is given while DUCC0's pool starts its own threads anyway: two such
transforms of 384² with eight coils, four threads each on four cores, take
151 ms on oneMKL and 254 ms on DUCC0.  The library does not call FINUFFT
from inside a parallel region on the host.

### Packaging

oneMKL's FFTW3 interface is in the `mkl` wheel alone (224 MB, 672 MB installed,
with `intel-openmp` and `tbb`), so a FINUFFT that calls it makes that wheel a
dependency of every install, and a second MKL beside the one torch links on
Linux.  On Windows the `mkl` extra is not offered and torch exports no MKL,
so it would be a new runtime dependency with nothing to share; macOS has none.
The CPU wheels therefore keep DUCC0, which adds nothing to load, and
`BARTORCH_FINUFFT_FFT=MKL` is for a source build on x86-64 Linux where oneMKL
is installed: the library then names `libmkl_rt` in `NEEDED` with the build's
MKL directory on its run path, and with the `mkl` extra the same `libmkl_rt`
serves BART's DFTI table (`tests/test_finufft.py`).  cuFINUFFT and the CUDA
paths are unchanged.

## Instruction-set levels

FINUFFT has no dispatch of its own on the host: xsimd picks its vector width
when FINUFFT is compiled, from `FINUFFT_ARCH_FLAGS`, and the whole transform --
spreading, interpolation, deconvolution and DUCC0's FFT -- is compiled for that
one level.  A wheel has to run on any x86-64 processor, so its baseline is
`-march=x86-64`, SSE2.

So the baseline is compiled into the library, and each level
`BARTORCH_FINUFFT_SIMD` lists -- `x86-64-v2` (SSE4.2), `x86-64-v3` (AVX2 and
FMA) and `x86-64-v4` (AVX-512) by default -- is compiled again as a module
beside it: `libbartorch_finufft_x86_64_v3.so` on Linux, `.dll` on Windows.  macOS arm64
has one level and builds none.  At the first plan the library tests the
processor (`__builtin_cpu_supports`, which also checks that the operating
system saves the vector state) and opens the newest module it runs;
`BARTORCH_FINUFFT_SIMD` in the environment or `_finufft.use_simd` chooses
another, and a plan keeps the entry points it was made with, so a change of
level never meets a live plan.

A module is a library of its own rather than a second copy of FINUFFT linked
into this one, because two copies in one link share whatever the compiler
emits as a weak or COMDAT definition -- FINUFFT's templates, xsimd's and
DUCC0's inline functions, the standard library's -- and the linker keeps one of
each, which may be the AVX2 one on a processor without AVX2.  A module is
opened with `RTLD_LOCAL` (on Windows every DLL resolves its own imports), links
its C++ runtime statically as the library does, and exports FINUFFT's C API
and nothing else (`cmake/finufft_module.map`), so nothing it defines is seen by
the library or by a `finufft` package loaded beside it.  It links the same
OpenMP runtime as the library, which the process has already loaded.

The module's targets are FINUFFT's own compiled again: `cmake/finufft.cmake`
reads the sources, definitions, options and dependencies off the targets
FINUFFT's CMake defined (`finufft`, `finufft_f32`, `ducc0`) and replaces
`-march=x86-64` with the level's.  A pin whose targets stop carrying that flag
stops the configure rather than building a module at the baseline.
`finufft_common` holds the kernel's parameters, is compiled with no arch flag,
and is linked into each module as it is.

Best execution in ms, DUCC0, the transforms of the method above, four threads
on the AVX-512 Xeon, all four builds in one interleaved run:

| | N | transforms | σ | tolerance | x86-64 | v2 | v3 | v4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D type 2 | 256 | 8 | 1.25 | 1e-3 | 24.4 | 12.5 | 11.5 | 12.5 |
| 2D type 1 | 256 | 8 | 2 | 1e-3 | 19.8 | 16.4 | 16.8 | 19.5 |
| 2D type 2 | 256 | 8 | 1.25 | 1e-6 | 23.6 | 13.1 | 13.9 | 14.7 |
| 2D type 2 | 512 | 8 | 1.25 | 1e-3 | 85.5 | 48.1 | 52.6 | 48.0 |
| 2D type 1 | 512 | 8 | 2 | 1e-3 | 91.1 | 64.5 | 67.1 | 69.2 |
| 3D type 2 | 192 | 4 | 1.25 | 1e-3 | 755.6 | 589.0 | 591.0 | 607.4 |
| 3D type 1 | 192 | 4 | 2 | 1e-3 | 1796.7 | 1630.9 | 1592.2 | 1624.9 |
| 3D type 1 | 192 | 4 | 1.25 | 1e-6 | 933.7 | 749.0 | 772.5 | 756.8 |

Most of the gain is already at v2: 0.51 to 0.83 of the baseline's time in 2D
and 0.77 to 0.92 in 3D, which points to the SSE4.1 instructions xsimd
reaches for in the kernel's evaluation rather than from the vector width.  On
this processor v3 and v4 sit within the run-to-run scatter of v2, a few per
cent either way; on one whose AVX2 or AVX-512 units are wider relative to its
SSE ones they need not.  All three are built, so each processor runs the
newest level it has.  The error against the explicit sum is the same at every
level to three digits.  Through the library, a 256² eight-coil forward and
adjoint of 403 spokes takes 46 ms at the baseline and 29 ms at v3, and the two
agree to 6e-07 forward and 1e-06 adjoint.

What it costs: FINUFFT and DUCC0 compiled once more per level, and about 5 MB
per level, stripped, in the wheel.

