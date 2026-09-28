# Development prerequisites

| Tool | Requirement |
| --- | --- |
| Git | With submodule support; BART and FINUFFT are the submodules `external/bart` and `external/finufft`.  FINUFFT's build also fetches the xsimd, POET and DUCC0 versions it pins, and CCCL for a CUDA build; `CPM_SOURCE_CACHE` keeps them between builds |
| Python | 3.10 or later, with PyTorch installed first as described in {doc}`../user/installation` |
| C and C++ compiler | clang, or GCC 14 or later.  BART's nested functions are compiled as Blocks by clang and as heap trampolines (`-ftrampoline-impl=heap`) by GCC 14; CMake rejects older GCC at configuration |
| CMake | 3.25 or later, with Ninja or Make |
| OpenMP | Linux: the runtime matching the compiler, for example `libomp-dev` with clang on Debian and Ubuntu.  macOS: `brew install libomp`, for `omp.h` only; the library links against PyTorch's runtime.  Without it the build fails unless `-DBARTORCH_OPENMP=OFF` asks for a single-threaded library |
| CUDA (optional) | A CUDA toolkit with `nvcc` for `-DBARTORCH_CUDA=ON`, 12.1 or later for compute capability 9.0; a CUDA build of PyTorch does not include a compiler |

FINUFFT and cuFINUFFT are compiled into the library from `external/finufft`;
no FINUFFT package is installed or used.  `pip install mkl deepinv` enables
the tests that need MKL and the DeepInverse adapter.

On Windows the compiler is clang from MSYS2's CLANG64 environment, with the
packages `mingw-w64-clang-x86_64-{clang,cmake,ninja,llvm-openmp,llvm-tools}`,
`clang64/bin` on `PATH` and `CMAKE_GENERATOR=Ninja`.  The build links against
the OpenMP runtime PyTorch installs rather than MSYS2's, and loads into the
official CPython with nothing from MSYS2 at run time.  There is no CUDA build on
Windows.
