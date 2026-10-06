cmake_minimum_required(VERSION 3.25)
project(bartorch LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# What is built here
#
# One shared library, libbartorch, holding every BART translation unit from
# the submodule plus the handful of units that give BART its FFT, BLAS,
# LAPACK and in-memory I/O.  Nothing links against PyTorch or Python: the
# host reaches the library through the C ABI in src/csrc/include/bartorch.h.
#
# BART is written against GNU C.  The nested functions it uses become Blocks
# under clang, with the vendored runtime on platforms that lack one, which
# keeps the library loadable without an executable stack.
# ---------------------------------------------------------------------------

set(BART_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/bart" CACHE PATH "BART source tree")
set(BART_SRC "${BART_ROOT}/src")

if(NOT EXISTS "${BART_SRC}/bart.c")
    message(FATAL_ERROR "BART sources not found at ${BART_SRC}; run `git submodule update --init`")
endif()

option(BARTORCH_OPENMP "Build BART and FINUFFT with OpenMP" ON)
option(BARTORCH_CUDA "Build BART's CUDA kernels" OFF)
set(BARTORCH_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING
    "Architectures to emit SASS for; PTX for the last one is added for newer cards")

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# BART checks its arguments with assert and src/csrc/abi/api.c puts those on its error
# path, so NDEBUG must not remove them.
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Do not let the compiler fold a multiply into an add.
#
# `vecops.c` has one kernel behind `axpy`, `xpay` and `axpbz`, and
# `flpmath.c`'s `md_axpy` another: `dst[i] = a1 * src1[i] + a2 * src2[i]`.
# clang's default is `-ffp-contract=on` and GCC's is `fast`, so on a target
# with a fused multiply-add -- arm64 always, x86-64 as soon as `-march` allows
# it -- the product is not rounded before the add, and BART computes something
# slightly different from what it computes elsewhere.
#
# That would be nobody's business but the compiler's, except that this package
# claims to reproduce BART's iterations to the bit, and it cannot fold across
# two torch kernels.  Measured on arm64, the primal-dual solver drifted a few
# times 1e-5 from the library over twelve steps, and every solve carrying a
# quadratic weight drifted too, because `normaleq_l2_apply` adds it with
# `md_axpy`.
#
# So the arithmetic is pinned instead: the same two roundings on every target,
# which is what x86-64 was already doing and what every comparison in the
# suite was checked against.  It costs the fused instruction in kernels that
# are memory-bound anyway.
add_compile_options(-ffp-contract=off)

# ---------------------------------------------------------------------------
# Compiler mode
# ---------------------------------------------------------------------------
set(BART_C_OPTIONS "")
set(BART_C_DEFS "")
set(BART_LINK_OPTIONS "")
set(BART_EXTRA_LIBS "")

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    list(APPEND BART_C_OPTIONS -fblocks)
    set(BARTORCH_NESTED "clang-blocks")
    if(WIN32)
        # clang declares the runtime dllimport on Windows, and the linker
        # resolves an import to a definition in the same image only when that
        # definition is already present: objects are, archive members are not
        # pulled in for it.
        add_library(blocksruntime OBJECT
            external/blocksruntime/runtime.c
            external/blocksruntime/data.c)
        list(APPEND BART_EXTRA_LIBS $<TARGET_OBJECTS:blocksruntime>)
    elseif(NOT APPLE)
        add_library(blocksruntime STATIC
            external/blocksruntime/runtime.c
            external/blocksruntime/data.c)
        target_include_directories(blocksruntime PUBLIC external/blocksruntime)
        list(APPEND BART_EXTRA_LIBS blocksruntime)
    endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    # GCC turns BART's nested functions into trampolines.  On the stack those
    # need an executable stack, which glibc 2.41 refuses to dlopen; GCC 14
    # places them on the heap instead, so the library loads anywhere.
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS 14)
        message(FATAL_ERROR
            "GCC ${CMAKE_C_COMPILER_VERSION} would need an executable stack for BART's nested "
            "functions.  Use GCC 14 or newer, which has -ftrampoline-impl=heap, or clang.")
    endif()
    list(APPEND BART_C_OPTIONS -ftrampoline-impl=heap
        -Wno-vla-parameter -Wno-nonnull -Wno-maybe-uninitialized
        -include assert.h)
    list(APPEND BART_LINK_OPTIONS -Wl,-z,noexecstack)
    set(BARTORCH_NESTED "gcc-heap-trampolines")
else()
    message(FATAL_ERROR "BART needs clang or GCC 14+; ${CMAKE_C_COMPILER_ID} cannot compile it")
endif()

# BART's STL writer declares its triangle with `_Float32`, the interchange type
# C23 names and glibc has provided for years.  Apple's clang has neither, and
# BART already says what it means there: the same declaration under
# `__EMSCRIPTEN__` uses plain `float`.  So where the type is missing, spell it
# the way BART's own fallback does.  Tested rather than guessed, because
# defining it where it does exist collides with the system header.
include(CheckCSourceCompiles)
# The data model, reported in build_info: long is 64 bits on LP64 (Linux,
# macOS) and 32 on LLP64 (Windows), and nothing BART or the ABI stores in an
# array's shape depends on it.
include(CheckTypeSize)
check_type_size(long BARTORCH_SIZEOF_LONG LANGUAGE C)
# Through <complex.h>, because that is where BART's translation units get it:
# glibc typedefs it in <bits/floatn-common.h>, which <complex.h> and <math.h>
# pull in and <stdio.h> does not on every glibc, and clang does not offer it as
# a keyword.
check_c_source_compiles(
    "#include <complex.h>\n_Float32 x; int main(void) { return (int)x; }"
    BARTORCH_HAS_FLOAT32)
if(NOT BARTORCH_HAS_FLOAT32)
    list(APPEND BART_C_DEFS _Float32=float)
endif()

# The library exports a C ABI and exchanges no C++ objects with the process it
# is loaded into, so its GCC runtime can be its own.  Linked dynamically it
# would carry the toolchain's floor into every target system -- a GCC 14 build
# asks for GCC_14.0.0 from libgcc_s, which no released distribution has -- and
# that is a load-time failure, not a fallback.  Static leaves libc and libgomp.
#
# On Windows the same goes for everything the MinGW toolchain would otherwise
# bring as a DLL -- winpthreads and the compiler's runtime -- because Python
# does not search PATH for a library's dependencies.  What is left is the
# Universal C Runtime and the system's own libraries.
if(WIN32)
    list(APPEND BART_LINK_OPTIONS -static)
elseif(NOT APPLE)
    # --exclude-libs keeps what those runtimes and FINUFFT's archives define
    # out of the dynamic symbol table, which is then the bartorch_* ABI and
    # BART's own symbols: nothing another library in the process could bind.
    list(APPEND BART_LINK_OPTIONS -static-libgcc -static-libstdc++ -Wl,--exclude-libs,ALL)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# CUDA
#
# BART's kernels are compiled by nvcc and the runtime libraries are linked
# dynamically, so the wheel carries device code and nothing else: cudart,
# cuFFT, cuBLAS and cuSOLVER come from the nvidia wheels torch already
# depends on.
#
# CUDA_GET_CUDA_DEVICE_NUM makes BART ask the driver whether a pointer is on
# a device rather than consult its own allocation table, which is what lets a
# tensor the host allocated be passed straight in.
# ---------------------------------------------------------------------------
if(BARTORCH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    list(APPEND BART_C_DEFS USE_CUDA USE_GPU CUDA_GET_CUDA_DEVICE_NUM)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

    set(_arch "")
    foreach(_a ${BARTORCH_CUDA_ARCHITECTURES})
        list(APPEND _arch "${_a}-real")
    endforeach()
    list(GET BARTORCH_CUDA_ARCHITECTURES -1 _newest)
    list(APPEND _arch "${_newest}-virtual")
    set(BARTORCH_CUDA_ARCH_SPEC "${_arch}")

    message(STATUS "CUDA ${CUDAToolkit_VERSION}: architectures ${BARTORCH_CUDA_ARCH_SPEC}")
endif()

include(cmake/openmp.cmake)
if(OpenMP_C_FOUND)
    list(APPEND BART_EXTRA_LIBS OpenMP::OpenMP_C)
endif()

include(cmake/finufft.cmake)

# ---------------------------------------------------------------------------
# Files BART's Makefile generates: the version string and the tool table.
# They are written under the build tree so the submodule stays untouched.
# ---------------------------------------------------------------------------
set(BART_GEN "${CMAKE_BINARY_DIR}/bart_gen")
file(MAKE_DIRECTORY "${BART_GEN}/misc")

file(READ "${BART_ROOT}/version.txt" BART_VERSION)
string(STRIP "${BART_VERSION}" BART_VERSION)
file(WRITE "${BART_GEN}/misc/version.inc" "\"${BART_VERSION}\"\n")

file(STRINGS "${BART_ROOT}/Makefile" _tool_lines REGEX "^T(BASE|FLP|NUM|IO|RECO|CALIB|MRI|SIM|NN|MOTION)\\+=")
set(_mainlist "")
set(_all_tools "")
foreach(_cat BASE FLP NUM IO RECO CALIB MRI SIM NN MOTION)
    set(_tools "")
    foreach(_line ${_tool_lines})
        if(_line MATCHES "^T${_cat}\\+=(.*)$")
            string(REPLACE " " ";" _names "${CMAKE_MATCH_1}")
            foreach(_name ${_names})
                if(EXISTS "${BART_SRC}/${_name}.c")
                    list(APPEND _tools "${_name}")
                    list(APPEND _all_tools "${_name}")
                endif()
            endforeach()
        endif()
    endforeach()
    list(SORT _tools)
    string(REPLACE ";" ", " _joined "${_tools}")
    if(_joined)
        string(APPEND _mainlist "#define MAIN_${_cat} ${_joined}, ()\n")
    else()
        string(APPEND _mainlist "#define MAIN_${_cat} ()\n")
    endif()
endforeach()
list(REMOVE_DUPLICATES _all_tools)
list(SORT _all_tools)
string(REPLACE ";" ", " _joined "${_all_tools}")
string(PREPEND _mainlist "#define MAIN_LIST ${_joined}, ()\n")
file(WRITE "${BART_GEN}/mainlist.inc" "${_mainlist}")
list(LENGTH _all_tools _ntools)
message(STATUS "BART ${BART_VERSION}: ${_ntools} tools, nested functions via ${BARTORCH_NESTED}")

# ---------------------------------------------------------------------------
# Sources: every module directory and every tool, minus what needs a
# library this build does not provide, and minus misc/memcfl.c which
# src/csrc/abi/memcfl.c replaces.
# ---------------------------------------------------------------------------
set(BART_SOURCES "")
# CONFIGURE_DEPENDS, because BART's sources are a glob and a submodule bump
# adds and removes files.  Without it the build system keeps the list it was
# configured with and `cmake --build` happily rebuilds a library that is
# missing a translation unit BART has since grown -- which is a link error at
# best and the old behaviour at worst.  The cost is a directory scan per
# build, over some hundreds of files.
file(GLOB _module_dirs LIST_DIRECTORIES true CONFIGURE_DEPENDS "${BART_SRC}/*")
foreach(_dir ${_module_dirs})
    if(IS_DIRECTORY "${_dir}")
        get_filename_component(_mod "${_dir}" NAME)
        # `win/` is BART's own port of mmap, fmemopen, getsubopt and the rest
        # of what the Windows C runtime lacks, and is built there alone.
        if(_mod STREQUAL "win" AND NOT WIN32)
            continue()
        endif()
        if(NOT _mod MATCHES "^(ismrm|lapacke)$")
            file(GLOB _srcs CONFIGURE_DEPENDS "${_dir}/*.c")
            list(APPEND BART_SOURCES ${_srcs})
        endif()
    endif()
endforeach()
file(GLOB _tool_srcs CONFIGURE_DEPENDS "${BART_SRC}/*.c")
list(APPEND BART_SOURCES ${_tool_srcs})

if(BARTORCH_CUDA)
    foreach(_dir calib motion noncart num simu wavelet)
        file(GLOB _cu CONFIGURE_DEPENDS "${BART_SRC}/${_dir}/*.cu")
        list(APPEND BART_SOURCES ${_cu})
    endforeach()
endif()
list(REMOVE_ITEM BART_SOURCES
    "${BART_SRC}/main.c"
    "${BART_SRC}/bbox.c"
    "${BART_SRC}/ismrmrd.c"
    "${BART_SRC}/misc/memcfl.c")

# Three things, and a file belongs to whichever it is.  `abi/` is the boundary:
# what the host calls, and what BART's environment asks of the host in return.
# `ops/` is what the host builds and drives -- the operators and the solve.
# `substitute/` is what runs in BART's place: its transforms, and the libraries
# it would otherwise have been linked against.
set(BARTORCH_SOURCES
    src/csrc/abi/api.c
    src/csrc/abi/cuda.c
    src/csrc/abi/host_reads.c
    src/csrc/abi/memcfl.c

    src/csrc/ops/ops.c
    src/csrc/ops/sense.c
    src/csrc/ops/grid.c
    src/csrc/ops/iter.c

    src/csrc/substitute/backend.c
    src/csrc/substitute/ref_blas.c
    src/csrc/substitute/cblas_shim.c
    src/csrc/substitute/critical.c
    src/csrc/substitute/lapacke_shim.c
    src/csrc/substitute/fft.cpp
    src/csrc/substitute/finufft.c
    src/csrc/substitute/nufft_finufft.c
    src/csrc/substitute/psf.c)
if(OpenMP_C_FOUND AND (WIN32 OR APPLE))
    list(APPEND BARTORCH_SOURCES src/csrc/substitute/openmp.c)
endif()

if(BARTORCH_CUDA)
    list(APPEND BARTORCH_SOURCES src/csrc/ops/kernels.cu src/csrc/ops/fft_callbacks.cu)

    # The passes around a volume's transform, which cuFFT links into the
    # transform when a plan is made (src/csrc/ops/fft_callbacks.cu).  It takes them as
    # LTO-IR at run time, so src/csrc/ops/fft_callbacks_lto.cu is compiled to that
    # alone and embedded in the library as an array.
    set(_lto_src "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ops/fft_callbacks_lto.cu")
    set(_lto_bin "${CMAKE_CURRENT_BINARY_DIR}/fft_callbacks_lto.fatbin")
    set(_lto_hdr "${CMAKE_CURRENT_BINARY_DIR}/fft_callbacks_lto.h")
    set(_lto_arch "")
    foreach(_a ${BARTORCH_CUDA_ARCHITECTURES})
        list(APPEND _lto_arch "-gencode=arch=compute_${_a},code=lto_${_a}")
    endforeach()
    set(_lto_ccbin "")
    if(CMAKE_CUDA_HOST_COMPILER)
        set(_lto_ccbin -ccbin "${CMAKE_CUDA_HOST_COMPILER}")
    endif()
    add_custom_command(
        OUTPUT "${_lto_hdr}"
        COMMAND "${CMAKE_CUDA_COMPILER}" ${_lto_ccbin} -std=c++17 -dc -fatbin ${_lto_arch}
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc" -o "${_lto_bin}" "${_lto_src}"
        COMMAND "${CMAKE_COMMAND}" "-DIN=${_lto_bin}" "-DOUT=${_lto_hdr}"
                -DNAME=bartorch_fft_callbacks_lto
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
        DEPENDS "${_lto_src}" "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ops/coset.cuh"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ops/grid.cuh"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
        VERBATIM)
    list(APPEND BARTORCH_SOURCES "${_lto_hdr}")
endif()

# The paired Toeplitz kernels (src/csrc/ops/paired.cu) are cuFFTDx, which is headers
# only and comes with MathDx: BARTORCH_MATHDX_DIR names its nvidia/mathdx
# directory -- `pip install nvidia-mathdx` puts one in site-packages -- and
# the kernels are compiled for the cubic grid sizes in BARTORCH_PAIRED_SIZES,
# tuned for the newest architecture listed.  The default is every size from
# 192 to 512 that cuFFTDx transforms at cuFFT's speed or near it, plus 32 for
# the tests; 340, whose factor 17 cuFFTDx takes through Bluestein's algorithm
# at 2.6 times cuFFT's time, is left to cuFFT.  The pass along x is compiled
# again for each number of coefficients in BARTORCH_PAIRED_RANKS, one
# translation unit per rank (src/csrc/ops/paired_rank.cu): 1 is the function
# without a subspace.  Without MathDx the library has no paired kernels and
# every size and rank is served by the transforms it has otherwise.
set(BARTORCH_MATHDX_DIR "" CACHE PATH "MathDx's nvidia/mathdx directory, for the paired Toeplitz kernels")
set(BARTORCH_PAIRED_SIZES "32;192;200;224;240;256;288;300;320;360;384;400;448;480;512" CACHE STRING "Cubic grid sizes the paired Toeplitz kernels are compiled for")
set(BARTORCH_PAIRED_RANKS "1;4;5;6;7;8;9;10" CACHE STRING "Numbers of coefficients the paired Toeplitz kernels are compiled for")
if(BARTORCH_CUDA AND BARTORCH_MATHDX_DIR)
    if(NOT EXISTS "${BARTORCH_MATHDX_DIR}/include/cufftdx.hpp")
        message(FATAL_ERROR "BARTORCH_MATHDX_DIR=${BARTORCH_MATHDX_DIR} has no include/cufftdx.hpp")
    endif()
    set(_sizes "")
    foreach(_n ${BARTORCH_PAIRED_SIZES})
        string(APPEND _sizes " X(${_n})")
    endforeach()
    set(_ranks "")
    set(_rank_sources "")
    foreach(_r ${BARTORCH_PAIRED_RANKS})
        if(NOT _r MATCHES "^[1-9][0-9]*$")
            message(FATAL_ERROR "BARTORCH_PAIRED_RANKS: ${_r} is not a number of coefficients")
        endif()
        string(APPEND _ranks " X(${_r})")
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/paired_rank_${_r}.cu.in"
            "/* Generated from BARTORCH_PAIRED_RANKS; see CMakeLists.txt. */\n#define BARTORCH_PAIRED_RANK ${_r}\n#include \"ops/paired_rank.cu\"\n")
        configure_file("${CMAKE_CURRENT_BINARY_DIR}/paired_rank_${_r}.cu.in" "${CMAKE_CURRENT_BINARY_DIR}/paired_rank_${_r}.cu" COPYONLY)
        list(APPEND _rank_sources "${CMAKE_CURRENT_BINARY_DIR}/paired_rank_${_r}.cu")
    endforeach()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/paired_sizes.h.in"
        "/* Generated from BARTORCH_PAIRED_SIZES and BARTORCH_PAIRED_RANKS; see CMakeLists.txt. */\n#define BARTORCH_PAIRED_SIZES(X)${_sizes}\n#define BARTORCH_PAIRED_RANKS(X)${_ranks}\n")
    configure_file("${CMAKE_CURRENT_BINARY_DIR}/paired_sizes.h.in" "${CMAKE_CURRENT_BINARY_DIR}/paired_sizes.h" COPYONLY)
    list(GET BARTORCH_CUDA_ARCHITECTURES -1 _paired_arch)
    set(_paired_archs "")
    foreach(_a ${BARTORCH_CUDA_ARCHITECTURES})
        list(APPEND _paired_archs "${_a}-real")
    endforeach()
    # Compiled on their own and for the listed architectures only: PTX of a
    # template instantiated once per grid size is megabytes, and a card newer
    # than the list is served by the transforms the library has otherwise.
    add_library(bartorch_paired OBJECT src/csrc/ops/paired.cu ${_rank_sources})
    set_target_properties(bartorch_paired PROPERTIES
        CUDA_ARCHITECTURES "${_paired_archs}"
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(bartorch_paired PRIVATE
        "${BARTORCH_MATHDX_DIR}/include"
        "${BARTORCH_MATHDX_DIR}/external/cutlass/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc"
        "${BART_SRC}"
        "${BART_GEN}"
        "${BART_GEN}/misc"
        "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_definitions(bartorch_paired PRIVATE ${BART_C_DEFS} _GNU_SOURCE
        BARTORCH_PAIRED_SM=${_paired_arch}0)
    target_compile_options(bartorch_paired PRIVATE -Xcompiler=-fPIC -Xfatbin=-compress-all)
    list(APPEND BART_C_DEFS BARTORCH_PAIRED)
    message(STATUS "Paired Toeplitz kernels: cuFFTDx from ${BARTORCH_MATHDX_DIR}, sizes ${BARTORCH_PAIRED_SIZES}, ranks ${BARTORCH_PAIRED_RANKS}")
endif()

add_library(bartorch SHARED ${BART_SOURCES} ${BARTORCH_SOURCES})

if(TARGET bartorch_paired)
    target_sources(bartorch PRIVATE $<TARGET_OBJECTS:bartorch_paired>)
endif()

# BART's NUFFT operator keeps its implementation under another name, so
# src/csrc/substitute/nufft_finufft.c can answer with FINUFFT's transform and still hand back
# anything it does not serve.  The rename applies to these files, not to their
# callers, so every tool that builds a NUFFT goes through the substitution and
# nufft.c's own internal uses stay with BART's operator.
set_source_files_properties("${BART_SRC}/noncart/nufft.c" PROPERTIES
    COMPILE_DEFINITIONS "nufft_create=bart_nufft_create;nufft_create2=bart_nufft_create2;nufft_get_psf_dims=bart_nufft_get_psf_dims;nufft_get_psf=bart_nufft_get_psf;nufft_get_psf2=bart_nufft_get_psf2;nufft_update_psf=bart_nufft_update_psf;nufft_update_psf2=bart_nufft_update_psf2;nufft_update_traj=bart_nufft_update_traj;compute_psf=bart_compute_psf;compute_psf2=bart_compute_psf2;compute_psf2_decomposed=bart_compute_psf2_decomposed")

# The SENSE operators keep their implementations under another name, so
# src/csrc/ops/sense.c can walk the coils a slab at a time and still hand back BART's
# own chain for an arrangement it cannot serve.
if(FINUFFT_ARCH_FLAGS MATCHES "^-march=(.+)$")
    set(_finufft_simd_base "${CMAKE_MATCH_1}")
else()
    set(_finufft_simd_base "default")
endif()
set(_finufft_defs "BARTORCH_FINUFFT_VERSION=\"${BARTORCH_FINUFFT_VERSION}\""
    "BARTORCH_FINUFFT_SIMD_BASE=\"${_finufft_simd_base}\""
    "BARTORCH_FINUFFT_FFT_NAME=\"${BARTORCH_FINUFFT_FFT_NAME}\"")
foreach(_level IN LISTS BARTORCH_FINUFFT_SIMD_LEVELS)
    string(REGEX REPLACE "^x86-64-" "BARTORCH_FINUFFT_SIMD_" _def "${_level}")
    string(TOUPPER "${_def}" _def)
    list(APPEND _finufft_defs "${_def}")
endforeach()
foreach(_level IN LISTS BARTORCH_FINUFFT_MKL_LEVELS)
    string(REGEX REPLACE "^x86-64-" "BARTORCH_FINUFFT_MKL_" _def "${_level}")
    string(TOUPPER "${_def}" _def)
    list(APPEND _finufft_defs "${_def}")
endforeach()
set_source_files_properties(src/csrc/substitute/finufft.c PROPERTIES
    COMPILE_DEFINITIONS "${_finufft_defs}")
string(REPLACE ";" "+" _finufft_simd_info "${_finufft_simd_base};${BARTORCH_FINUFFT_SIMD_LEVELS}")
string(REGEX REPLACE "\\+$" "" _finufft_simd_info "${_finufft_simd_info}")
string(REPLACE ";" "+" _finufft_mkl_info "${BARTORCH_FINUFFT_MKL_LEVELS}")

set_source_files_properties("${BART_SRC}/sense/modelnc.c" PROPERTIES
    COMPILE_DEFINITIONS "sense_nc_init=bart_sense_nc_init")

set_source_files_properties("${BART_SRC}/sense/model.c" PROPERTIES
    COMPILE_DEFINITIONS "sense_init=bart_sense_init")

set_source_files_properties("${BART_SRC}/noncart/precond.c" PROPERTIES
    COMPILE_DEFINITIONS "nufft_precond_create=bart_nufft_precond_create")

# The few entry points BART reads element by element rather than through md_
# operations; src/csrc/abi/host_reads.c answers them over a host copy when a tool is
# handed memory on a card.
set_source_files_properties("${BART_SRC}/misc/mri2.c" PROPERTIES
    COMPILE_DEFINITIONS "estimate_im_dims=bart_estimate_im_dims;estimate_fast_sq_im_dims=bart_estimate_fast_sq_im_dims")

set_source_files_properties("${BART_SRC}/sense/optcom.c" PROPERTIES
    COMPILE_DEFINITIONS "estimate_scaling_norm=bart_estimate_scaling_norm")

target_include_directories(bartorch PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/compat"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/pocketfft"
    "${BART_SRC}"
    "${BART_GEN}"
    "${BART_GEN}/misc"
    "${CMAKE_CURRENT_BINARY_DIR}")

target_compile_definitions(bartorch PRIVATE
    USE_LOG_BACKEND
    REDEFINE_PRINTF_FOR_TRACE
    NO_PNG
    _GNU_SOURCE
    ${BART_C_DEFS}
    # Windows: a 64-bit off_t, so a file past 2 GiB can be mapped; BART's own
    # functions linked into this library rather than imported from its DLL
    # (misc/dllspec.h); and no named pipes, which are a POSIX file type.
    $<$<BOOL:${WIN32}>:_FILE_OFFSET_BITS=64;BARTLIB_STATIC;NO_FIFO>
    # Elsewhere misc/dllspec.h gives BARTLIB_API default visibility, which
    # would put BART's seq_*, num_rand_init and debug_printf in the dynamic
    # table beside the bartorch_* ABI; defining its guard leaves them hidden.
    $<$<NOT:$<BOOL:${WIN32}>>:BART_MISC_DLLSPEC;BARTLIB_API=;BARTLIB_CALL=>
    BARTORCH_BUILD_INFO="bart=${BART_VERSION},compiler=${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION},nested=${BARTORCH_NESTED},openmp=${OpenMP_C_FOUND},cuda=${BARTORCH_CUDA},finufft=${BARTORCH_FINUFFT_VERSION},finufft_fft=${BARTORCH_FINUFFT_FFT_NAME},finufft_simd=${_finufft_simd_info},finufft_simd_mkl=${_finufft_mkl_info},long=${BARTORCH_SIZEOF_LONG}")

target_compile_options(bartorch PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${BART_C_OPTIONS}>
    # nvcc reaches the host compiler for the code around each kernel, and
    # -compress-all keeps the fatbinaries from dominating the wheel.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC;-Xfatbin=-compress-all>)

if(BARTORCH_CUDA)
    set_target_properties(bartorch PROPERTIES
        CUDA_ARCHITECTURES "${BARTORCH_CUDA_ARCH_SPEC}")
    target_link_libraries(bartorch PRIVATE
        CUDA::cudart CUDA::cufft CUDA::cublas ${CMAKE_DL_LIBS})
    # The runtime libraries live beside torch in the nvidia wheels; torch
    # loads them before this library, and the rpath covers a bare import.
    set_property(TARGET bartorch APPEND PROPERTY INSTALL_RPATH
        "$ORIGIN/../nvidia/cuda_runtime/lib"
        "$ORIGIN/../nvidia/cufft/lib"
        "$ORIGIN/../nvidia/cublas/lib"
        "$ORIGIN/../nvidia/cu12/lib"
        "$ORIGIN/../nvidia/cu13/lib")
endif()

target_link_options(bartorch PRIVATE ${BART_LINK_OPTIONS})

target_link_libraries(bartorch PRIVATE Threads::Threads ${BART_EXTRA_LIBS} ${CMAKE_DL_LIBS})

target_link_libraries(bartorch PRIVATE m)

target_link_libraries(bartorch PRIVATE ${BARTORCH_FINUFFT_LIBS})
if(BARTORCH_FINUFFT_MODULES)
    add_dependencies(bartorch ${BARTORCH_FINUFFT_MODULES})
endif()

# FINUFFT on oneMKL needs libmkl_rt where the build found it.
if(TARGET bartorch_mkl_fftw)
    set_property(TARGET bartorch APPEND PROPERTY INSTALL_RPATH "${BARTORCH_MKL_LIBRARY_DIR}")
endif()

# Linux: libgomp.so.1 is torch's in a wheel, which carries none of its own;
# torch is its sibling package.  The copy torch has loaded satisfies the name
# first, so this is where it is found when torch has not been imported.
if(NOT APPLE AND NOT WIN32 AND OpenMP_C_FOUND)
    set_property(TARGET bartorch APPEND PROPERTY INSTALL_RPATH "$ORIGIN/../torch/lib")
endif()

# macOS: the OpenMP stub names @rpath/libomp.dylib, which is torch's.  In a
# wheel torch is the library's sibling package; an in-place build finds it
# where the interpreter that configured the build has it.
if(APPLE AND OpenMP_C_FOUND)
    set_property(TARGET bartorch APPEND PROPERTY INSTALL_RPATH "@loader_path/../torch/lib")
    set_property(TARGET bartorch APPEND PROPERTY BUILD_RPATH "@loader_path/../torch/lib")
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, pathlib; s = importlib.util.find_spec('torch'); print(pathlib.Path(s.origin).parent / 'lib')"
            OUTPUT_VARIABLE _torch_lib OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _torch_rc)
        if(_torch_rc EQUAL 0 AND EXISTS "${_torch_lib}")
            set_property(TARGET bartorch APPEND PROPERTY INSTALL_RPATH "${_torch_lib}")
            set_property(TARGET bartorch APPEND PROPERTY BUILD_RPATH "${_torch_lib}")
        endif()
    endif()
endif()

set_target_properties(bartorch PROPERTIES
    OUTPUT_NAME bartorch
    PREFIX "lib")

install(TARGETS bartorch
    LIBRARY DESTINATION bartorch
    RUNTIME DESTINATION bartorch)
