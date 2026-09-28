# ---------------------------------------------------------------------------
# FINUFFT, and cuFINUFFT in a CUDA build, compiled from external/finufft.
#
# The submodule is upstream FINUFFT at one reviewed commit, built through its
# own CMake as static libraries and linked into libbartorch, so the library
# calls FINUFFT's C API directly and nothing is loaded beside it at run time.
# Sets BARTORCH_FINUFFT_LIBS to what libbartorch links.
#
# FINUFFT's own dependencies -- xsimd, POET and DUCC0 on the host, CCCL where
# the CUDA toolkit lacks it -- are the versions its CMake pins, fetched by its
# CPM at configure time.  CPM_SOURCE_CACHE (a CMake or environment variable)
# keeps them between builds and serves a build without network access.
#
# The choices made here, each against FINUFFT's own default:
#
#   FINUFFT_USE_DUCC0      ON: FINUFFT's FFT is DUCC0's, compiled in.  FFTW
#                          would be a GPL library to build and carry.
#   FINUFFT_USE_OPENMP     BARTORCH_OPENMP, on the runtime cmake/openmp.cmake
#                          chose for BART.
#   FINUFFT_ARCH_FLAGS     -march=x86-64 on x86-64 and nothing elsewhere, as
#                          FINUFFT's own wheels are built: a wheel runs on any
#                          CPU of its platform, where FINUFFT's default of
#                          -march=native would not.
#   CMAKE_CUDA_ARCHITECTURES
#                          BARTORCH_CUDA_ARCHITECTURES, as for BART's kernels.
# ---------------------------------------------------------------------------

set(FINUFFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/finufft")
if(NOT EXISTS "${FINUFFT_ROOT}/include/finufft.h")
    message(FATAL_ERROR "FINUFFT sources not found at ${FINUFFT_ROOT}; run `git submodule update --init`")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(_finufft_arch "-march=x86-64")
else()
    set(_finufft_arch "")
endif()
set(FINUFFT_ARCH_FLAGS "${_finufft_arch}" CACHE STRING
    "Target flags FINUFFT is compiled with; native tunes it to the building machine")

set(FINUFFT_USE_CPU ON)
set(FINUFFT_USE_CUDA ${BARTORCH_CUDA})
set(FINUFFT_STATIC_LINKING ON)
set(FINUFFT_POSITION_INDEPENDENT_CODE ON)
set(FINUFFT_USE_DUCC0 ON)
set(FINUFFT_USE_OPENMP ${OpenMP_C_FOUND})
set(FINUFFT_ENABLE_INSTALL OFF)
set(FINUFFT_BUILD_TESTS OFF)
set(FINUFFT_BUILD_EXAMPLES OFF)
set(FINUFFT_BUILD_PYTHON OFF)
set(FINUFFT_WARNINGS_AS_ERRORS OFF)

if(BARTORCH_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "${BARTORCH_CUDA_ARCH_SPEC}")
endif()

# FINUFFT is somebody else's code: its warnings are not this build's, and
# nothing it builds is installed with the library.
add_subdirectory("${FINUFFT_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/finufft" EXCLUDE_FROM_ALL SYSTEM)

set(BARTORCH_FINUFFT_LIBS finufft finufft_common)
if(BARTORCH_CUDA)
    list(APPEND BARTORCH_FINUFFT_LIBS cufinufft)
endif()

file(STRINGS "${FINUFFT_ROOT}/CMakeLists.txt" _finufft_project REGEX "^project\\(FINUFFT VERSION")
string(REGEX MATCH "VERSION ([0-9.]+)" _ "${_finufft_project}")
set(BARTORCH_FINUFFT_VERSION "${CMAKE_MATCH_1}")
message(STATUS "FINUFFT ${BARTORCH_FINUFFT_VERSION} from ${FINUFFT_ROOT}: cpu, openmp=${FINUFFT_USE_OPENMP}, cuda=${BARTORCH_CUDA}, fft=ducc0, arch='${FINUFFT_ARCH_FLAGS}'")

# The notices of what FINUFFT compiles in with it, which the build fetched
# rather than the checkout carrying.  scikit-build-core puts what is installed
# under SKBUILD_METADATA_DIR into the wheel's .dist-info.  DUCC0's own LICENSE
# is the GPL; the files FINUFFT compiles are each BSD-3-Clause OR
# GPL-2.0-or-later, and are taken under the first, whose text their headers
# carry.
set(_notices "${CMAKE_CURRENT_BINARY_DIR}/finufft_notices")
file(MAKE_DIRECTORY "${_notices}")
foreach(_dep xsimd POET)
    if(EXISTS "${CPM_PACKAGE_${_dep}_SOURCE_DIR}/LICENSE")
        configure_file("${CPM_PACKAGE_${_dep}_SOURCE_DIR}/LICENSE" "${_notices}/${_dep}/LICENSE" COPYONLY)
    else()
        message(FATAL_ERROR "FINUFFT's dependency ${_dep} was not fetched, or has no LICENSE")
    endif()
endforeach()
set(_ducc_header "${CPM_PACKAGE_ducc0_SOURCE_DIR}/src/ducc0/fft/fft.h")
if(NOT EXISTS "${_ducc_header}")
    message(FATAL_ERROR "FINUFFT's dependency DUCC0 was not fetched")
endif()
file(READ "${_ducc_header}" _ducc_text)
string(FIND "${_ducc_text}" "Copyright (C)" _from)
string(FIND "${_ducc_text}" "*/" _to)
string(FIND "${_ducc_text}" "All rights reserved." _bsd_from)
string(FIND "${_ducc_text}" "SUCH DAMAGE." _bsd_to)
if(_from GREATER -1 AND _to GREATER _from AND _bsd_from GREATER _to AND _bsd_to GREATER _bsd_from)
    math(EXPR _len "${_to} - ${_from}")
    string(SUBSTRING "${_ducc_text}" ${_from} ${_len} _ducc_copyright)
    math(EXPR _len "${_bsd_to} + 12 - ${_bsd_from}")
    string(SUBSTRING "${_ducc_text}" ${_bsd_from} ${_len} _ducc_bsd)
endif()
if(NOT _ducc_copyright OR NOT _ducc_bsd)
    message(FATAL_ERROR "DUCC0's BSD-3-Clause notice was not where ${_ducc_header} used to carry it")
endif()
file(WRITE "${_notices}/DUCC0/LICENSE"
    "The DUCC0 sources compiled into this library are licensed\n"
    "BSD-3-Clause OR GPL-2.0-or-later, and are used under BSD-3-Clause.\n\n"
    "${_ducc_copyright}\n\n${_ducc_bsd}\n")
if(BARTORCH_CUDA AND CPM_PACKAGE_CCCL_SOURCE_DIR AND EXISTS "${CPM_PACKAGE_CCCL_SOURCE_DIR}/LICENSE")
    configure_file("${CPM_PACKAGE_CCCL_SOURCE_DIR}/LICENSE" "${_notices}/CCCL/LICENSE" COPYONLY)
endif()
if(DEFINED SKBUILD_METADATA_DIR)
    install(DIRECTORY "${_notices}/" DESTINATION "${SKBUILD_METADATA_DIR}/licenses/finufft-dependencies")
endif()
