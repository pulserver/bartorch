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
#   FINUFFT_USE_DUCC0      BARTORCH_FINUFFT_FFT, below: DUCC0's, compiled in,
#                          unless a source build asks for oneMKL.  FFTW
#                          itself would be a GPL library to build and carry.
#   FINUFFT_USE_OPENMP     BARTORCH_OPENMP, on the runtime cmake/openmp.cmake
#                          chose for BART.
#   FINUFFT_ARCH_FLAGS     -march=x86-64 on x86-64 and nothing elsewhere, as
#                          FINUFFT's own wheels are built: a wheel runs on any
#                          CPU of its platform, where FINUFFT's default of
#                          -march=native would not.  BARTORCH_FINUFFT_SIMD,
#                          below, adds builds for newer x86-64 levels, chosen
#                          between at run time.
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

# The FFT inside FINUFFT's CPU transform.  DUCC0 is compiled in and is what
# every wheel carries.  MKL is oneMKL's FFT reached through its FFTW3
# interface, which is FINUFFT's own FFTW path linked against libmkl_rt: a
# source build on x86-64 Linux with oneMKL installed (the `mkl` and
# `mkl-devel` packages), whose library then needs that libmkl_rt at run time.
# The `mkl` extra is the same installation, so BART's DFTI table and FINUFFT
# share one MKL.  docs/design/finufft-embedding.md has the measurements behind
# the default.
set(BARTORCH_FINUFFT_FFT "AUTO" CACHE STRING "FFT inside FINUFFT's CPU transform: AUTO (DUCC0), DUCC0 or MKL")
set_property(CACHE BARTORCH_FINUFFT_FFT PROPERTY STRINGS AUTO DUCC0 MKL)
string(TOUPPER "${BARTORCH_FINUFFT_FFT}" _finufft_fft)
if(_finufft_fft STREQUAL "AUTO")
    set(_finufft_fft "DUCC0")
endif()
if(NOT _finufft_fft MATCHES "^(DUCC0|MKL)$")
    message(FATAL_ERROR "BARTORCH_FINUFFT_FFT=${BARTORCH_FINUFFT_FFT}: expected AUTO, DUCC0 or MKL")
endif()

if(_finufft_fft STREQUAL "MKL")
    if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"))
        message(FATAL_ERROR "BARTORCH_FINUFFT_FFT=MKL is supported on x86-64 Linux only")
    endif()
    # The header and the library have to come from one installation: MKL's
    # fftw3.h beside its fftw3_mkl.h, and libmkl_rt under the same root.
    # MKLROOT, then the interpreter's prefix, which is where the pip packages
    # put them.
    find_package(Python COMPONENTS Interpreter QUIET)
    set(_mkl_hints "$ENV{MKLROOT}")
    if(Python_Interpreter_FOUND)
        execute_process(COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
            OUTPUT_VARIABLE _prefix OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        list(APPEND _mkl_hints "${_prefix}")
    endif()
    find_path(BARTORCH_MKL_FFTW_INCLUDE_DIR fftw3_mkl.h
        HINTS ${_mkl_hints} PATH_SUFFIXES include/fftw)
    if(NOT BARTORCH_MKL_FFTW_INCLUDE_DIR OR NOT EXISTS "${BARTORCH_MKL_FFTW_INCLUDE_DIR}/fftw3.h")
        message(FATAL_ERROR "BARTORCH_FINUFFT_FFT=MKL: no oneMKL include/fftw/fftw3_mkl.h found; install mkl-devel or set MKLROOT")
    endif()
    get_filename_component(_mkl_root "${BARTORCH_MKL_FFTW_INCLUDE_DIR}/../.." ABSOLUTE)
    find_library(BARTORCH_MKL_RT NAMES mkl_rt libmkl_rt.so.2 libmkl_rt.so.3
        PATHS "${_mkl_root}/lib" "${_mkl_root}/lib/intel64" NO_DEFAULT_PATH)
    if(NOT BARTORCH_MKL_RT)
        message(FATAL_ERROR "BARTORCH_FINUFFT_FFT=MKL: no libmkl_rt under ${_mkl_root}, where its fftw3.h is")
    endif()
    file(STRINGS "${_mkl_root}/include/mkl_version.h" _mkl_version REGEX "define INTEL_MKL_VERSION ")
    string(REGEX MATCH "[0-9]+" _mkl_version "${_mkl_version}")
    add_library(bartorch_mkl_fftw SHARED IMPORTED)
    set_target_properties(bartorch_mkl_fftw PROPERTIES
        IMPORTED_LOCATION "${BARTORCH_MKL_RT}"
        INTERFACE_INCLUDE_DIRECTORIES "${BARTORCH_MKL_FFTW_INCLUDE_DIR}")
    get_filename_component(BARTORCH_MKL_LIBRARY_DIR "${BARTORCH_MKL_RT}" DIRECTORY)
    # A cache entry, because FINUFFT's own CMake clears the normal variable
    # before it reads this one.
    set(FINUFFT_FFTW_LIBRARIES bartorch_mkl_fftw CACHE STRING "Custom FFTW library" FORCE)
    set(FINUFFT_USE_DUCC0 OFF)
    set(BARTORCH_FINUFFT_FFT_NAME "mkl")
    message(STATUS "FINUFFT's FFT: oneMKL ${_mkl_version} through its FFTW3 interface, ${BARTORCH_MKL_RT}")
else()
    set(FINUFFT_USE_DUCC0 ON)
    set(BARTORCH_FINUFFT_FFT_NAME "ducc0")
endif()

set(FINUFFT_USE_CPU ON)
set(FINUFFT_USE_CUDA ${BARTORCH_CUDA})
set(FINUFFT_STATIC_LINKING ON)
set(FINUFFT_POSITION_INDEPENDENT_CODE ON)
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

# FINUFFT's CPU transform again for each x86-64 level BARTORCH_FINUFFT_SIMD
# lists, each as a module beside the library, which src/csrc/substitute/finufft.c
# opens when the processor runs that level.  The baseline stays compiled in, so
# a processor that runs none of them, or a module that does not load, costs
# nothing but the speed.  x86-64-v3 (AVX2, FMA) makes FINUFFT 15 to 45 per cent
# faster than the baseline; x86-64-v4 (AVX-512) adds nothing measurable beside
# it, which is why it is not in the default; docs/design/finufft-embedding.md
# has the measurements.
#
# A module is FINUFFT's own targets compiled again with the level's -march:
# their sources, definitions, options and dependencies are read off the
# targets FINUFFT defined above rather than listed here, so a bump of the pin
# carries them.  Only the baseline's arch flag is replaced; a pin whose targets
# no longer carry it stops the configure.
if(FINUFFT_ARCH_FLAGS STREQUAL "-march=x86-64" AND NOT APPLE)
    set(_simd_default "x86-64-v3")
else()
    set(_simd_default "")
endif()
set(BARTORCH_FINUFFT_SIMD "${_simd_default}" CACHE STRING
    "x86-64 levels FINUFFT's CPU transform is also built for, chosen at run time: any of x86-64-v2, x86-64-v3, x86-64-v4")
set(BARTORCH_FINUFFT_SIMD_LEVELS "")
if(BARTORCH_FINUFFT_SIMD)
    if(NOT FINUFFT_ARCH_FLAGS STREQUAL "-march=x86-64")
        message(FATAL_ERROR "BARTORCH_FINUFFT_SIMD=${BARTORCH_FINUFFT_SIMD} needs FINUFFT_ARCH_FLAGS=-march=x86-64, the baseline it is chosen against")
    endif()
    include(CheckCSourceCompiles)
    foreach(_level IN LISTS BARTORCH_FINUFFT_SIMD)
        if(NOT _level MATCHES "^x86-64-v[234]$")
            message(FATAL_ERROR "BARTORCH_FINUFFT_SIMD: ${_level} is not one of x86-64-v2, x86-64-v3, x86-64-v4")
        endif()
        string(REPLACE "-" "_" _id "${_level}")
        check_c_source_compiles(
            "int main(void) { __builtin_cpu_init(); return __builtin_cpu_supports(\"${_level}\"); }"
            BARTORCH_CPU_SUPPORTS_${_id})
        if(NOT BARTORCH_CPU_SUPPORTS_${_id})
            message(FATAL_ERROR "BARTORCH_FINUFFT_SIMD: the C compiler cannot test for ${_level} at run time")
        endif()
    endforeach()
endif()

function(_bartorch_finufft_clone src dst flags)
    get_target_property(_dir ${src} SOURCE_DIR)
    get_target_property(_sources ${src} SOURCES)
    set(_files)
    foreach(_s IN LISTS _sources)
        if(_s MATCHES "^\\$<")
            continue()
        endif()
        if(NOT IS_ABSOLUTE "${_s}")
            set(_s "${_dir}/${_s}")
        endif()
        list(APPEND _files "${_s}")
    endforeach()
    add_library(${dst} OBJECT EXCLUDE_FROM_ALL ${_files})
    set(_replaced FALSE)
    foreach(_p COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FEATURES INCLUDE_DIRECTORIES
            LINK_LIBRARIES INTERFACE_INCLUDE_DIRECTORIES CXX_STANDARD
            CXX_VISIBILITY_PRESET VISIBILITY_INLINES_HIDDEN)
        get_target_property(_v ${src} ${_p})
        if(NOT _v)
            continue()
        endif()
        if(_p STREQUAL "COMPILE_OPTIONS")
            string(REGEX REPLACE "-march=x86-64([;>]|$)" "${flags}\\1" _new "${_v}")
            if(NOT _new STREQUAL _v)
                set(_replaced TRUE)
            endif()
            set(_v "${_new}")
        endif()
        if(_p STREQUAL "LINK_LIBRARIES" AND FINUFFT_USE_DUCC0)
            string(REGEX REPLACE "(^|[:;])finufft_fftlibs([;>]|$)" "\\1${ARGV3}\\2" _v "${_v}")
        endif()
        set_property(TARGET ${dst} PROPERTY ${_p} "${_v}")
    endforeach()
    if(NOT _replaced)
        message(FATAL_ERROR "FINUFFT's target ${src} no longer carries -march=x86-64; the SIMD builds cannot replace it")
    endif()
    set_target_properties(${dst} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

foreach(_level IN LISTS BARTORCH_FINUFFT_SIMD)
    string(REPLACE "-" "_" _id "${_level}")
    set(_ducc)
    if(FINUFFT_USE_DUCC0)
        set(_ducc bartorch_ducc0_${_id})
        _bartorch_finufft_clone(ducc0 ${_ducc} "-march=${_level}")
    endif()
    _bartorch_finufft_clone(finufft_f32 bartorch_finufft_f32_${_id} "-march=${_level}" ${_ducc})
    _bartorch_finufft_clone(finufft bartorch_finufft_f64_${_id} "-march=${_level}" ${_ducc})
    # The module's C API: FINUFFT's own export macros, as its shared build
    # sets them.
    foreach(_t bartorch_finufft_f32_${_id} bartorch_finufft_f64_${_id})
        target_compile_definitions(${_t} PRIVATE FINUFFT_DLL $<$<BOOL:${WIN32}>:dll_EXPORTS>)
    endforeach()

    set(_module bartorch_finufft_${_id})
    set(_objects bartorch_finufft_f32_${_id} bartorch_finufft_f64_${_id} ${_ducc})
    list(TRANSFORM _objects REPLACE "(.+)" "$<TARGET_OBJECTS:\\1>")
    add_library(${_module} MODULE ${_objects})
    set_target_properties(${_module} PROPERTIES
        PREFIX "lib" OUTPUT_NAME "bartorch_finufft_${_id}" LINKER_LANGUAGE CXX)
    target_link_libraries(${_module} PRIVATE finufft_common Threads::Threads)
    if(NOT FINUFFT_USE_DUCC0)
        target_link_libraries(${_module} PRIVATE finufft_fftlibs)
        if(TARGET bartorch_mkl_fftw)
            set_property(TARGET ${_module} APPEND PROPERTY INSTALL_RPATH "${BARTORCH_MKL_LIBRARY_DIR}")
        endif()
    endif()
    if(FINUFFT_USE_OPENMP)
        target_link_libraries(${_module} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(NOT WIN32)
        target_link_libraries(${_module} PRIVATE m)
    endif()
    # The same runtime arrangement as the library itself, and on ELF nothing
    # exported but FINUFFT's C API.
    target_link_options(${_module} PRIVATE ${BART_LINK_OPTIONS})
    if(NOT WIN32 AND NOT APPLE)
        target_link_options(${_module} PRIVATE
            "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/finufft_module.map")
    endif()
    install(TARGETS ${_module} LIBRARY DESTINATION bartorch RUNTIME DESTINATION bartorch)
    list(APPEND BARTORCH_FINUFFT_SIMD_LEVELS "${_level}")
    list(APPEND BARTORCH_FINUFFT_MODULES ${_module})
endforeach()

file(STRINGS "${FINUFFT_ROOT}/CMakeLists.txt" _finufft_project REGEX "^project\\(FINUFFT VERSION")
string(REGEX MATCH "VERSION ([0-9.]+)" _ "${_finufft_project}")
set(BARTORCH_FINUFFT_VERSION "${CMAKE_MATCH_1}")
message(STATUS "FINUFFT ${BARTORCH_FINUFFT_VERSION} from ${FINUFFT_ROOT}: cpu, openmp=${FINUFFT_USE_OPENMP}, cuda=${BARTORCH_CUDA}, fft=${BARTORCH_FINUFFT_FFT_NAME}, arch='${FINUFFT_ARCH_FLAGS}', simd='${BARTORCH_FINUFFT_SIMD}'")

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
if(FINUFFT_USE_DUCC0)
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
endif()
if(BARTORCH_CUDA AND CPM_PACKAGE_CCCL_SOURCE_DIR AND EXISTS "${CPM_PACKAGE_CCCL_SOURCE_DIR}/LICENSE")
    configure_file("${CPM_PACKAGE_CCCL_SOURCE_DIR}/LICENSE" "${_notices}/CCCL/LICENSE" COPYONLY)
endif()
if(DEFINED SKBUILD_METADATA_DIR)
    install(DIRECTORY "${_notices}/" DESTINATION "${SKBUILD_METADATA_DIR}/licenses/finufft-dependencies")
endif()
