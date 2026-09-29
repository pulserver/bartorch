# ---------------------------------------------------------------------------
# One OpenMP runtime for BART, FINUFFT and torch.
#
# Defines OpenMP::OpenMP_C and OpenMP::OpenMP_CXX for the whole build, and sets
# OpenMP_C_FOUND.  BART links the first and FINUFFT the second, so both halves
# of the library are bound to the same runtime.
#
# Linux: the toolchain's own runtime, found by CMake's FindOpenMP -- libgomp
# under GCC, libomp under clang.  libgomp.so.1 is also the name of the copy
# torch carries, so the copy torch has loaded satisfies the library's NEEDED
# entry; the wheel is repaired with --exclude libgomp.so.1 and carries none.
#
# Windows and macOS: the runtime torch carries and has already loaded.  LLVM's
# runtime ends the process when a second copy of itself initialises (OMP: Error
# #15), so a library that brought its own could not be loaded beside torch.
# Both share the ABI clang emits calls to, so the library is linked against a
# stub that names torch's runtime and declares the entry points clang calls:
#
#   Windows  an import library for libiomp5md.dll, made by dlltool from
#            src/csrc/compat/libiomp5md.def;
#   macOS    a text-based stub for @rpath/libomp.dylib, written here from the
#            same list, with torch/lib on the rpath.
#
# The loader binds either to the copy in the process.  A name clang emits and
# the list lacks is a link error, not a second runtime.
#
# FINUFFT asks for OpenMP with its own find_package(OpenMP), which a Find
# module would answer with whatever runtime the toolchain has.  A config file
# in CMAKE_FIND_PACKAGE_REDIRECTS_DIR takes precedence over the Find module,
# so on these two platforms that call is answered with the targets below.
# ---------------------------------------------------------------------------

set(OpenMP_C_FOUND FALSE)

if(NOT BARTORCH_OPENMP)
    return()
endif()

if(NOT WIN32 AND NOT APPLE)
    find_package(OpenMP COMPONENTS C CXX)
    if(OpenMP_C_FOUND AND NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "OpenMP was found for C and not for C++; FINUFFT is C++ and needs both")
    endif()
    return()
endif()

set(_omp_def "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/compat/libiomp5md.def")
file(STRINGS "${_omp_def}" _omp_lines REGEX "^[ \t]+[_A-Za-z]")
set(_omp_symbols "")
foreach(_line ${_omp_lines})
    string(STRIP "${_line}" _name)
    list(APPEND _omp_symbols "${_name}")
endforeach()

if(WIN32)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "OpenMP on Windows is built with clang; pass -DBARTORCH_OPENMP=OFF to build without it")
    endif()
    find_program(BARTORCH_DLLTOOL NAMES llvm-dlltool dlltool x86_64-w64-mingw32-dlltool REQUIRED)
    set(_omp_stub "${CMAKE_CURRENT_BINARY_DIR}/libiomp5md.dll.a")
    execute_process(
        COMMAND "${BARTORCH_DLLTOOL}" -m i386:x86-64 -d "${_omp_def}" -l "${_omp_stub}" -D libiomp5md.dll
        RESULT_VARIABLE _omp_rc)
    if(NOT _omp_rc EQUAL 0)
        message(FATAL_ERROR "${BARTORCH_DLLTOOL} could not make an import library from ${_omp_def}")
    endif()
    set(_omp_compile -fopenmp)
    set(_omp_include "")
else()
    # Apple's clang compiles OpenMP when asked through the preprocessor and
    # ships no omp.h; the header is LLVM's, which Homebrew's libomp installs.
    # Only the header is taken from there: nothing is linked against that
    # copy of the runtime.
    find_path(BARTORCH_OMP_INCLUDE_DIR omp.h
        HINTS ENV BARTORCH_OMP_INCLUDE_DIR
        PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include
        NO_DEFAULT_PATH)
    if(NOT BARTORCH_OMP_INCLUDE_DIR)
        message(FATAL_ERROR
            "omp.h not found.  `brew install libomp` provides it (only the header is used: "
            "the library links against the OpenMP runtime torch carries), or set "
            "BARTORCH_OMP_INCLUDE_DIR, or pass -DBARTORCH_OPENMP=OFF to build without OpenMP.")
    endif()
    if(CMAKE_OSX_ARCHITECTURES)
        set(_omp_archs ${CMAKE_OSX_ARCHITECTURES})
    else()
        set(_omp_archs ${CMAKE_SYSTEM_PROCESSOR})
    endif()
    set(_omp_targets "")
    foreach(_arch ${_omp_archs})
        list(APPEND _omp_targets "${_arch}-macos")
    endforeach()
    string(REPLACE ";" ", " _omp_targets "${_omp_targets}")
    set(_omp_exports "")
    foreach(_name ${_omp_symbols})
        list(APPEND _omp_exports "_${_name}")
    endforeach()
    string(REPLACE ";" ", " _omp_exports "${_omp_exports}")
    # torch's libomp.dylib is LLVM's, identified as @rpath/libomp.dylib with
    # compatibility version 5.0.0.
    set(_omp_stub "${CMAKE_CURRENT_BINARY_DIR}/libomp.tbd")
    file(WRITE "${_omp_stub}"
        "--- !tapi-tbd\n"
        "tbd-version: 4\n"
        "targets: [ ${_omp_targets} ]\n"
        "install-name: '@rpath/libomp.dylib'\n"
        "current-version: 5.0.0\n"
        "compatibility-version: 5.0.0\n"
        "exports:\n"
        "  - targets: [ ${_omp_targets} ]\n"
        "    symbols: [ ${_omp_exports} ]\n"
        "...\n")
    set(_omp_compile -Xpreprocessor -fopenmp)
    set(_omp_include "${BARTORCH_OMP_INCLUDE_DIR}")
endif()

foreach(_lang C CXX)
    if(NOT TARGET OpenMP::OpenMP_${_lang})
        add_library(OpenMP::OpenMP_${_lang} INTERFACE IMPORTED GLOBAL)
    endif()
    set_target_properties(OpenMP::OpenMP_${_lang} PROPERTIES
        INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:${_lang}>:${_omp_compile}>"
        INTERFACE_INCLUDE_DIRECTORIES "${_omp_include}"
        INTERFACE_LINK_LIBRARIES "${_omp_stub}")
    set(OpenMP_${_lang}_FOUND TRUE)
endforeach()
set(OpenMP_FOUND TRUE)

file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/openmp-config.cmake"
    "# Written by bartorch's cmake/openmp.cmake: OpenMP is torch's runtime.\n"
    "set(OpenMP_FOUND TRUE)\n"
    "set(OpenMP_C_FOUND TRUE)\n"
    "set(OpenMP_CXX_FOUND TRUE)\n")

message(STATUS "OpenMP: the runtime torch loads, through ${_omp_stub}")
