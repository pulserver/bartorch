/* The FFTW3 single-precision surface BART's num/fft_plan.c uses, served by
 * src/csrc/substitute/fft.cpp.
 *
 * The functions are defined under names of their own, so that FFTW's names in
 * this library belong to whatever FINUFFT's FFT is linked against: a FINUFFT
 * built on FFTW's interface calls fftwf_plan_many_dft from that library and
 * has to execute and destroy the plan there too. */
#ifndef BARTORCH_FFTW3_H
#define BARTORCH_FFTW3_H

#define fftwf_plan_guru64_dft bartorch_fftwf_plan_guru64_dft
#define fftwf_execute_dft bartorch_fftwf_execute_dft
#define fftwf_destroy_plan bartorch_fftwf_destroy_plan
#define fftwf_export_wisdom_to_filename bartorch_fftwf_export_wisdom_to_filename
#define fftwf_import_wisdom_from_filename bartorch_fftwf_import_wisdom_from_filename
#define fftwf_init_threads bartorch_fftwf_init_threads
#define fftwf_plan_with_nthreads bartorch_fftwf_plan_with_nthreads
#define fftwf_cleanup_threads bartorch_fftwf_cleanup_threads

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
typedef void bartorch_cfloat;
extern "C" {
#else
#include <complex.h>
typedef float _Complex bartorch_cfloat;
#endif

typedef struct { ptrdiff_t n; ptrdiff_t is; ptrdiff_t os; } fftwf_iodim64;
typedef void* fftwf_plan;

#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (+1)
#define FFTW_MEASURE (0U)
#define FFTW_ESTIMATE (1U << 6)

fftwf_plan fftwf_plan_guru64_dft(int rank, const fftwf_iodim64* dims, int howmany_rank, const fftwf_iodim64* howmany_dims, bartorch_cfloat* in, bartorch_cfloat* out, int sign, unsigned flags);
void fftwf_execute_dft(const fftwf_plan p, bartorch_cfloat* in, bartorch_cfloat* out);
void fftwf_destroy_plan(fftwf_plan p);
int fftwf_export_wisdom_to_filename(const char* filename);
int fftwf_import_wisdom_from_filename(const char* filename);
int fftwf_init_threads(void);
void fftwf_plan_with_nthreads(int n);
void fftwf_cleanup_threads(void);

#ifdef __cplusplus
}
#endif

#endif
