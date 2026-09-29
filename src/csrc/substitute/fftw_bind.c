/*
 * The FFTW entry points FINUFFT calls, in a FINUFFT module built on its FFTW
 * path, answered by whichever library the host hands over at run time.
 *
 * The module names no FFT library: it is loaded only once the host has found
 * oneMKL in the process and passed its FFTW3 interface in through
 * bartorch_fftw_bind, so the module works with whatever release of oneMKL is
 * installed and loads nothing of its own.  bartorch_fftw_bind is all or
 * nothing: a module with an entry point missing is not used.
 */
#include <stddef.h>

#include <fftw3.h>

#define FFTW_ENTRY_POINTS(X) \
	X(fftwf_plan_many_dft) \
	X(fftwf_execute_dft) \
	X(fftwf_destroy_plan) \
	X(fftwf_init_threads) \
	X(fftwf_plan_with_nthreads) \
	X(fftwf_cleanup_threads) \
	X(fftwf_forget_wisdom) \
	X(fftwf_cleanup) \
	X(fftw_plan_many_dft) \
	X(fftw_execute_dft) \
	X(fftw_destroy_plan) \
	X(fftw_init_threads) \
	X(fftw_plan_with_nthreads) \
	X(fftw_cleanup_threads) \
	X(fftw_forget_wisdom) \
	X(fftw_cleanup)

static struct {
#define FIELD(name) __typeof__(&name) name;
	FFTW_ENTRY_POINTS(FIELD)
#undef FIELD
} fftw;

#ifdef _WIN32
#define FFTW_BIND_EXPORT __declspec(dllexport)
#else
#define FFTW_BIND_EXPORT __attribute__((visibility("default")))
#endif

FFTW_BIND_EXPORT int bartorch_fftw_bind(void* (*lookup)(const char* name))
{
	int complete = 1;

#define RESOLVE(name) \
	fftw.name = (__typeof__(&name))lookup(#name); \
	complete &= (NULL != fftw.name);
	FFTW_ENTRY_POINTS(RESOLVE)
#undef RESOLVE

	return complete ? 0 : -1;
}

fftwf_plan fftwf_plan_many_dft(int rank, const int* n, int howmany,
		fftwf_complex* in, const int* inembed, int istride, int idist,
		fftwf_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags)
{
	return fftw.fftwf_plan_many_dft(rank, n, howmany, in, inembed, istride, idist,
			out, onembed, ostride, odist, sign, flags);
}

fftw_plan fftw_plan_many_dft(int rank, const int* n, int howmany,
		fftw_complex* in, const int* inembed, int istride, int idist,
		fftw_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags)
{
	return fftw.fftw_plan_many_dft(rank, n, howmany, in, inembed, istride, idist,
			out, onembed, ostride, odist, sign, flags);
}

void fftwf_execute_dft(const fftwf_plan p, fftwf_complex* in, fftwf_complex* out) { fftw.fftwf_execute_dft(p, in, out); }
void fftw_execute_dft(const fftw_plan p, fftw_complex* in, fftw_complex* out) { fftw.fftw_execute_dft(p, in, out); }
void fftwf_destroy_plan(fftwf_plan p) { fftw.fftwf_destroy_plan(p); }
void fftw_destroy_plan(fftw_plan p) { fftw.fftw_destroy_plan(p); }
int fftwf_init_threads(void) { return fftw.fftwf_init_threads(); }
int fftw_init_threads(void) { return fftw.fftw_init_threads(); }
void fftwf_plan_with_nthreads(int n) { fftw.fftwf_plan_with_nthreads(n); }
void fftw_plan_with_nthreads(int n) { fftw.fftw_plan_with_nthreads(n); }
void fftwf_cleanup_threads(void) { fftw.fftwf_cleanup_threads(); }
void fftw_cleanup_threads(void) { fftw.fftw_cleanup_threads(); }
void fftwf_forget_wisdom(void) { fftw.fftwf_forget_wisdom(); }
void fftw_forget_wisdom(void) { fftw.fftw_forget_wisdom(); }
void fftwf_cleanup(void) { fftw.fftwf_cleanup(); }
void fftw_cleanup(void) { fftw.fftw_cleanup(); }
