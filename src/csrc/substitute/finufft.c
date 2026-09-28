/*
 * FINUFFT and cuFINUFFT, as BART reaches them.
 *
 * Both are compiled from external/finufft into this library and called
 * through their C API.  The two answer different memory -- cuFINUFFT carries a
 * device number in its options where FINUFFT carries a thread count -- so a
 * plan records which of them made it, picked by where the data is.
 *
 * What is done with a plan belongs to nufft_finufft.c, which builds BART's
 * NUFFT operator out of a pair of them.
 */
#include <complex.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <finufft.h>
#ifdef USE_CUDA
#include <cufinufft.h>
#endif

#include "include/bartorch.h"

static struct {

	int use_in_tools;
	double tolerance;
	double upsampling;
	int threads;

} fi = { .use_in_tools = 0, .tolerance = 1.e-3, .upsampling = 1.25, .threads = 0 };

static pthread_mutex_t fi_lock = PTHREAD_MUTEX_INITIALIZER;

/* The version of FINUFFT compiled in, from the submodule's CMakeLists. */
const char* bartorch_finufft_version(void)
{
	return BARTORCH_FINUFFT_VERSION;
}

/* Whether this build carries the transform for that side: FINUFFT always,
 * cuFINUFFT in a CUDA build. */
int bartorch_finufft_built_on(int device)
{
#ifdef USE_CUDA
	return 1;
#else
	return device ? 0 : 1;
#endif
}

void bartorch_finufft_set_tolerance(double eps)
{
	if ((eps > 0.) && (eps < 1.))
		fi.tolerance = eps;
}

double bartorch_finufft_tolerance(void)
{
	return fi.tolerance;
}

/* How far past the image FINUFFT spreads before it transforms.
 *
 * Two is the textbook grid; a quarter over trades a smaller one for a wider
 * kernel, and that is the default here: it holds a quarter of the memory the
 * textbook grid does at the tolerance this asks for, and a reconstruction is
 * not made better by a transform an order more accurate than the data.  Zero
 * lets FINUFFT weigh it per problem instead. */
void bartorch_finufft_set_upsampling(double upsampling)
{
	if ((0. == upsampling) || ((upsampling > 1.) && (upsampling <= 4.)))
		fi.upsampling = upsampling;
}

double bartorch_finufft_upsampling(void)
{
	return fi.upsampling;
}

/* How many threads a transform on the host is given.
 *
 * Zero, the state this starts in, leaves the count to FINUFFT, which takes a
 * thread per physical core.  bartorch_set_num_threads sets this along with
 * BART's own, so one number covers the process; this is how that is undone
 * without setting BART to a count of its own.  A card has no say in it --
 * cuFINUFFT carries a device number where FINUFFT carries this. */
void bartorch_finufft_set_threads(int n)
{
	fi.threads = (n > 0) ? n : 0;
}

int bartorch_finufft_threads(void)
{
	return fi.threads;
}

int bartorch_finufft_usable_on(int device)
{
	return (bartorch_finufft_built_on(device) && fi.use_in_tools) ? 1 : 0;
}

int bartorch_finufft_usable(void)
{
	return bartorch_finufft_usable_on(0);
}

void bartorch_finufft_use_in_tools(int enable)
{
	fi.use_in_tools = (0 != enable);
}

/* A plan carries which library made it, so the operator does not have to. */
struct bartorch_fi_plan {

	int device;
	void* plan;
};

static void destroy(int device, void* plan)
{
#ifdef USE_CUDA
	if (device) {

		cufinufftf_destroy(plan);
		return;
	}
#endif
	(void)device;
	finufftf_destroy(plan);
}

/* Plans made and not yet destroyed.  A plan belongs to whatever made it -- an
 * operator, a point spread function, the spreading a compressed one is masked
 * with -- and outlives none of them, so this is back at zero once the last of
 * them is gone.  That is a property worth testing, and one RSS cannot be read
 * for: FINUFFT's own multithreaded execute retains a kilobyte per thread per
 * call, which any measurement of the process would drown this in. */
static int64_t fi_live_plans;

/* `spread_only` asks FINUFFT for the spreading alone -- no transform, no
 * deapodisation -- which puts the kernel's own footprint on the grid.  That is
 * what a compressed point spread function's mask is: which grid points the
 * samples reach. */
int bartorch_finufft_plan(int device, int type, int dim, const int64_t n_modes[3], int ntrans,
		int isign, double eps, double upsampling, int spread_only, void** plan)
{
	if (!bartorch_finufft_built_on(device))
		return -1;

	void* p = NULL;
	int ret;

	pthread_mutex_lock(&fi_lock);

	if (device) {
#ifdef USE_CUDA
		cufinufft_opts opts;
		cufinufft_default_opts(&opts);

		int which = bartorch_cuda_device();
		opts.gpu_device_id = (which > 0) ? which : 0;

		if (0. != upsampling)
			opts.upsampfac = upsampling;

		if (0 != spread_only)
			opts.gpu_spreadinterponly = 1;

		cufinufftf_plan q = NULL;
		ret = cufinufftf_makeplan(type, dim, n_modes, isign, ntrans, (float)eps, &q, &opts);
		p = q;
#else
		ret = -1;
#endif
	} else {

		finufft_opts opts;
		finufftf_default_opts(&opts);

		opts.nthreads = fi.threads;

		/* A tolerance below what single precision reaches at this size is
		 * planned at the tolerance it can reach rather than refused, which is
		 * what a caller asking for the most accurate transform means. */
		opts.allow_eps_too_small = 1;

		if (0. != upsampling)
			opts.upsampfac = upsampling;

		if (0 != spread_only)
			opts.spreadinterponly = 1;

		finufftf_plan q = NULL;
		ret = finufftf_makeplan(type, dim, n_modes, isign, ntrans, (float)eps, &q, &opts);
		p = q;
	}

	pthread_mutex_unlock(&fi_lock);

	if (0 != ret)
		return -1;

	struct bartorch_fi_plan* held = malloc(sizeof *held);

	if (NULL == held) {

		destroy(device, p);
		return -1;
	}

	held->device = device;
	held->plan = p;
	*plan = held;

#pragma omp atomic
	fi_live_plans++;

	return 0;
}

/* A plan keeps the points by pointer rather than copying them, so the arrays
 * given here have to outlive it. */
int bartorch_finufft_setpts(void* plan, int64_t M, float* x, float* y, float* z)
{
	const struct bartorch_fi_plan* p = plan;

#ifdef USE_CUDA
	if (p->device)
		return cufinufftf_setpts(p->plan, M, x, y, z, 0, NULL, NULL, NULL);
#endif
	return finufftf_setpts(p->plan, M, x, y, z, 0, NULL, NULL, NULL);
}

int bartorch_finufft_exec(void* plan, complex float* c, complex float* f)
{
	const struct bartorch_fi_plan* p = plan;

#ifdef USE_CUDA
	if (p->device)
		return cufinufftf_execute(p->plan, (cuFloatComplex*)c, (cuFloatComplex*)f);
#endif
	return finufftf_execute(p->plan, c, f);
}

void bartorch_finufft_free(void* plan)
{
	if (NULL == plan)
		return;

	struct bartorch_fi_plan* p = plan;

	pthread_mutex_lock(&fi_lock);
	destroy(p->device, p->plan);
	pthread_mutex_unlock(&fi_lock);

	free(p);

#pragma omp atomic
	fi_live_plans--;
}

int64_t bartorch_finufft_live_plans(void)
{
	return fi_live_plans;
}
