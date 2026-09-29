/*
 * FINUFFT and cuFINUFFT, as BART reaches them.
 *
 * Both are compiled from external/finufft into this library and called
 * through their C API.  The two answer different memory -- cuFINUFFT carries a
 * device number in its options where FINUFFT carries a thread count -- so a
 * plan records which of them made it, picked by where the data is.
 *
 * FINUFFT's CPU transform is compiled more than once on x86-64: once into this
 * library for the baseline instruction set, and once more into a module beside
 * it for each newer level the build lists (BARTORCH_FINUFFT_SIMD_VARIANTS).  A
 * plan is made by the newest level the processor runs, unless
 * BARTORCH_FINUFFT_SIMD or bartorch_finufft_set_simd names another, and
 * carries the entry points that made it, so a plan outlives a change of level.
On x86-64 Linux each level is built a second time on FINUFFT's FFTW path,
bound at run time to oneMKL's FFTW3 interface (fftw_bind.c), and that one is
preferred once the host has handed oneMKL over.
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

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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

/* FINUFFT's CPU entry points, as one build of it answers them. */
struct fi_api {

	const char* level;
	const char* fft;
	__typeof__(&finufftf_default_opts) default_opts;
	__typeof__(&finufftf_makeplan) makeplan;
	__typeof__(&finufftf_setpts) setpts;
	__typeof__(&finufftf_execute) execute;
	__typeof__(&finufftf_destroy) destroy;
};

static const struct fi_api fi_builtin = {

	.level = BARTORCH_FINUFFT_SIMD_BASE,
	.fft = BARTORCH_FINUFFT_FFT_NAME,
	.default_opts = finufftf_default_opts,
	.makeplan = finufftf_makeplan,
	.setpts = finufftf_setpts,
	.execute = finufftf_execute,
	.destroy = finufftf_destroy,
};

/* A module holds one build of FINUFFT for one instruction-set level and one
 * FFT, and is loaded the first time it is asked for.  Each is a library of
 * its own, opened without adding its names to the process, so its C++
 * runtime, its FFT and its template instances are its own and none can stand
 * in for the baseline's on a processor that cannot run them.  A module on
 * oneMKL links no MKL library: it is handed the FFTW3 interface of the oneMKL
 * the process has loaded (bartorch_finufft_fftw_set), and is not opened until
 * all of it has been. */
struct fi_module {

	const char* level;
	const char* fft;
	int tried;
	struct fi_api api;
};

#define FI_MODULE(level, fft) { level, fft, 0, { NULL } }

static struct fi_module fi_modules[] = {
#ifdef BARTORCH_FINUFFT_SIMD_V2
	FI_MODULE("x86-64-v2", BARTORCH_FINUFFT_FFT_NAME),
#endif
#ifdef BARTORCH_FINUFFT_MKL_V2
	FI_MODULE("x86-64-v2", "mkl"),
#endif
#ifdef BARTORCH_FINUFFT_SIMD_V3
	FI_MODULE("x86-64-v3", BARTORCH_FINUFFT_FFT_NAME),
#endif
#ifdef BARTORCH_FINUFFT_MKL_V3
	FI_MODULE("x86-64-v3", "mkl"),
#endif
#ifdef BARTORCH_FINUFFT_SIMD_V4
	FI_MODULE("x86-64-v4", BARTORCH_FINUFFT_FFT_NAME),
#endif
#ifdef BARTORCH_FINUFFT_MKL_V4
	FI_MODULE("x86-64-v4", "mkl"),
#endif
	{ NULL, NULL, 0, { NULL } },
};

/* The FFTW3 entry points a module on oneMKL calls, as the host found them in
 * the process. */
static const char* const fi_fftw_names[] = {

	"fftwf_plan_many_dft", "fftwf_execute_dft", "fftwf_destroy_plan", "fftwf_init_threads",
	"fftwf_plan_with_nthreads", "fftwf_cleanup_threads", "fftwf_forget_wisdom", "fftwf_cleanup",
	"fftw_plan_many_dft", "fftw_execute_dft", "fftw_destroy_plan", "fftw_init_threads",
	"fftw_plan_with_nthreads", "fftw_cleanup_threads", "fftw_forget_wisdom", "fftw_cleanup",
};

#define FI_FFTW_COUNT ((int)(sizeof fi_fftw_names / sizeof fi_fftw_names[0]))

static void* fi_fftw[FI_FFTW_COUNT];

static int fi_fftw_complete(void)
{
	for (int i = 0; i < FI_FFTW_COUNT; i++)
		if (NULL == fi_fftw[i])
			return 0;

	return 1;
}

static void* fi_fftw_lookup(const char* name)
{
	for (int i = 0; i < FI_FFTW_COUNT; i++)
		if (0 == strcmp(name, fi_fftw_names[i]))
			return fi_fftw[i];

	return NULL;
}

/* The level and FFT plans are made with, and the ones asked for -- the level
 * by a caller or BARTORCH_FINUFFT_SIMD, the FFT by a caller -- "" for the
 * newest level and for oneMKL where the process has it; fi_current is NULL
 * until the first plan decides it. */
static const struct fi_api* fi_current;
static char fi_requested[16];
static char fi_requested_fft[16];
static int fi_environment_read;

/* Only the levels this build carries are tested for, so a compiler that
 * cannot name a level at run time fails the configure that asked for it
 * rather than this file. */
static int fi_cpu_runs(const char* level)
{
#if defined(BARTORCH_FINUFFT_SIMD_V2) || defined(BARTORCH_FINUFFT_SIMD_V3) || defined(BARTORCH_FINUFFT_SIMD_V4)
	__builtin_cpu_init();
#endif
#ifdef BARTORCH_FINUFFT_SIMD_V2
	if (0 == strcmp(level, "x86-64-v2"))
		return __builtin_cpu_supports("x86-64-v2");
#endif
#ifdef BARTORCH_FINUFFT_SIMD_V3
	if (0 == strcmp(level, "x86-64-v3"))
		return __builtin_cpu_supports("x86-64-v3");
#endif
#ifdef BARTORCH_FINUFFT_SIMD_V4
	if (0 == strcmp(level, "x86-64-v4"))
		return __builtin_cpu_supports("x86-64-v4");
#endif
	(void)level;
	return 0;
}

/* The module sits beside this library, named after its level, and after its
 * FFT where that is not the library's own: libbartorch_finufft_x86_64_v3.so
 * for x86-64-v3, libbartorch_finufft_x86_64_v3_mkl.so for it on oneMKL. */
static void* fi_open(const char* level, const char* fft)
{
	char name[64] = "libbartorch_finufft_";
	size_t n = strlen(name);

	for (const char* c = level; *c && (n < sizeof name - 16); c++)
		name[n++] = ('-' == *c) ? '_' : *c;

	if (0 != strcmp(fft, fi_builtin.fft)) {

		name[n++] = '_';

		for (const char* c = fft; *c && (n < sizeof name - 8); c++)
			name[n++] = *c;
	}

	name[n] = '\0';

#ifdef _WIN32
	strcat(name, ".dll");

	HMODULE self = NULL;

	if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			(LPCWSTR)(void*)&bartorch_finufft_version, &self))
		return NULL;

	wchar_t path[4096];
	DWORD len = GetModuleFileNameW(self, path, 4096);

	if ((0 == len) || (len >= 4096))
		return NULL;

	wchar_t* slash = wcsrchr(path, L'\\');
	size_t at = (NULL == slash) ? 0 : (size_t)(slash - path + 1);

	if (at + strlen(name) + 1 > 4096)
		return NULL;

	for (size_t i = 0; i <= strlen(name); i++)
		path[at + i] = (wchar_t)name[i];

	return LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
	strcat(name, ".so");

	Dl_info self;

	if (!dladdr((void*)&bartorch_finufft_version, &self) || (NULL == self.dli_fname))
		return NULL;

	const char* slash = strrchr(self.dli_fname, '/');
	size_t at = (NULL == slash) ? 0 : (size_t)(slash - self.dli_fname + 1);
	char* path = malloc(at + strlen(name) + 1);

	if (NULL == path)
		return NULL;

	memcpy(path, self.dli_fname, at);
	strcpy(path + at, name);

	void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);

	free(path);

	return lib;
#endif
}

static void* fi_symbol(void* lib, const char* name)
{
#ifdef _WIN32
	return (void*)GetProcAddress(lib, name);
#else
	return dlsym(lib, name);
#endif
}

/* A module's entry points, or NULL.  A module is opened once; one that fails
 * to open, lacks an entry point or refuses the FFT it is handed stays
 * unavailable for the life of the process.  One on oneMKL is not tried at all
 * until the process has handed oneMKL over. */
static const struct fi_api* fi_module_api(struct fi_module* m)
{
	int on_mkl = (0 != strcmp(m->fft, fi_builtin.fft));

	if (!m->tried && !(on_mkl && !fi_fftw_complete())) {

		m->tried = 1;

		void* lib = fi_cpu_runs(m->level) ? fi_open(m->level, m->fft) : NULL;

		if (NULL == lib)
			return NULL;

		if (on_mkl) {

			int (*bind)(void* (*)(const char*)) = (int (*)(void* (*)(const char*)))fi_symbol(lib, "bartorch_fftw_bind");

			if ((NULL == bind) || (0 != bind(fi_fftw_lookup)))
				return NULL;
		}

		m->api.default_opts = fi_symbol(lib, "finufftf_default_opts");
		m->api.makeplan = fi_symbol(lib, "finufftf_makeplan");
		m->api.setpts = fi_symbol(lib, "finufftf_setpts");
		m->api.execute = fi_symbol(lib, "finufftf_execute");
		m->api.destroy = fi_symbol(lib, "finufftf_destroy");

		if (m->api.default_opts && m->api.makeplan && m->api.setpts && m->api.execute && m->api.destroy) {

			m->api.level = m->level;
			m->api.fft = m->fft;
		}
	}

	return (NULL != m->api.level) ? &m->api : NULL;
}

/* The build of FINUFFT at `level` on `fft`, where this library carries it and
 * this processor runs it, or NULL.  An empty `fft` is oneMKL where the process
 * has handed it over, and the library's own FFT otherwise. */
static const struct fi_api* fi_level(const char* level, const char* fft)
{
	if ('\0' == fft[0]) {

		const struct fi_api* api = fi_level(level, "mkl");

		return (NULL != api) ? api : fi_level(level, fi_builtin.fft);
	}

	if ((0 == strcmp(level, fi_builtin.level)) && (0 == strcmp(fft, fi_builtin.fft)))
		return &fi_builtin;

	for (struct fi_module* m = fi_modules; NULL != m->level; m++)
		if ((0 == strcmp(level, m->level)) && (0 == strcmp(fft, m->fft)))
			return fi_module_api(m);

	return NULL;
}

/* What was asked for, or the newest level available on the preferred FFT.
 * Called under fi_lock. */
static const struct fi_api* fi_api(void)
{
	if (NULL != fi_current)
		return fi_current;

	if (!fi_environment_read) {

		fi_environment_read = 1;

		const char* level = getenv("BARTORCH_FINUFFT_SIMD");

		if ((NULL != level) && (strlen(level) < sizeof fi_requested))
			strcpy(fi_requested, level);
	}

	if ('\0' != fi_requested[0])
		fi_current = fi_level(fi_requested, fi_requested_fft);

	int newest = (int)(sizeof fi_modules / sizeof fi_modules[0]) - 2;

	for (int i = newest; (NULL == fi_current) && (i >= 0); i--)
		fi_current = fi_level(fi_modules[i].level, fi_requested_fft);

	if (NULL == fi_current)
		fi_current = fi_level(fi_builtin.level, fi_requested_fft);

	if (NULL == fi_current)
		fi_current = &fi_builtin;

	return fi_current;
}

/* Hand over one of oneMKL's FFTW3 entry points, by name.  Fails for a name a
 * module on oneMKL does not call.  Plans made from then on are made on
 * oneMKL, once all of them are here, unless an FFT was asked for. */
int bartorch_finufft_fftw_set(const char* name, void* address)
{
	int ret = -1;

	pthread_mutex_lock(&fi_lock);

	for (int i = 0; i < FI_FFTW_COUNT; i++) {

		if (0 == strcmp(name, fi_fftw_names[i])) {

			fi_fftw[i] = address;
			fi_current = NULL;
			ret = 0;
		}
	}

	pthread_mutex_unlock(&fi_lock);

	return ret;
}

/* The instruction-set levels FINUFFT's CPU transform is built for, the one
 * compiled into this library first, separated by commas. */
const char* bartorch_finufft_simd_built(void)
{
	static char levels[128];

	pthread_mutex_lock(&fi_lock);

	if ('\0' == levels[0]) {

		strcpy(levels, fi_builtin.level);

		for (const struct fi_module* m = fi_modules; NULL != m->level; m++) {

			if (0 != strcmp(m->fft, fi_builtin.fft))
				continue;

			strcat(levels, ",");
			strcat(levels, m->level);
		}
	}

	pthread_mutex_unlock(&fi_lock);

	return levels;
}

/* The FFTs FINUFFT's CPU transform is built on, the library's own first,
 * separated by commas. */
const char* bartorch_finufft_fft_built(void)
{
	static char ffts[64];

	pthread_mutex_lock(&fi_lock);

	if ('\0' == ffts[0]) {

		strcpy(ffts, fi_builtin.fft);

		for (const struct fi_module* m = fi_modules; NULL != m->level; m++) {

			if ((0 == strcmp(m->fft, fi_builtin.fft)) || (NULL != strstr(ffts, m->fft)))
				continue;

			strcat(ffts, ",");
			strcat(ffts, m->fft);
		}
	}

	pthread_mutex_unlock(&fi_lock);

	return ffts;
}

/* The level plans on the host are made at from now on. */
const char* bartorch_finufft_simd(void)
{
	pthread_mutex_lock(&fi_lock);
	const char* level = fi_api()->level;
	pthread_mutex_unlock(&fi_lock);

	return level;
}

/* The FFT plans on the host are made on from now on. */
const char* bartorch_finufft_fft(void)
{
	pthread_mutex_lock(&fi_lock);
	const char* fft = fi_api()->fft;
	pthread_mutex_unlock(&fi_lock);

	return fft;
}

/* Make plans at `level` on `fft`, where either may be NULL or "" for the
 * default.  Fails, and changes nothing, for a combination this build does not
 * carry, this processor does not run or the process has no FFT for.  Plans
 * already made keep theirs. */
static int fi_request(const char* level, const char* fft)
{
	level = (NULL == level) ? "" : level;
	fft = (NULL == fft) ? "" : fft;

	if ((strlen(level) >= sizeof fi_requested) || (strlen(fft) >= sizeof fi_requested_fft))
		return -1;

	pthread_mutex_lock(&fi_lock);

	fi_environment_read = 1;

	char old_level[sizeof fi_requested];
	char old_fft[sizeof fi_requested_fft];

	strcpy(old_level, fi_requested);
	strcpy(old_fft, fi_requested_fft);

	const struct fi_api* old = fi_current;

	strcpy(fi_requested, level);
	strcpy(fi_requested_fft, fft);
	fi_current = NULL;

	const struct fi_api* api = fi_api();
	int ret = 0;

	if ((('\0' != level[0]) && (0 != strcmp(level, api->level)))
			|| (('\0' != fft[0]) && (0 != strcmp(fft, api->fft)))) {

		strcpy(fi_requested, old_level);
		strcpy(fi_requested_fft, old_fft);
		fi_current = old;
		ret = -1;
	}

	pthread_mutex_unlock(&fi_lock);

	return ret;
}

int bartorch_finufft_set_simd(const char* level)
{
	pthread_mutex_lock(&fi_lock);
	fi_api();
	char fft[sizeof fi_requested_fft];
	strcpy(fft, fi_requested_fft);
	pthread_mutex_unlock(&fi_lock);

	return fi_request(level, fft);
}

int bartorch_finufft_set_fft(const char* fft)
{
	pthread_mutex_lock(&fi_lock);
	fi_api();
	char level[sizeof fi_requested];
	strcpy(level, fi_requested);
	pthread_mutex_unlock(&fi_lock);

	return fi_request(level, fft);
}

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
	const struct fi_api* api;
	void* plan;
};

static void destroy(int device, const struct fi_api* api, void* plan)
{
#ifdef USE_CUDA
	if (device) {

		cufinufftf_destroy(plan);
		return;
	}
#endif
	(void)device;
	api->destroy(plan);
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
	const struct fi_api* api = NULL;
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

		api = fi_api();

		finufft_opts opts;
		api->default_opts(&opts);

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
		ret = api->makeplan(type, dim, n_modes, isign, ntrans, (float)eps, &q, &opts);
		p = q;
	}

	pthread_mutex_unlock(&fi_lock);

	if (0 != ret)
		return -1;

	struct bartorch_fi_plan* held = malloc(sizeof *held);

	if (NULL == held) {

		destroy(device, api, p);
		return -1;
	}

	held->device = device;
	held->api = api;
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
	return p->api->setpts(p->plan, M, x, y, z, 0, NULL, NULL, NULL);
}

int bartorch_finufft_exec(void* plan, complex float* c, complex float* f)
{
	const struct bartorch_fi_plan* p = plan;

#ifdef USE_CUDA
	if (p->device)
		return cufinufftf_execute(p->plan, (cuFloatComplex*)c, (cuFloatComplex*)f);
#endif
	return p->api->execute(p->plan, c, f);
}

void bartorch_finufft_free(void* plan)
{
	if (NULL == plan)
		return;

	struct bartorch_fi_plan* p = plan;

	pthread_mutex_lock(&fi_lock);
	destroy(p->device, p->api, p->plan);
	pthread_mutex_unlock(&fi_lock);

	free(p);

#pragma omp atomic
	fi_live_plans--;
}

int64_t bartorch_finufft_live_plans(void)
{
	return fi_live_plans;
}
