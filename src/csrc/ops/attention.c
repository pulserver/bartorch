/*
 * The neighbourhood-attention entry points of a build without CUDA: the
 * passes are in attention.cu, and here they report that they are absent.
 */
#include <stdint.h>

#include "include/bartorch.h"

#ifndef USE_CUDA

int bartorch_na3d_built(void) { return 0; }

int bartorch_na3d_gather_dot(const void* a, const void* b, float* s, int64_t groups,
		int nz, int ny, int nx, int channels, int window, int dtype, void* stream)
{
	(void)a; (void)b; (void)s; (void)groups; (void)nz; (void)ny; (void)nx;
	(void)channels; (void)window; (void)dtype; (void)stream;
	return -1;
}

int bartorch_na3d_gather_sum(const float* s, const void* b, void* o, int64_t groups,
		int nz, int ny, int nx, int channels, int window, int dtype, void* stream)
{
	(void)s; (void)b; (void)o; (void)groups; (void)nz; (void)ny; (void)nx;
	(void)channels; (void)window; (void)dtype; (void)stream;
	return -1;
}

int bartorch_na3d_scatter_sum(const float* s, const void* a, void* o, int64_t groups,
		int nz, int ny, int nx, int channels, int window, int dtype, void* stream)
{
	(void)s; (void)a; (void)o; (void)groups; (void)nz; (void)ny; (void)nx;
	(void)channels; (void)window; (void)dtype; (void)stream;
	return -1;
}

#endif
