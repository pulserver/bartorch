/*
 * The fused group-norm entry points of a build without CUDA: the passes are
 * in groupnorm.cu, and here they report that they are absent.
 */
#include <stdint.h>

#include "include/bartorch.h"

#ifndef USE_CUDA

int bartorch_gn_built(void) { return 0; }

int64_t bartorch_gn_workspace(int64_t items, int64_t channels, int64_t voxels, int groups)
{
	(void)items; (void)channels; (void)voxels; (void)groups;
	return -1;
}

int bartorch_gn_silu(const void* x, void* y, int64_t items, int64_t channels, int64_t voxels, int groups,
		const void* weight, const void* bias, const void* scale, const void* shift, float eps, int silu,
		int channels_last, int dtype, void* workspace, void* stream)
{
	(void)x; (void)y; (void)items; (void)channels; (void)voxels; (void)groups; (void)weight; (void)bias;
	(void)scale; (void)shift; (void)eps; (void)silu; (void)channels_last; (void)dtype; (void)workspace; (void)stream;
	return -1;
}

#endif
