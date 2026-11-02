/*
 * Neighbourhood attention over a 3D grid, on a card.
 *
 * Each voxel attends to the w x w x w voxels around it.  Near an edge the
 * window is shifted inwards rather than cut, so every voxel sees w^3
 * neighbours: along an axis of n voxels the window of voxel i starts at
 * min(max(i - w/2, 0), n - w), which needs n >= w.
 *
 * Three passes make up the operation and its gradient.  The arrays are
 * (groups, voxels, channels), channels fastest, where a group is one head
 * of one item; the voxels are z-major (x fastest) and the w^3 neighbours of
 * a voxel are ordered the same way.  Weights between a voxel and its
 * neighbours are (groups, voxels, w^3) in float32 throughout.
 *
 *   gather_dot   s[i, m] = sum_c a[i, c] b[n(i, m), c]        logits; dL/dattn
 *   gather_sum   o[i, c] = sum_m s[i, m] b[n(i, m), c]        output; dL/dq
 *   scatter_sum  o[j, c] = sum_{i, m : n(i, m) = j} s[i, m] a[i, c]   dL/dv; dL/dk
 *
 * scatter_sum is written as a gather over the voxels whose window holds j,
 * so it needs no atomics and its sums are the same from run to run.
 *
 * Channels are a template parameter, and the element type is float, half or
 * bfloat16 with every sum in float.  Each pass runs on the caller's stream
 * and returns 0, or the CUDA error it met.  No torch here: the host owns the
 * memory and hands over its pointers.
 */
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdint.h>

#include "include/bartorch.h"

namespace {

struct grid3 {
	int nz, ny, nx;  /* voxels per axis */
	int w;           /* window edge */
	int64_t groups;  /* heads times items */
};

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

/* First voxel of the window of voxel i along an axis of n voxels. */
__device__ __forceinline__ int window_start(int i, int n, int w)
{
	int s = i - w / 2;
	s = (s < 0) ? 0 : s;
	return (s > n - w) ? n - w : s;
}

template <typename T, int C>
__global__ void gather_dot(grid3 g, const T* __restrict__ a, const T* __restrict__ b, float* __restrict__ s)
{
	const int64_t voxels = (int64_t)g.nz * g.ny * g.nx;
	const int w3 = g.w * g.w * g.w;

	for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < g.groups * voxels;
			t += (int64_t)blockDim.x * gridDim.x) {

		const int64_t group = t / voxels;
		const int64_t i = t % voxels;
		const int x = (int)(i % g.nx), y = (int)((i / g.nx) % g.ny), z = (int)(i / ((int64_t)g.nx * g.ny));
		const int sz = window_start(z, g.nz, g.w), sy = window_start(y, g.ny, g.w), sx = window_start(x, g.nx, g.w);

		float q[C];
#pragma unroll
		for (int c = 0; c < C; c++)
			q[c] = to_float(a[t * C + c]);

		const T* base = b + group * voxels * C;
		float* out = s + t * w3;
		int m = 0;

		for (int dz = 0; dz < g.w; dz++)
			for (int dy = 0; dy < g.w; dy++)
				for (int dx = 0; dx < g.w; dx++, m++) {

					const T* kv = base + (((int64_t)(sz + dz) * g.ny + (sy + dy)) * g.nx + (sx + dx)) * C;
					float acc = 0.f;
#pragma unroll
					for (int c = 0; c < C; c++)
						acc += q[c] * to_float(kv[c]);
					out[m] = acc;
				}
	}
}

template <typename T, int C>
__global__ void gather_sum(grid3 g, const float* __restrict__ s, const T* __restrict__ b, T* __restrict__ o)
{
	const int64_t voxels = (int64_t)g.nz * g.ny * g.nx;
	const int w3 = g.w * g.w * g.w;

	for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < g.groups * voxels;
			t += (int64_t)blockDim.x * gridDim.x) {

		const int64_t group = t / voxels;
		const int64_t i = t % voxels;
		const int x = (int)(i % g.nx), y = (int)((i / g.nx) % g.ny), z = (int)(i / ((int64_t)g.nx * g.ny));
		const int sz = window_start(z, g.nz, g.w), sy = window_start(y, g.ny, g.w), sx = window_start(x, g.nx, g.w);

		float acc[C];
#pragma unroll
		for (int c = 0; c < C; c++)
			acc[c] = 0.f;

		const T* base = b + group * voxels * C;
		const float* weights = s + t * w3;
		int m = 0;

		for (int dz = 0; dz < g.w; dz++)
			for (int dy = 0; dy < g.w; dy++)
				for (int dx = 0; dx < g.w; dx++, m++) {

					const float wm = weights[m];
					const T* kv = base + (((int64_t)(sz + dz) * g.ny + (sy + dy)) * g.nx + (sx + dx)) * C;
#pragma unroll
					for (int c = 0; c < C; c++)
						acc[c] += wm * to_float(kv[c]);
				}

#pragma unroll
		for (int c = 0; c < C; c++)
			o[t * C + c] = from_float<T>(acc[c]);
	}
}

/* The voxels along an axis whose window holds voxel j, and j's place in each window. */
__device__ __forceinline__ int holders(int j, int n, int w, int* who, int* where)
{
	int count = 0;
	const int lo = (j - w + 1 < 0) ? 0 : j - w + 1;
	const int hi = (j + w - 1 > n - 1) ? n - 1 : j + w - 1;

	for (int i = lo; i <= hi; i++) {

		const int s = window_start(i, n, w);

		if (s <= j && j < s + w) {

			who[count] = i;
			where[count] = j - s;
			count++;
		}
	}

	return count;
}

#define BARTORCH_NA_MAX_WINDOW 9

template <typename T, int C>
__global__ void scatter_sum(grid3 g, const float* __restrict__ s, const T* __restrict__ a, T* __restrict__ o)
{
	const int64_t voxels = (int64_t)g.nz * g.ny * g.nx;
	const int w3 = g.w * g.w * g.w;

	for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < g.groups * voxels;
			t += (int64_t)blockDim.x * gridDim.x) {

		const int64_t group = t / voxels;
		const int64_t j = t % voxels;
		const int x = (int)(j % g.nx), y = (int)((j / g.nx) % g.ny), z = (int)(j / ((int64_t)g.nx * g.ny));

		int iz[2 * BARTORCH_NA_MAX_WINDOW], mz[2 * BARTORCH_NA_MAX_WINDOW];
		int iy[2 * BARTORCH_NA_MAX_WINDOW], my[2 * BARTORCH_NA_MAX_WINDOW];
		int ix[2 * BARTORCH_NA_MAX_WINDOW], mx[2 * BARTORCH_NA_MAX_WINDOW];
		const int cz = holders(z, g.nz, g.w, iz, mz);
		const int cy = holders(y, g.ny, g.w, iy, my);
		const int cx = holders(x, g.nx, g.w, ix, mx);

		float acc[C];
#pragma unroll
		for (int c = 0; c < C; c++)
			acc[c] = 0.f;

		const float* sbase = s + group * voxels * w3;
		const T* abase = a + group * voxels * C;

		for (int pz = 0; pz < cz; pz++)
			for (int py = 0; py < cy; py++)
				for (int px = 0; px < cx; px++) {

					const int64_t i = ((int64_t)iz[pz] * g.ny + iy[py]) * g.nx + ix[px];
					const int m = (mz[pz] * g.w + my[py]) * g.w + mx[px];
					const float wm = sbase[i * w3 + m];
					const T* av = abase + i * C;
#pragma unroll
					for (int c = 0; c < C; c++)
						acc[c] += wm * to_float(av[c]);
				}

#pragma unroll
		for (int c = 0; c < C; c++)
			o[t * C + c] = from_float<T>(acc[c]);
	}
}

enum pass { GATHER_DOT, GATHER_SUM, SCATTER_SUM };

template <typename T, int C>
cudaError_t launch(enum pass p, grid3 g, const void* x, const void* y, void* out, cudaStream_t stream)
{
	const int64_t work = g.groups * (int64_t)g.nz * g.ny * g.nx;
	const int threads = 128;
	int64_t blocks = (work + threads - 1) / threads;
	blocks = (blocks > 65535 * 16) ? 65535 * 16 : blocks;

	switch (p) {
	case GATHER_DOT:
		gather_dot<T, C><<<(unsigned)blocks, threads, 0, stream>>>(g, (const T*)x, (const T*)y, (float*)out);
		break;
	case GATHER_SUM:
		gather_sum<T, C><<<(unsigned)blocks, threads, 0, stream>>>(g, (const float*)x, (const T*)y, (T*)out);
		break;
	case SCATTER_SUM:
		scatter_sum<T, C><<<(unsigned)blocks, threads, 0, stream>>>(g, (const float*)x, (const T*)y, (T*)out);
		break;
	}

	return cudaGetLastError();
}

template <typename T>
cudaError_t by_channels(enum pass p, grid3 g, int channels, const void* x, const void* y, void* out, cudaStream_t stream)
{
	switch (channels) {
	case 8: return launch<T, 8>(p, g, x, y, out, stream);
	case 16: return launch<T, 16>(p, g, x, y, out, stream);
	case 32: return launch<T, 32>(p, g, x, y, out, stream);
	case 64: return launch<T, 64>(p, g, x, y, out, stream);
	default: return cudaErrorInvalidValue;
	}
}

int run(enum pass p, const void* x, const void* y, void* out, int64_t groups, int nz, int ny, int nx,
		int channels, int window, int dtype, void* stream)
{
	if (window < 1 || window > BARTORCH_NA_MAX_WINDOW || nz < window || ny < window || nx < window || groups < 1)
		return (int)cudaErrorInvalidValue;

	grid3 g = { nz, ny, nx, window, groups };
	cudaStream_t s = (cudaStream_t)stream;
	cudaError_t e;

	switch (dtype) {
	case BARTORCH_NA_FLOAT32: e = by_channels<float>(p, g, channels, x, y, out, s); break;
	case BARTORCH_NA_FLOAT16: e = by_channels<__half>(p, g, channels, x, y, out, s); break;
	case BARTORCH_NA_BFLOAT16: e = by_channels<__nv_bfloat16>(p, g, channels, x, y, out, s); break;
	default: e = cudaErrorInvalidValue;
	}

	return (cudaSuccess == e) ? 0 : (int)e;
}

} /* namespace */

extern "C" {

int bartorch_na3d_gather_dot(const void* a, const void* b, float* s, int64_t groups, int nz, int ny, int nx,
		int channels, int window, int dtype, void* stream)
{
	return run(GATHER_DOT, a, b, s, groups, nz, ny, nx, channels, window, dtype, stream);
}

int bartorch_na3d_gather_sum(const float* s, const void* b, void* o, int64_t groups, int nz, int ny, int nx,
		int channels, int window, int dtype, void* stream)
{
	return run(GATHER_SUM, s, b, o, groups, nz, ny, nx, channels, window, dtype, stream);
}

int bartorch_na3d_scatter_sum(const float* s, const void* a, void* o, int64_t groups, int nz, int ny, int nx,
		int channels, int window, int dtype, void* stream)
{
	return run(SCATTER_SUM, s, a, o, groups, nz, ny, nx, channels, window, dtype, stream);
}

int bartorch_na3d_built(void)
{
	return 1;
}

}
