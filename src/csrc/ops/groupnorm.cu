/*
 * Group normalisation, a per-item scale and shift, and SiLU, in one pass over memory.
 *
 * The block every diffusion-style UNet repeats -- GroupNorm, then the noise
 * level's scale and shift (FiLM), then SiLU -- as eager PyTorch runs it reads
 * and writes the activations four or five times, and in float16 networks it
 * adds two casts around the norm.  Here the statistics take one read, and the
 * normalisation, affine, scale, shift and SiLU one more read and one write:
 *
 *   y[n, c, v] = act( ((x[n, c, v] - mean[n, g]) * rstd[n, g] * w[c] + b[c]) * (1 + s[n, c]) + t[n, c] )
 *
 * with the norm's affine w, b and the FiLM s, t each optional (NULL), and act
 * SiLU when asked for, the identity otherwise.  x and y are
 * (items, channels, voxels), voxels fastest -- a contiguous NC(D)HW tensor --
 * or with channels_last (items, voxels, channels), channels fastest -- a
 * channels-last N(D)HWC one, which cuDNN's convolutions prefer -- in float,
 * half or bfloat16; w, b are (channels) float32, s, t (items,
 * channels) float32.  Statistics are in float32 by Welford's method; a group
 * is spread over several blocks, whose partial sums a second pass merges.
 * y may be x.
 *
 * The caller hands over a float32 workspace of bartorch_gn_workspace(...)
 * floats.  Each pass runs on the caller's stream; the entry point returns 0
 * or the CUDA error it met.
 */
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdint.h>

#include "include/bartorch.h"

namespace {

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

const int THREADS = 256;
/* Elements of a group each statistics block takes, at least. */
const int64_t CHUNK = 16384;
const int MAX_PARTS = 64;

int64_t parts_of(int64_t group_size)
{
	int64_t p = (group_size + CHUNK - 1) / CHUNK;
	return (p < 1) ? 1 : (p > MAX_PARTS) ? MAX_PARTS : p;
}

struct welford {
	float count, mean, m2;
};

__device__ __forceinline__ welford merge(welford a, welford b)
{
	const float count = a.count + b.count;
	if (count == 0.f)
		return a;
	const float delta = b.mean - a.mean;
	const float fb = b.count / count;
	welford r;
	r.count = count;
	r.mean = a.mean + delta * fb;
	r.m2 = a.m2 + b.m2 + delta * delta * a.count * fb;
	return r;
}

__device__ __forceinline__ welford shuffle_down(welford w, int offset)
{
	welford r;
	r.count = __shfl_down_sync(0xffffffffu, w.count, offset);
	r.mean = __shfl_down_sync(0xffffffffu, w.mean, offset);
	r.m2 = __shfl_down_sync(0xffffffffu, w.m2, offset);
	return r;
}

__device__ welford block_merge(welford w)
{
	__shared__ welford warps[THREADS / 32];

	for (int offset = 16; offset > 0; offset /= 2)
		w = merge(w, shuffle_down(w, offset));

	const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
	if (lane == 0)
		warps[warp] = w;
	__syncthreads();

	if (warp == 0) {
		w = (lane < THREADS / 32) ? warps[lane] : welford{ 0.f, 0.f, 0.f };
		for (int offset = 16; offset > 0; offset /= 2)
			w = merge(w, shuffle_down(w, offset));
	}
	return w;  /* valid in thread 0 */
}

/* Partial statistics of part blockIdx.y of group blockIdx.x (an item's group). */
template <typename T>
__global__ void partial_stats(const T* __restrict__ x, int64_t group_size, int parts, float* __restrict__ partial)
{
	const int64_t group = blockIdx.x;
	const int part = blockIdx.y;
	const int64_t per = (group_size + parts - 1) / parts;
	const int64_t lo = part * per;
	const int64_t hi = (lo + per < group_size) ? lo + per : group_size;
	const T* base = x + group * group_size;

	welford w = { 0.f, 0.f, 0.f };
	for (int64_t i = lo + threadIdx.x; i < hi; i += THREADS) {
		const float v = to_float(base[i]);
		w.count += 1.f;
		const float delta = v - w.mean;
		w.mean += delta / w.count;
		w.m2 += delta * (v - w.mean);
	}
	w = block_merge(w);

	if (threadIdx.x == 0) {
		float* out = partial + (group * parts + part) * 3;
		out[0] = w.count;
		out[1] = w.mean;
		out[2] = w.m2;
	}
}

/* Mean and reciprocal standard deviation of each group from its parts. */
__global__ void finish_stats(const float* __restrict__ partial, int64_t groups, int parts, float eps, float* __restrict__ stats)
{
	for (int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; g < groups; g += (int64_t)blockDim.x * gridDim.x) {
		welford w = { 0.f, 0.f, 0.f };
		for (int p = 0; p < parts; p++) {
			const float* in = partial + (g * parts + p) * 3;
			w = merge(w, welford{ in[0], in[1], in[2] });
		}
		stats[2 * g] = w.mean;
		stats[2 * g + 1] = rsqrtf(w.m2 / ((w.count > 0.f) ? w.count : 1.f) + eps);
	}
}

/* y = act(x * a + b), with a and b worked out once for each row (an item's channel, blockIdx.x). */
template <typename T>
__global__ void apply(const T* x, T* y, int64_t channels, int64_t voxels, int per_group,
		const float* __restrict__ stats, const float* __restrict__ weight, const float* __restrict__ bias,
		const float* __restrict__ scale, const float* __restrict__ shift, int silu)
{
	const int64_t row = blockIdx.x;  /* item * channels + channel */
	const int64_t c = row % channels;
	const int64_t item = row / channels;
	const int64_t group = item * (channels / per_group) + c / per_group;

	const float mean = stats[2 * group], rstd = stats[2 * group + 1];
	const float w = weight ? weight[c] : 1.f;
	const float b = bias ? bias[c] : 0.f;
	const float s = scale ? 1.f + scale[row] : 1.f;
	const float t = shift ? shift[row] : 0.f;
	const float a = rstd * w * s;
	const float d = (b - mean * rstd * w) * s + t;

	const T* in = x + row * voxels;
	T* out = y + row * voxels;
	for (int64_t i = blockIdx.y * (int64_t)THREADS + threadIdx.x; i < voxels; i += (int64_t)THREADS * gridDim.y) {
		float v = to_float(in[i]) * a + d;
		if (silu)
			v = v / (1.f + __expf(-v));
		out[i] = from_float<T>(v);
	}
}

/*
 * Channels-last (items, voxels, channels): the partial statistics of every group of an
 * item over part blockIdx.y of its voxels.  Each thread follows one channel down the
 * voxels (or several, when there are more channels than threads), so a row of channels
 * is read whole by neighbouring threads; the channels of each group are merged at the end.
 */
template <typename T>
__global__ void partial_stats_last(const T* __restrict__ x, int64_t voxels, int channels, int groups, int parts,
		float* __restrict__ partial)
{
	extern __shared__ welford entries[];

	const int64_t item = blockIdx.x;
	const int part = blockIdx.y;
	const int64_t per = (voxels + parts - 1) / parts;
	const int64_t lo = part * per;
	const int64_t hi = (lo + per < voxels) ? lo + per : voxels;
	const T* base = x + item * voxels * channels;
	const int rows = (channels <= THREADS) ? THREADS / channels : 1;

	if (channels <= THREADS) {
		const int t = threadIdx.x;
		if (t < rows * channels) {
			const int r = t / channels, c = t % channels;
			welford w = { 0.f, 0.f, 0.f };
			for (int64_t v = lo + r; v < hi; v += rows) {
				const float value = to_float(base[v * channels + c]);
				w.count += 1.f;
				const float delta = value - w.mean;
				w.mean += delta / w.count;
				w.m2 += delta * (value - w.mean);
			}
			entries[t] = w;
		}
	} else {
		for (int c = threadIdx.x; c < channels; c += THREADS) {
			welford w = { 0.f, 0.f, 0.f };
			for (int64_t v = lo; v < hi; v++) {
				const float value = to_float(base[v * channels + c]);
				w.count += 1.f;
				const float delta = value - w.mean;
				w.mean += delta / w.count;
				w.m2 += delta * (value - w.mean);
			}
			entries[c] = w;
		}
	}
	__syncthreads();

	const int per_group = channels / groups;
	for (int g = threadIdx.x; g < groups; g += THREADS) {
		welford w = { 0.f, 0.f, 0.f };
		for (int r = 0; r < rows; r++)
			for (int j = 0; j < per_group; j++)
				w = merge(w, entries[r * channels + g * per_group + j]);
		float* out = partial + ((item * groups + g) * parts + part) * 3;
		out[0] = w.count;
		out[1] = w.mean;
		out[2] = w.m2;
	}
}

/* Each item's channel as one multiply-add: y = act(x * ab[0] + ab[1]). */
__global__ void coefficients(int64_t items, int channels, int per_group, const float* __restrict__ stats,
		const float* __restrict__ weight, const float* __restrict__ bias, const float* __restrict__ scale,
		const float* __restrict__ shift, float* __restrict__ ab)
{
	for (int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; row < items * channels;
			row += (int64_t)blockDim.x * gridDim.x) {
		const int64_t c = row % channels, item = row / channels;
		const int64_t group = item * (channels / per_group) + c / per_group;
		const float mean = stats[2 * group], rstd = stats[2 * group + 1];
		const float w = weight ? weight[c] : 1.f;
		const float b = bias ? bias[c] : 0.f;
		const float s = scale ? 1.f + scale[row] : 1.f;
		const float t = shift ? shift[row] : 0.f;
		ab[2 * row] = rstd * w * s;
		ab[2 * row + 1] = (b - mean * rstd * w) * s + t;
	}
}

template <typename T>
__global__ void apply_last(const T* x, T* y, int64_t items, int64_t voxels, int channels, const float* __restrict__ ab, int silu)
{
	const int64_t total = items * voxels * channels;
	for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)blockDim.x * gridDim.x) {
		const int64_t c = i % channels, item = i / (voxels * channels);
		const int64_t row = item * channels + c;
		float v = to_float(x[i]) * ab[2 * row] + ab[2 * row + 1];
		if (silu)
			v = v / (1.f + __expf(-v));
		y[i] = from_float<T>(v);
	}
}

template <typename T>
cudaError_t run_last(const void* x, void* y, int64_t items, int64_t channels, int64_t voxels, int groups,
		const float* weight, const float* bias, const float* scale, const float* shift, float eps, int silu,
		float* workspace, cudaStream_t stream)
{
	const int per_group = (int)(channels / groups);
	const int parts = (int)parts_of(per_group * voxels);
	const int64_t n_groups = items * groups;
	float* partial = workspace;
	float* stats = partial + n_groups * parts * 3;
	float* ab = stats + 2 * n_groups;
	const int entries = (channels <= THREADS) ? (int)(THREADS / channels * channels) : (int)channels;

	partial_stats_last<T><<<dim3((unsigned)items, parts), THREADS, entries * sizeof(welford), stream>>>(
			(const T*)x, voxels, (int)channels, groups, parts, partial);
	finish_stats<<<(unsigned)((n_groups + 127) / 128), 128, 0, stream>>>(partial, n_groups, parts, eps, stats);
	coefficients<<<(unsigned)((items * channels + 127) / 128), 128, 0, stream>>>(
			items, (int)channels, per_group, stats, weight, bias, scale, shift, ab);

	int64_t blocks = (items * voxels * channels + THREADS * 8 - 1) / (THREADS * 8);
	blocks = (blocks < 1) ? 1 : (blocks > 65535 * 16) ? 65535 * 16 : blocks;
	apply_last<T><<<(unsigned)blocks, THREADS, 0, stream>>>((const T*)x, (T*)y, items, voxels, (int)channels, ab, silu);

	return cudaGetLastError();
}

template <typename T>
cudaError_t run(const void* x, void* y, int64_t items, int64_t channels, int64_t voxels, int groups,
		const float* weight, const float* bias, const float* scale, const float* shift, float eps, int silu,
		float* workspace, cudaStream_t stream)
{
	const int per_group = (int)(channels / groups);
	const int64_t group_size = per_group * voxels;
	const int parts = (int)parts_of(group_size);
	const int64_t n_groups = items * groups;
	float* partial = workspace;
	float* stats = workspace + n_groups * parts * 3;

	partial_stats<T><<<dim3((unsigned)n_groups, parts), THREADS, 0, stream>>>((const T*)x, group_size, parts, partial);
	finish_stats<<<(unsigned)((n_groups + 127) / 128), 128, 0, stream>>>(partial, n_groups, parts, eps, stats);

	int64_t ychunks = (voxels + THREADS * 8 - 1) / (THREADS * 8);
	ychunks = (ychunks < 1) ? 1 : (ychunks > 65535) ? 65535 : ychunks;
	apply<T><<<dim3((unsigned)(items * channels), (unsigned)ychunks), THREADS, 0, stream>>>(
			(const T*)x, (T*)y, channels, voxels, per_group, stats, weight, bias, scale, shift, silu);

	return cudaGetLastError();
}

} /* namespace */

extern "C" {

int64_t bartorch_gn_workspace(int64_t items, int64_t channels, int64_t voxels, int groups)
{
	if (groups < 1 || channels % groups)
		return -1;
	const int64_t n_groups = items * groups;
	return n_groups * parts_of(channels / groups * voxels) * 3 + 2 * n_groups + 2 * items * channels;
}

int bartorch_gn_silu(const void* x, void* y, int64_t items, int64_t channels, int64_t voxels, int groups,
		const void* weight, const void* bias, const void* scale, const void* shift, float eps, int silu,
		int channels_last, int dtype, void* workspace, void* stream)
{
	if (items < 1 || channels < 1 || voxels < 1 || groups < 1 || channels % groups || items * channels > 2147483647LL)
		return (int)cudaErrorInvalidValue;
	if (channels_last && channels > 4096)  /* the statistics' shared memory holds a Welford sum per channel */
		return (int)cudaErrorInvalidValue;

	const float* w = (const float*)weight;
	const float* b = (const float*)bias;
	const float* s = (const float*)scale;
	const float* t = (const float*)shift;
	float* ws = (float*)workspace;
	cudaStream_t st = (cudaStream_t)stream;
	cudaError_t e;

#define BARTORCH_GN_RUN(T) (channels_last \
		? run_last<T>(x, y, items, channels, voxels, groups, w, b, s, t, eps, silu, ws, st) \
		: run<T>(x, y, items, channels, voxels, groups, w, b, s, t, eps, silu, ws, st))
	switch (dtype) {
	case BARTORCH_NA_FLOAT32: e = BARTORCH_GN_RUN(float); break;
	case BARTORCH_NA_FLOAT16: e = BARTORCH_GN_RUN(__half); break;
	case BARTORCH_NA_BFLOAT16: e = BARTORCH_GN_RUN(__nv_bfloat16); break;
	default: e = cudaErrorInvalidValue;
	}
#undef BARTORCH_GN_RUN

	return (cudaSuccess == e) ? 0 : (int)e;
}

int bartorch_gn_built(void)
{
	return 1;
}

}
