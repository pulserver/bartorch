/*
 * iGROG gridding on the card, for bartorch.tools.igrog: the host loop of
 * grog.c with one target per warp.  The warp's 32 lanes interpolate Q from
 * the table into the packed lower triangle of R_ss + mu I in shared memory
 * (Q(-d) = Q(d)^H, so only the blocks Q(d_r - d_s), r >= s; a lane per
 * coil pair, so the table rows are read coalesced), factor it by Cholesky
 * (a lane per row), and solve W a row at a time (a lane per entry),
 * interpolating that row of R_ts as it goes; the row is then applied to every
 * set by a warp reduction.  Single precision, no workspace: the card holds
 * the inputs and the results, and each warp n (n + 1) / 2 + n complex floats
 * and 3 sources floats of shared memory (about 1.6 KB at 18).
 *
 * An earlier version kept a target per thread with the system in local
 * memory; at 1.8 KB a thread that streamed through L2 and ran at about
 * 1.4 M targets a second on an A40.
 */
#include <cuda_runtime.h>
#include <stdint.h>

#include "include/bartorch.h"

namespace {

constexpr int WARPS = 4; /* per block */
constexpr unsigned FULL = 0xffffffffu;

struct cf {
	float x, y;
};

__device__ __forceinline__ cf operator+(cf a, cf b) { return { a.x + b.x, a.y + b.y }; }
__device__ __forceinline__ cf operator-(cf a, cf b) { return { a.x - b.x, a.y - b.y }; }
__device__ __forceinline__ cf operator*(cf a, cf b) { return { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x }; }
__device__ __forceinline__ cf operator*(float s, cf a) { return { s * a.x, s * a.y }; }
__device__ __forceinline__ cf conj(cf a) { return { a.x, -a.y }; }
/* a * conj(b) */
__device__ __forceinline__ cf mulc(cf a, cf b) { return { a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y }; }

__device__ __forceinline__ float warp_sum(float v)
{
	for (int o = 16; o > 0; o >>= 1)
		v += __shfl_xor_sync(FULL, v, o);
	return v;
}

__device__ __forceinline__ cf warp_sum(cf v)
{
	return { warp_sum(v.x), warp_sum(v.y) };
}

struct params {
	int ndim;
	int C, D, reach, upsample;
	float lam;
	const cf* table;
	int64_t shots, P;
	const float* traj;
	int sets;
	const cf* const* data;
	int64_t targets;
	const int64_t* shot;
	const int64_t* sample;
	const int64_t* src;
	cf* const* out;
	float* noise;
	int* status;
};

__device__ void lagrange4(float t, float w[4])
{
	w[0] = -t * (t - 1) * (t - 2) / 6;
	w[1] = (t + 1) * (t - 1) * (t - 2) / 2;
	w[2] = -(t + 1) * t * (t - 2) / 2;
	w[3] = (t + 1) * t * (t - 1) / 6;
}

/* Where disp falls in the table: the first node and the weights along each axis; false outside it. */
struct taps {
	int64_t base[3];
	float w[3][4];
};

__device__ bool locate(const params& p, const float* disp, taps& t)
{
	const int64_t n = 2 * (int64_t)p.reach * p.upsample + 1;
	for (int a = 0; a < 3; a++) {
		t.base[a] = 0;
		t.w[a][0] = 1;
		t.w[a][1] = t.w[a][2] = t.w[a][3] = 0;
	}
	for (int a = 0; a < p.ndim; a++) {
		const float u = (disp[a] + (float)p.reach) * (float)p.upsample;
		const float f = floorf(u);
		t.base[a] = (int64_t)f - 1;
		if (t.base[a] < 0 || t.base[a] + 3 >= n)
			return false;
		lagrange4(u - f, t.w[a]);
	}
	return true;
}

/* Q_e(disp), entry e = c C + c2 of the (C, C) matrix at the taps t. */
__device__ cf interpolate(const params& p, const taps& t, int e)
{
	const int64_t n = 2 * (int64_t)p.reach * p.upsample + 1;
	const int64_t CC = (int64_t)p.C * p.C;
	const int t1 = p.ndim > 1 ? 4 : 1, t2 = p.ndim > 2 ? 4 : 1;
	cf acc = { 0.f, 0.f };
	for (int i0 = 0; i0 < 4; i0++)
		for (int i1 = 0; i1 < t1; i1++)
			for (int i2 = 0; i2 < t2; i2++) {
				int64_t flat = t.base[0] + i0;
				float wt = t.w[0][i0];
				if (p.ndim > 1) {
					flat = flat * n + t.base[1] + i1;
					wt *= t.w[1][i1];
				}
				if (p.ndim > 2) {
					flat = flat * n + t.base[2] + i2;
					wt *= t.w[2][i2];
				}
				acc = acc + wt * p.table[flat * CC + e];
			}
	return acc;
}

/* Packed lower triangle: entry (i, k), k <= i. */
__device__ __forceinline__ int tri(int i, int k) { return i * (i + 1) / 2 + k; }

/* Every lane of a warp calls it with the same values, so every branch below is uniform across the warp. */
__device__ void fail(int lane, int* status, int code)
{
	if (lane == 0)
		atomicMax(status, code);
}

__global__ void __launch_bounds__(WARPS * 32) grog_kernel(params p)
{
	extern __shared__ float2 shared[];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int64_t m = (int64_t)blockIdx.x * WARPS + warp;
	if (m >= p.targets)
		return;

	const int C = p.C, D = p.D, n = C * D, nd = p.ndim;
	const int per_warp = n * (n + 1) / 2 + n + (3 * D + 1) / 2;
	cf* A = (cf*)(shared + (size_t)warp * per_warp); /* packed lower triangle */
	cf* z = A + n * (n + 1) / 2;                       /* one row of W */
	float* off = (float*)(z + n);                      /* (D, 3) source offsets */

	const int64_t s = p.shot[m], j = p.sample[m];
	const int64_t* js = p.src + m * D;
	bool bad = s < 0 || s >= p.shots || j < 0 || j >= p.P;
	for (int i = 0; i < D && !bad; i++)
		bad = js[i] < 0 || js[i] >= p.P;
	if (bad)
		return fail(lane, p.status, 4);

	const float* tj = p.traj + (s * p.P + j) * nd;
	for (int e = lane; e < D * nd; e += 32) {
		const int i = e / nd, a = e % nd;
		off[i * 3 + a] = p.traj[(s * p.P + js[i]) * nd + a] - nearbyintf(tj[a]);
	}
	__syncwarp();

	/* lower block (r, k), r >= k: Q(d_r - d_k); a lane per coil pair */
	for (int r = 0; r < D; r++)
		for (int k = 0; k <= r; k++) {
			float dd[3] = { 0, 0, 0 };
			for (int a = 0; a < nd; a++)
				dd[a] = off[r * 3 + a] - off[k * 3 + a];
			taps t;
			if (!locate(p, dd, t))
				return fail(lane, p.status, 6);
			for (int e = lane; e < C * C; e += 32) {
				const int i = r * C + e / C, l = k * C + e % C;
				if (l <= i)
					A[tri(i, l)] = interpolate(p, t, e);
			}
		}
	/* every -d_k must be in the table too, checked here while the warp is uniform */
	for (int k = 0; k < D; k++) {
		float dd[3] = { 0, 0, 0 };
		for (int a = 0; a < nd; a++)
			dd[a] = -off[k * 3 + a];
		taps t;
		if (!locate(p, dd, t))
			return fail(lane, p.status, 6);
	}
	__syncwarp();

	float tr = 0;
	for (int i = lane; i < n; i += 32)
		tr += A[tri(i, i)].x;
	const float mu = p.lam * warp_sum(tr) / (float)n;
	for (int i = lane; i < n; i += 32)
		A[tri(i, i)].x += mu;
	__syncwarp();

	/* Cholesky, left-looking: column c, a lane per row below it */
	for (int c = 0; c < n; c++) {
		float part = 0;
		for (int k = lane; k < c; k += 32)
			part += A[tri(c, k)].x * A[tri(c, k)].x + A[tri(c, k)].y * A[tri(c, k)].y;
		const float d2 = A[tri(c, c)].x - warp_sum(part);
		if (!(d2 > 0))
			return fail(lane, p.status, 5);
		const float d = sqrtf(d2), inv = 1.f / d;
		for (int i = c + 1 + lane; i < n; i += 32) {
			cf acc = A[tri(i, c)];
			for (int k = 0; k < c; k++)
				acc = acc - mulc(A[tri(i, k)], A[tri(c, k)]);
			A[tri(i, c)] = inv * acc;
		}
		__syncwarp();
		if (lane == 0)
			A[tri(c, c)] = { d, 0.f };
		__syncwarp();
	}

	/* W[c, :] = (A^-1 R_ts[c, :]^H)^H, R_ts[c, (k, c2)] = Q_{c c2}(-d_k); out_c = W[c, :] x per set */
	float gain = 0;
	for (int c = 0; c < C; c++) {
		for (int e = lane; e < n; e += 32) {
			const int k = e / C, c2 = e % C;
			float dd[3] = { 0, 0, 0 };
			for (int a = 0; a < nd; a++)
				dd[a] = -off[k * 3 + a];
			taps t;
			locate(p, dd, t); /* inside, checked above */
			z[e] = conj(interpolate(p, t, c * C + c2));
		}
		__syncwarp();
		/* L y = z, a column at a time: a lane per row below it */
		for (int i = 0; i < n; i++) {
			const cf zi = (1.f / A[tri(i, i)].x) * z[i];
			__syncwarp();
			if (lane == 0)
				z[i] = zi;
			for (int r = i + 1 + lane; r < n; r += 32)
				z[r] = z[r] - A[tri(r, i)] * zi;
			__syncwarp();
		}
		/* L^H w = y, from the bottom */
		for (int i = n - 1; i >= 0; i--) {
			const cf zi = (1.f / A[tri(i, i)].x) * z[i];
			__syncwarp();
			if (lane == 0)
				z[i] = zi;
			for (int r = lane; r < i; r += 32)
				z[r] = z[r] - conj(A[tri(i, r)]) * zi;
			__syncwarp();
		}
		float g = 0;
		for (int e = lane; e < n; e += 32)
			g += z[e].x * z[e].x + z[e].y * z[e].y;
		gain += warp_sum(g);
		for (int e = 0; e < p.sets; e++) {
			const cf* x = p.data[e];
			cf acc = { 0.f, 0.f };
			for (int q = lane; q < n; q += 32) {
				const int i = q / C, c2 = q % C;
				acc = acc + mulc(x[((int64_t)c2 * p.shots + s) * p.P + js[i]], z[q]);
			}
			acc = warp_sum(acc);
			if (lane == 0)
				p.out[e][(int64_t)c * p.targets + m] = acc;
		}
		__syncwarp();
	}
	if (p.noise && lane == 0)
		p.noise[m] = gain / (float)C;
}

}  // namespace

extern "C" {

int bartorch_grog_cuda_built(void)
{
	return 1;
}

int bartorch_grog_grid_cuda(int ndim, int64_t coils, int64_t sources, int64_t reach, int64_t upsample, float lam,
		const void* table, int64_t shots, int64_t samples, const float* traj, int64_t sets, const void* data,
		int64_t targets, const int64_t* shot, const int64_t* sample, const int64_t* src, const void* out,
		float* noise, void* status, void* stream)
{
	const int64_t n = coils * sources;
	if ((ndim != 2 && ndim != 3) || coils < 1 || sources < 1 || sources > samples || sets < 1)
		return -2;
	if (n > 48 || (targets + WARPS - 1) / WARPS > 2147483647LL)
		return -2;
	if (targets < 1)
		return 0;

	params p;
	p.ndim = ndim;
	p.C = (int)coils;
	p.D = (int)sources;
	p.reach = (int)reach;
	p.upsample = (int)upsample;
	p.lam = lam;
	p.table = (const cf*)table;
	p.shots = shots;
	p.P = samples;
	p.traj = traj;
	p.sets = (int)sets;
	p.data = (const cf* const*)data;
	p.targets = targets;
	p.shot = shot;
	p.sample = sample;
	p.src = src;
	p.out = (cf* const*)out;
	p.noise = noise;
	p.status = (int*)status;

	/* per warp: the packed triangle, one row of W, and the offsets, in float2 units */
	const size_t per_warp = (size_t)(n * (n + 1) / 2 + n + (3 * sources + 1) / 2);
	const size_t bytes = per_warp * WARPS * sizeof(float2);
	const int64_t blocks = (targets + WARPS - 1) / WARPS;
	grog_kernel<<<(unsigned)blocks, WARPS * 32, bytes, (cudaStream_t)stream>>>(p);
	return (int)cudaGetLastError();
}

}  // extern "C"
