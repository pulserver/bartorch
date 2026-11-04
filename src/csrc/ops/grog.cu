/*
 * iGROG gridding on the card, for bartorch.tools.igrog: the host loop of
 * grog.c with one target per thread.  Each thread interpolates Q from the
 * table straight into the packed lower triangle of R_ss + mu I (Q(-d) =
 * Q(d)^H, so only the blocks Q(d_r - d_s), r >= s), factors it by Cholesky,
 * and solves W a row at a time, interpolating that row of R_ts as it goes;
 * it then applies the row to every set.  Single precision, no workspace: the
 * card holds the inputs and the results, and each thread
 * N (N + 1) / 2 + N complex floats of local memory (about 1.5 KB at 18).
 */
#include <cuda_runtime.h>
#include <stdint.h>

#include "include/bartorch.h"

namespace {

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

/*
 * add(c, c2, Q_{c c2}(disp)) for the rows c0 <= c < c1; false when disp leaves
 * the table.  The caller zeroes what add accumulates into.
 */
template <class Add>
__device__ bool table_at(const params& p, const float* disp, int c0, int c1, Add add)
{
	const int64_t n = 2 * (int64_t)p.reach * p.upsample + 1;
	const int C = p.C, CC = C * C;
	int64_t base[3] = { 0, 0, 0 };
	float w[3][4] = { { 1, 0, 0, 0 }, { 1, 0, 0, 0 }, { 1, 0, 0, 0 } };

	for (int a = 0; a < p.ndim; a++) {
		const float u = (disp[a] + (float)p.reach) * (float)p.upsample;
		const float f = floorf(u);
		base[a] = (int64_t)f - 1;
		if (base[a] < 0 || base[a] + 3 >= n)
			return false;
		lagrange4(u - f, w[a]);
	}
	const int t1 = p.ndim > 1 ? 4 : 1, t2 = p.ndim > 2 ? 4 : 1;
	for (int i0 = 0; i0 < 4; i0++)
		for (int i1 = 0; i1 < t1; i1++)
			for (int i2 = 0; i2 < t2; i2++) {
				int64_t flat = base[0] + i0;
				float wt = w[0][i0];
				if (p.ndim > 1) {
					flat = flat * n + base[1] + i1;
					wt *= w[1][i1];
				}
				if (p.ndim > 2) {
					flat = flat * n + base[2] + i2;
					wt *= w[2][i2];
				}
				const cf* q = p.table + flat * CC;
				for (int c = c0; c < c1; c++)
					for (int c2 = 0; c2 < C; c2++)
						add(c, c2, wt * q[c * C + c2]);
			}
	return true;
}

/* Packed lower triangle: entry (i, k), k <= i. */
__device__ __forceinline__ int tri(int i, int k) { return i * (i + 1) / 2 + k; }

/*
 * Per thread: the packed lower triangle of R_ss + mu I (then its Cholesky
 * factor) and one kernel row, N (N + 1) / 2 + N complex floats of local
 * memory; R_ts is interpolated a row at a time as each row of W is solved.
 */
template <int N>
__global__ void __launch_bounds__(128) grog_kernel(params p)
{
	const int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (m >= p.targets)
		return;

	const int C = p.C, D = p.D, n = C * D, nd = p.ndim;
	cf A[N * (N + 1) / 2];
	cf z[N];
	float off[N * 3];

	const int64_t s = p.shot[m], j = p.sample[m];
	const int64_t* js = p.src + m * D;
	bool bad = s < 0 || s >= p.shots || j < 0 || j >= p.P;
	for (int i = 0; i < D && !bad; i++)
		bad = js[i] < 0 || js[i] >= p.P;
	if (bad) {
		atomicMax(p.status, 4);
		return;
	}
	const float* tj = p.traj + (s * p.P + j) * nd;
	for (int i = 0; i < D; i++)
		for (int a = 0; a < nd; a++)
			off[i * 3 + a] = p.traj[(s * p.P + js[i]) * nd + a] - nearbyintf(tj[a]);

	/* lower block (r, k), r >= k: Q(d_r - d_k) */
	for (int i = 0; i < n * (n + 1) / 2; i++)
		A[i] = { 0.f, 0.f };
	for (int r = 0; r < D; r++)
		for (int k = 0; k <= r; k++) {
			float dd[3];
			for (int a = 0; a < nd; a++)
				dd[a] = off[r * 3 + a] - off[k * 3 + a];
			const bool in = table_at(p, dd, 0, C, [&](int c, int c2, cf v) {
				const int i = r * C + c, l = k * C + c2;
				if (l <= i)
					A[tri(i, l)] = A[tri(i, l)] + v;
			});
			if (!in) {
				atomicMax(p.status, 6);
				return;
			}
		}

	float tr = 0;
	for (int i = 0; i < n; i++)
		tr += A[tri(i, i)].x;
	const float mu = p.lam * tr / (float)n;
	for (int i = 0; i < n; i++)
		A[tri(i, i)].x += mu;

	/* Cholesky in place */
	for (int c = 0; c < n; c++) {
		float d = A[tri(c, c)].x;
		for (int k = 0; k < c; k++)
			d -= A[tri(c, k)].x * A[tri(c, k)].x + A[tri(c, k)].y * A[tri(c, k)].y;
		if (!(d > 0)) {
			atomicMax(p.status, 5);
			return;
		}
		d = sqrtf(d);
		A[tri(c, c)] = { d, 0.f };
		const float inv = 1.f / d;
		for (int i = c + 1; i < n; i++) {
			cf acc = A[tri(i, c)];
			for (int k = 0; k < c; k++)
				acc = acc - mulc(A[tri(i, k)], A[tri(c, k)]);
			A[tri(i, c)] = inv * acc;
		}
	}

	/* W[c, :] = (A^-1 R_ts[c, :]^H)^H, R_ts[c, (k, c2)] = Q_{c c2}(-d_k); out_c = W[c, :] x per set */
	float gain = 0;
	for (int c = 0; c < C; c++) {
		for (int k = 0; k < n; k++)
			z[k] = { 0.f, 0.f };
		for (int k = 0; k < D; k++) {
			float dd[3];
			for (int a = 0; a < nd; a++)
				dd[a] = -off[k * 3 + a];
			const bool in = table_at(p, dd, c, c + 1, [&](int, int c2, cf v) { z[k * C + c2] = z[k * C + c2] + v; });
			if (!in) {
				atomicMax(p.status, 6);
				return;
			}
		}
		for (int k = 0; k < n; k++)
			z[k] = conj(z[k]);
		for (int i = 0; i < n; i++) {
			cf acc = z[i];
			for (int k = 0; k < i; k++)
				acc = acc - A[tri(i, k)] * z[k];
			z[i] = (1.f / A[tri(i, i)].x) * acc;
		}
		for (int i = n - 1; i >= 0; i--) {
			cf acc = z[i];
			for (int k = i + 1; k < n; k++)
				acc = acc - conj(A[tri(k, i)]) * z[k];
			z[i] = (1.f / A[tri(i, i)].x) * acc;
		}
		for (int k = 0; k < n; k++)
			gain += z[k].x * z[k].x + z[k].y * z[k].y;
		for (int e = 0; e < p.sets; e++) {
			const cf* x = p.data[e];
			cf acc = { 0.f, 0.f };
			for (int i = 0; i < D; i++)
				for (int c2 = 0; c2 < C; c2++)
					acc = acc + mulc(x[((int64_t)c2 * p.shots + s) * p.P + js[i]], z[i * C + c2]);
			p.out[e][(int64_t)c * p.targets + m] = acc;
		}
	}
	if (p.noise)
		p.noise[m] = gain / (float)C;
}

template <int N>
cudaError_t launch(const params& p, cudaStream_t stream)
{
	const int threads = 128;
	const int64_t blocks = (p.targets + threads - 1) / threads;
	grog_kernel<N><<<(unsigned)blocks, threads, 0, stream>>>(p);
	return cudaGetLastError();
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
	if (n > 48 || targets > 2147483647LL * 128)
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

	cudaStream_t st = (cudaStream_t)stream;
	if (n <= 12)
		return (int)launch<12>(p, st);
	if (n <= 18)
		return (int)launch<18>(p, st);
	if (n <= 24)
		return (int)launch<24>(p, st);
	if (n <= 32)
		return (int)launch<32>(p, st);
	return (int)launch<48>(p, st);
}

}  // extern "C"
