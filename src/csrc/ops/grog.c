/*
 * iGROG gridding with closed-form kernels on the host, for bartorch.tools.igrog.
 * Per target, in single precision:
 *
 *   Q at the D(D-1)/2 source-pair displacements and the D source offsets
 *   (cubic Lagrange in the table; Q(-d) = Q(d)^H gives the other half),
 *   R_ss + mu I by Cholesky, W = R_ts (R_ss + mu I)^-1, target = W s,
 *   and the target's noise gain sum |W|^2 / C.
 *
 * Every target is independent, so they are spread over OpenMP threads.
 * Ported from fast-encoding's grog.cpp.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "include/bartorch.h"

/* Largest sources x coils the per-target scratch holds. */
#define GROG_MAX_N 256

static void lagrange4(float t, float w[4])
{
	w[0] = -t * (t - 1) * (t - 2) / 6;
	w[1] = (t + 1) * (t - 1) * (t - 2) / 2;
	w[2] = -(t + 1) * t * (t - 2) / 2;
	w[3] = (t + 1) * t * (t - 1) / 6;
}

/* out (C, C) = Q(disp); 0, or -1 when disp leaves the table. */
static int table_at(const complex float* table, int ndim, int64_t C, int64_t reach, int64_t upsample,
		const float* disp, complex float* out)
{
	const int64_t n = 2 * reach * upsample + 1;
	int64_t base[3] = { 0, 0, 0 };
	float w[3][4] = { { 1, 0, 0, 0 }, { 1, 0, 0, 0 }, { 1, 0, 0, 0 } };

	for (int a = 0; a < ndim; a++) {
		const float u = (disp[a] + (float)reach) * (float)upsample;
		const float f = floorf(u);
		base[a] = (int64_t)f - 1;
		if (base[a] < 0 || base[a] + 3 >= n)
			return -1;
		lagrange4(u - f, w[a]);
	}

	const int64_t CC = C * C;
	for (int64_t k = 0; k < CC; k++)
		out[k] = 0.;

	const int t1 = ndim > 1 ? 4 : 1;
	const int t2 = ndim > 2 ? 4 : 1;
	for (int i0 = 0; i0 < 4; i0++)
		for (int i1 = 0; i1 < t1; i1++)
			for (int i2 = 0; i2 < t2; i2++) {
				int64_t flat = base[0] + i0;
				float wt = w[0][i0];
				if (ndim > 1) {
					flat = flat * n + base[1] + i1;
					wt *= w[1][i1];
				}
				if (ndim > 2) {
					flat = flat * n + base[2] + i2;
					wt *= w[2][i2];
				}
				const complex float* q = table + flat * CC;
				for (int64_t k = 0; k < CC; k++)
					out[k] += wt * q[k];
			}
	return 0;
}

/* In-place Cholesky of a Hermitian positive definite (n, n), lower triangle; 0 or -1. */
static int cholesky(complex float* A, int64_t n)
{
	for (int64_t j = 0; j < n; j++) {
		float d = crealf(A[j * n + j]);
		for (int64_t k = 0; k < j; k++)
			d -= crealf(A[j * n + k] * conjf(A[j * n + k]));
		if (!(d > 0))
			return -1;
		d = sqrtf(d);
		A[j * n + j] = d;
		for (int64_t i = j + 1; i < n; i++) {
			complex float s = A[i * n + j];
			for (int64_t k = 0; k < j; k++)
				s -= A[i * n + k] * conjf(A[j * n + k]);
			A[i * n + j] = s / d;
		}
	}
	return 0;
}

static void cholesky_solve(const complex float* L, int64_t n, complex float* b)
{
	for (int64_t i = 0; i < n; i++) {
		complex float s = b[i];
		for (int64_t k = 0; k < i; k++)
			s -= L[i * n + k] * b[k];
		b[i] = s / crealf(L[i * n + i]);
	}
	for (int64_t i = n - 1; i >= 0; i--) {
		complex float s = b[i];
		for (int64_t k = i + 1; k < n; k++)
			s -= conjf(L[k * n + i]) * b[k];
		b[i] = s / crealf(L[i * n + i]);
	}
}

struct grog_scratch {
	complex float* A;   /* (n, n) R_ss + mu I, then its Cholesky factor */
	complex float* z;   /* (n) one kernel row */
	complex float* q;   /* (C, C) one interpolated Q */
	complex float* rts; /* (C, n) R_ts */
	complex float* acc; /* (sets) */
	float off[GROG_MAX_N * 3];
};

struct grog_table {
	const complex float* q;
	int ndim;
	int64_t C, D, reach, upsample;
};

/* R_ss + mu I factored in s->A and R_ts in s->rts for the sources at s->off; 0 or an error code. */
static int grog_system(const struct grog_table* t, float lam, struct grog_scratch* s)
{
	const int64_t C = t->C, D = t->D, n = D * C;

	/* R_ss: block (i, k) = Q(d_i - d_k), (k, i) its adjoint */
	for (int64_t i = 0; i < D; i++)
		for (int64_t k = i; k < D; k++) {
			float dd[3];
			for (int a = 0; a < t->ndim; a++)
				dd[a] = s->off[i * 3 + a] - s->off[k * 3 + a];
			if (table_at(t->q, t->ndim, C, t->reach, t->upsample, dd, s->q))
				return 6;
			for (int64_t c = 0; c < C; c++)
				for (int64_t c2 = 0; c2 < C; c2++) {
					s->A[(i * C + c) * n + k * C + c2] = s->q[c * C + c2];
					s->A[(k * C + c2) * n + i * C + c] = conjf(s->q[c * C + c2]);
				}
		}
	float tr = 0;
	for (int64_t i = 0; i < n; i++)
		tr += crealf(s->A[i * n + i]);
	const float mu = lam * tr / (float)n;
	for (int64_t i = 0; i < n; i++)
		s->A[i * n + i] += mu;
	if (cholesky(s->A, n))
		return 5;

	/* R_ts[c, (i, c2)] = Q_{c c2}(-d_i) */
	for (int64_t i = 0; i < D; i++) {
		float dd[3];
		for (int a = 0; a < t->ndim; a++)
			dd[a] = -s->off[i * 3 + a];
		if (table_at(t->q, t->ndim, C, t->reach, t->upsample, dd, s->q))
			return 6;
		for (int64_t c = 0; c < C; c++)
			for (int64_t c2 = 0; c2 < C; c2++)
				s->rts[c * n + i * C + c2] = s->q[c * C + c2];
	}
	return 0;
}

/*
 * W[c, :] = (A^-1 R_ts[c, :]^H)^H; out_c = W[c, :] x for each set's sources x
 * at (shot, js) in data (coils, shots, P); returns sum |W|^2.
 */
static float grog_apply(const struct grog_table* t, struct grog_scratch* s, int64_t sets,
		const void* const* data, int64_t shots, int64_t P, int64_t shot, const int64_t* js,
		void* const* out, int64_t targets, int64_t m)
{
	const int64_t C = t->C, D = t->D, n = D * C;
	float gain = 0;

	for (int64_t c = 0; c < C; c++) {
		for (int64_t k = 0; k < n; k++)
			s->z[k] = conjf(s->rts[c * n + k]);
		cholesky_solve(s->A, n, s->z);
		for (int64_t e = 0; e < sets; e++)
			s->acc[e] = 0;
		for (int64_t i = 0; i < D; i++)
			for (int64_t c2 = 0; c2 < C; c2++) {
				const complex float w = conjf(s->z[i * C + c2]);
				const int64_t at = (c2 * shots + shot) * P + js[i];
				for (int64_t e = 0; e < sets; e++)
					s->acc[e] += w * ((const complex float*)data[e])[at];
				gain += crealf(w * conjf(w));
			}
		for (int64_t e = 0; e < sets; e++)
			((complex float*)out[e])[c * targets + m] = s->acc[e];
	}
	return gain;
}

int bartorch_grog_grid(int ndim, int64_t coils, int64_t sources, int64_t reach, int64_t upsample, float lam,
		const void* table, int64_t shots, int64_t samples, const float* traj, int64_t sets,
		const void* const* data, int64_t targets, const int64_t* shot, const int64_t* sample,
		const int64_t* src, void* const* out, float* noise)
{
	const int64_t C = coils, D = sources, P = samples, n = D * C;

	if (!table || !traj || !data || !shot || !sample || !src || !out || sets < 1)
		return 1;
	if ((ndim != 2 && ndim != 3) || C <= 0 || D <= 0 || D > P || n > GROG_MAX_N)
		return 2;

	const struct grog_table t = { table, ndim, C, D, reach, upsample };
	int bad = 0;

#pragma omp parallel
	{
		struct grog_scratch s;
		s.A = malloc(sizeof(complex float) * (size_t)(n * n));
		s.z = malloc(sizeof(complex float) * (size_t)n);
		s.q = malloc(sizeof(complex float) * (size_t)(C * C));
		s.rts = malloc(sizeof(complex float) * (size_t)(C * n));
		s.acc = malloc(sizeof(complex float) * (size_t)sets);

		if (!s.A || !s.z || !s.q || !s.rts || !s.acc) {
#pragma omp atomic write
			bad = 3;
		}

#pragma omp for schedule(dynamic, 256)
		for (int64_t m = 0; m < targets; m++) {
			int err;
#pragma omp atomic read
			err = bad;
			if (err)
				continue;

			const int64_t sh = shot[m], j = sample[m];
			const int64_t* js = src + m * D;
			err = (sh < 0 || sh >= shots || j < 0 || j >= P) ? 4 : 0;
			for (int64_t i = 0; i < D && !err; i++)
				err = (js[i] < 0 || js[i] >= P) ? 4 : 0;

			if (!err) {
				const float* tj = traj + (sh * P + j) * ndim;
				for (int64_t i = 0; i < D; i++)
					for (int a = 0; a < ndim; a++)
						s.off[i * 3 + a] = traj[(sh * P + js[i]) * ndim + a] - nearbyintf(tj[a]);
				err = grog_system(&t, lam, &s);
			}
			if (err) {
#pragma omp atomic write
				bad = err;
				continue;
			}
			const float gain = grog_apply(&t, &s, sets, data, shots, P, sh, js, out, targets, m);
			if (noise)
				noise[m] = gain / (float)C;
		}
		free(s.A);
		free(s.z);
		free(s.q);
		free(s.rts);
		free(s.acc);
	}
	return bad;
}
