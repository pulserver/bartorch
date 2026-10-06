/*
 * The pass along x of the pair kernels for one rank -- BARTORCH_PAIRED_RANK
 * coefficients -- at every grid size the library is compiled for.  Each rank
 * is a translation unit of its own (CMakeLists.txt), so the ranks compile in
 * parallel.
 */
#include "ops/paired.cuh"

#ifndef BARTORCH_PAIRED_RANK
#error "BARTORCH_PAIRED_RANK names the number of coefficients this unit is compiled for"
#endif

using namespace paired;

namespace {

constexpr unsigned R = BARTORCH_PAIRED_RANK;

__device__ inline float widen(float v) { return v; }
__device__ inline float widen(__nv_bfloat16 v) { return __bfloat162float(v); }

/* A block transforms `lines` lines of R coefficients each: one line at four
 * coefficients or more, and below that as many as make four transforms. */
template <unsigned N>
struct Shape {

	static constexpr unsigned lines = (R < 4) ? 4 / R : 1;

	using XF = FFT1<N, cufftdx::fft_direction::forward, lines * R>;
	using XI = FFT1<N, cufftdx::fft_direction::inverse, lines * R>;

	static constexpr size_t smem = cmax(sizeof(cplx) * lines * R * N, cmax(XF::shared_memory_size, XI::shared_memory_size));

	static_assert(0 == ((size_t)N * N) % lines, "a block's lines tile the grid");
};

/* The pass along x of both sets of the pair: per line of R coefficients, each
 * set's phase, the transform, the multiplication by the set's function at the
 * places the samples reach, the transform back, the conjugate phase, and the
 * sum of the two.  Transform f of the block is coefficient f % R of the
 * block's line f / R. */
template <unsigned N, typename P>
__global__ void __launch_bounds__(Shape<N>::XF::max_threads_per_block) fused(XArgs a)
{
	using S = Shape<N>;
	using XF = typename S::XF;
	using XI = typename S::XI;

	extern __shared__ __align__(16) unsigned char smem[];
	cplx* exch = reinterpret_cast<cplx*>(smem);
	__shared__ int any;

	const unsigned f = threadIdx.y;
	const unsigned tid = threadIdx.x + threadIdx.y * XF::block_dim.x;
	const unsigned nthreads = XF::block_dim.x * XF::block_dim.y;
	const size_t first = (size_t)blockIdx.x * S::lines * N;
	const size_t g0 = first + (size_t)(f / R) * N;

	cplx* B = a.B + (size_t)(f % R) * N * N * N;

	/* Lines that reach no kept place contribute nothing. */
	if (NULL != a.mask) {

		if (0 == tid)
			any = 0;

		__syncthreads();

		for (size_t w = (first >> 5) + tid; w <= ((first + S::lines * N - 1) >> 5); w += nthreads)
			if (0 != a.mask[w])
				any = 1;

		__syncthreads();

		if (0 == any) {

			for (unsigned i = 0; i < XF::elements_per_thread; i++) {

				unsigned e = threadIdx.x + i * XF::stride;

				if (e < N) {

					cplx z;
					z.x = 0.f;
					z.y = 0.f;
					B[g0 + e] = z;
				}
			}

			return;
		}
	}

	cplx in[XF::storage_size];
	cplx w[XF::storage_size];
	cplx acc[XF::storage_size];

#pragma unroll
	for (unsigned i = 0; i < XF::elements_per_thread; i++) {

		unsigned e = threadIdx.x + i * XF::stride;
		cplx z;
		z.x = 0.f;
		z.y = 0.f;
		in[i] = (e < N) ? B[g0 + e] : z;
		acc[i] = z;
	}

	for (int sx = 0; sx < 2; sx++) {

#pragma unroll
		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;
			w[i] = (e < N) ? cmul(in[i], a.tx[sx * N + e]) : in[i];
		}

		__syncthreads();
		XF().execute(w, smem);
		__syncthreads();

#pragma unroll
		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;

			if (e < N)
				exch[f * N + e] = w[i];
		}

		__syncthreads();

		const P* psf = static_cast<const P*>(a.psf[sx]);

		for (unsigned k = tid; k < S::lines * N; k += nthreads) {

			const unsigned l = k / N;
			const unsigned e = k % N;
			const size_t g = first + k;
			cplx v[R];
			cplx o[R];

#pragma unroll
			for (unsigned c = 0; c < R; c++) {

				v[c] = exch[(l * R + c) * N + e];
				o[c].x = 0.f;
				o[c].y = 0.f;
			}

			bool kept = true;
			bart_dim_t j = (bart_dim_t)g;

			if (NULL != a.mask) {

				const unsigned word = a.mask[g >> 5];
				const unsigned bit = 1u << (g & 31);

				kept = (0 != (word & bit));
				j = a.prefix[g >> 5] + __popc(word & (bit - 1));
			}

			/* Each entry of the upper triangle is read once and serves both
			 * places it stands for; every o[r] still sums over c in order. */
			if (kept) {
#pragma unroll
				for (unsigned hi = 0; hi < R; hi++)
#pragma unroll
					for (unsigned lo = 0; lo <= hi; lo++) {

						const float m = widen(psf[(bart_dim_t)(lo + hi * (hi + 1) / 2) * a.L + j]);

						o[lo].x += m * v[hi].x;
						o[lo].y += m * v[hi].y;

						if (lo != hi) {

							o[hi].x += m * v[lo].x;
							o[hi].y += m * v[lo].y;
						}
					}
			}

#pragma unroll
			for (unsigned r = 0; r < R; r++)
				exch[(l * R + r) * N + e] = o[r];
		}

		__syncthreads();

#pragma unroll
		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;

			if (e < N)
				w[i] = exch[f * N + e];
		}

		__syncthreads();
		XI().execute(w, smem);

#pragma unroll
		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;

			if (e < N) {

				cplx t = cmulc(w[i], a.tx[sx * N + e]);
				acc[i].x += t.x;
				acc[i].y += t.y;
			}
		}
	}

#pragma unroll
	for (unsigned i = 0; i < XF::elements_per_thread; i++) {

		unsigned e = threadIdx.x + i * XF::stride;

		if (e < N)
			B[g0 + e] = acc[i];
	}
}

template <unsigned N>
int prepare(void)
{
	using S = Shape<N>;

	int device = 0;
	int optin = 0;

	if ((cudaSuccess != cudaGetDevice(&device))
	    || (cudaSuccess != cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device))
	    || ((size_t)optin < S::smem)
	    || (cudaSuccess != cudaFuncSetAttribute(fused<N, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S::smem))
	    || (cudaSuccess != cudaFuncSetAttribute(fused<N, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S::smem))) {

		cudaGetLastError();
		return -1;
	}

	return 0;
}

template <unsigned N>
void run(const XArgs& a, int bf16, cudaStream_t stream)
{
	using S = Shape<N>;

	const unsigned blocks = (unsigned)((size_t)N * N / S::lines);

	if (bf16)
		fused<N, __nv_bfloat16><<<blocks, S::XF::block_dim, S::smem, stream>>>(a);
	else
		fused<N, float><<<blocks, S::XF::block_dim, S::smem, stream>>>(a);
}

#define BARTORCH_PAIRED_FUSED(n) { n, { prepare<n>, run<n> } },

const struct { unsigned n; Fused fused; } table[] = { BARTORCH_PAIRED_SIZES(BARTORCH_PAIRED_FUSED) };

} // namespace

#define BARTORCH_PAIRED_RANK_NAME(r) BARTORCH_PAIRED_RANK_NAME2(r)
#define BARTORCH_PAIRED_RANK_NAME2(r) bartorch_paired_rank_##r

const Fused* BARTORCH_PAIRED_RANK_NAME(BARTORCH_PAIRED_RANK)(unsigned n)
{
	for (const auto& e : table)
		if (e.n == n)
			return &e.fused;

	return NULL;
}
