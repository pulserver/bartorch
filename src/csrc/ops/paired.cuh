/*
 * What the pair kernels' translation units share: the transform, the complex
 * arithmetic, and the pass along x that each rank's unit compiles for every
 * grid size (paired_rank.cu).
 */
#pragma once

#include <stddef.h>

#include <cuda_runtime_api.h>

#include <cuda_bf16.h>

#include <cufftdx.hpp>

#include "misc/dimtypes.h"

#include "paired_sizes.h"

#ifndef BARTORCH_PAIRED_SM
#define BARTORCH_PAIRED_SM 890
#endif

namespace paired {

template <unsigned N, cufftdx::fft_direction D, unsigned F>
using FFT1 = decltype(cufftdx::Size<N>() + cufftdx::Precision<float>() + cufftdx::Type<cufftdx::fft_type::c2c>()
		+ cufftdx::Direction<D>() + cufftdx::FFTsPerBlock<F>() + cufftdx::SM<BARTORCH_PAIRED_SM>() + cufftdx::Block());

using cplx = typename FFT1<64, cufftdx::fft_direction::forward, 1>::value_type;

__device__ inline cplx cmul(cplx a, cplx b) { cplx c; c.x = a.x * b.x - a.y * b.y; c.y = a.x * b.y + a.y * b.x; return c; }
__device__ inline cplx cmulc(cplx a, cplx b) { cplx c; c.x = a.x * b.x + a.y * b.y; c.y = a.y * b.x - a.x * b.y; return c; }	/* a conj(b) */

constexpr size_t cmax(size_t a, size_t b) { return (a > b) ? a : b; }

/* What the pass along x reads. */
struct XArgs {

	cplx* B;			/* the coefficients, a volume apart */
	const void* psf[2];		/* floats or bfloat16, as the instantiation says */
	const unsigned* mask;		/* NULL where the function is kept whole */
	const int* prefix;
	const cplx* tx;			/* 2 x N: each set's phase along x */
	bart_dim_t L;			/* the function's places: one entry is L apart from the next */
};

/* The pass along x for one rank, at one grid size. */
struct Fused {

	int (*prepare)(void);
	void (*run)(const XArgs& a, int bf16, cudaStream_t stream);
};

} // namespace paired

/* Each rank's unit answers for the sizes it was compiled for: NULL for one it
 * was not. */
#define BARTORCH_PAIRED_DECLARE_RANK(r) const paired::Fused* bartorch_paired_rank_##r(unsigned n);
BARTORCH_PAIRED_RANKS(BARTORCH_PAIRED_DECLARE_RANK)
#undef BARTORCH_PAIRED_DECLARE_RANK
