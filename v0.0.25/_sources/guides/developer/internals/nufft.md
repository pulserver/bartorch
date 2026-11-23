# NUFFT substitution

The details behind {doc}`../../../explanation/non-cartesian`: the defaults and
how each is set, the estimates built from the adjoint, the storage of the
point spread function, and which backend computes what.  The code is
`src/csrc/substitute/finufft.c` (FINUFFT and cuFINUFFT plans),
`src/csrc/substitute/nufft_finufft.c` (BART's NUFFT operator built from a pair
of plans), `src/csrc/substitute/psf.c` (the point spread function) and
`src/bartorch/_finufft.py`; the tests are `tests/test_finufft.py` and
`tests/test_density.py`.

## Tolerance and oversampling

| Setting | Default | Set by |
| --- | --- | --- |
| Tolerance $\varepsilon$ | $10^{-3}$; $10^{-6}$ for `ncalib` and `nlinv` (`_CALIBRATES` in `_dispatch.py`) | `width` of {class}`~bartorch.linop.NUFFT`, BART's `-w`, converted to the tolerance that gives that width |
| Oversampling $\sigma$ | 1.25; 2 for `ncalib` and `nlinv` | `oversampling` of {class}`~bartorch.linop.NUFFT`, BART's `-o` wherever it is not BART's default of 2 |

FINUFFT has no width parameter.  It sizes its kernel from the tolerance,

$$
n_s = \left\lceil \frac{\ln(t_f / \varepsilon)}{\pi \sqrt{1 - 1/\sigma}} + 1 \right\rceil,
\qquad t_f = 0.18 \cdot 1.4^{\,d-1},
$$

for a type-1 or type-2 transform in $d$ dimensions, so a width is carried across
by inverting that relation, no tighter than single precision reaches.  A
tolerance below what single precision reaches at a size is planned at the
tolerance that can be reached.  cuFINUFFT accepts $\sigma = 2$ and
$\sigma = 1.25$; any other oversampling plans on the host and fails on a card.

## Adjoint, density compensation and reconstruction

| Estimate | Definition | Property |
| --- | --- | --- |
| Adjoint | $\mathrm{NUFFT}^H y$ | Each sample is added onto the grid; for radial sampling the image is weighted in k-space by the sampling density, approximately $1/\lvert k \rvert$, and blurred |
| Density-compensated adjoint (gridding reconstruction) | $\mathrm{NUFFT}^H D y$, $d_j \approx 1/\rho(k_j)$ | Approximately equalizes the sampling density; unsampled k-space stays missing, and undersampling appears as streaks |
| Reconstruction | Solution of the least-squares or regularized problem with $A$ | Consistent with the samples; needs no density compensation, which may weight the data term instead |

{func}`~bartorch.estimate_density` iterates $w \leftarrow w / (C * w)$ from
$w = 1$, with $C$ the squared Dirichlet kernel of the grid applied through the
non-uniform transforms.  Weights given to an operator are applied on the
forward pass and conjugated on the adjoint, so $w_j = \sqrt{d_j}$ gives
$W^H W = D$.  Weights whose shape does not broadcast onto the samples of one
block are refused.

## Point spread function storage

`nufft.c` is compiled with `nufft_create`, `nufft_create2`, `nufft_get_psf*`,
`nufft_update_*` and `compute_psf*` renamed, so the original names resolve to
the substitution.  The normal operator sets BART's `conf.nopsf`, computes the
function with `compute_psf2` (whose transform is FINUFFT's), and stores it as
BART's operator expects.  Every storage form BART offers is served without a
gridding:

| BART `--nufft-conf` | Storage of the transfer function |
| --- | --- |
| default, `lowmem`, `no-precomp` | The complete complex array |
| `decomposed-psf` | $2^d$ grids of the image size, one at a time |
| `upper-triag-psf` | Half of the Hermitian kernel matrix of a subspace basis |
| `real-psf` | The real part |
| `compress-psf` | The entries within the trajectory's footprint, and their indices |

The mask of a compressed function is found by spreading the trajectory with
FINUFFT's own kernel (`spreadinterponly`), one set of frequencies at a time, at
the tolerance and oversampling the operator was planned with; the width it
spreads with is measured by spreading one sample, not computed from the
formula above.  The Toeplitz normal applies no roll-off: the deapodization is
inside the function FINUFFT computed.

## Backends

| Path | Backend | Device |
| --- | --- | --- |
| Forward and adjoint transform, point spread function | FINUFFT, compiled into every build | CPU |
| Forward and adjoint transform, point spread function | cuFINUFFT, compiled into the CUDA build | CUDA |
| Multiplication by the transfer function | BART, with MKL or pocketfft on the host and cuFFT on a card | Either |
| A configuration the substitution cannot serve | None: {class}`~bartorch.BartError` naming the reason, for example an image that varies along an axis the trajectory indexes in a BART command | Either |

Which backend serves a transform is decided by where its arguments are, not by
where the trajectory is: BART hands one operator memory on either side, so an
operator holds a pair of plans per side and builds each the first time a
transform is asked for there.  `_finufft.operators_built()` and
`_finufft.normals_built()` count which of FINUFFT and BART built each operator
and normal.  BART's gridder is reachable only through
`_finufft.barts_own_gridder()`, which the tests use for comparison.
