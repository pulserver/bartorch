# Regularized least squares

The mathematics behind {doc}`../../../explanation/solvers` and
{doc}`../../../explanation/regularizers`: the estimation problem, the
properties of the encoding that make it ill-posed, the proximal operators and
the convergence conditions of each iteration.  The code is
`src/bartorch/optim/_linear.py` (the solvers), `src/bartorch/optim/_blocks.py`
(one step of each), `src/bartorch/priors/` (the terms) and
`src/bartorch/apps/_pics.py` (the choice of iteration); the tests are
`tests/test_optim_iterators.py`, `tests/test_solve.py`, `tests/test_priors.py`
and `tests/test_apps.py`.

## Measurement model

The data $y \in \mathbb{C}^M$ and the image $x \in \mathbb{C}^N$ are related by
$y = Ax + \varepsilon$, with $A$ the encoding operator and $\varepsilon$ noise
that is complex Gaussian with independent, identically distributed entries
after prewhitening.  $A$ is applied without being formed as a matrix.  The
adjoint $A^H$ is defined by $\langle Ax, y\rangle = \langle x, A^H y\rangle$,
and the normal operator $A^HA$ is self-adjoint and positive semidefinite.  Every
solver needs only applications of $A$ and $A^H$, or of $A^HA$.  $A^Hy$ is the
simplest estimate; for Cartesian SENSE it is the coil combination of the
zero-filled inverse FFTs and retains the aliasing.

## Least squares and conditioning

The least-squares estimate solves the normal equations $A^HA\hat{x} = A^Hy$.
With the singular value decomposition $A = \sum_i \sigma_i u_i v_i^H$, the
minimizer of least norm is

$$
x^\dagger = \sum_{\sigma_i > 0} \frac{u_i^H y}{\sigma_i} \, v_i .
$$

| Property | Condition | Consequence |
| --- | --- | --- |
| Rank deficiency | some $x_0 \ne 0$ satisfy $Ax_0 = 0$ | $x^\dagger + x_0$ fits the data equally well; the data carry no information about $x_0$ |
| Ill-conditioning | full column rank, small $\sigma_{\min}$ | unique minimizer, with the noise along $v_i$ amplified by $1/\sigma_i$ |

An undersampled parallel-imaging acquisition measures fewer k-space positions
than the image has voxels, but each with several coils.  Where the coils
separate the voxels that alias onto one another, $A$ has full column rank and
the problem is ill-conditioned rather than underdetermined; the spatially
resolved noise amplification is the g-factor of SENSE.[^pruessmann1999]  Beyond
the acceleration the coil geometry supports, and outside the region the coils
cover, $A$ is rank deficient.  The eigenvalues of $A^HA$ are the $\sigma_i^2$, so
$\kappa(A^HA) = \kappa(A)^2$.

## Conjugate gradients and early stopping

{class}`~bartorch.optim.CG` on the normal equations, started from $x_0 = 0$,
keeps its iterates in the range of $A^H$ and so converges to $x^\dagger$; a
nonzero $x_0$ keeps its null-space component.  On noisy data the iteration
shows semi-convergence: early iterates recover the components with large
$\sigma_i$, later ones fit the noise along those with small $\sigma_i$.  The
iteration count acts as a regularization parameter,[^hansen] as in iterative
SENSE,[^pruessmann2001] with a best value that depends on the data and the
noise level.

## Regularization functionals

| Regularization | $R(x)$ | Effect |
| --- | --- | --- |
| Tikhonov | $\tfrac12 \lVert x \rVert_2^2$ | Solution $(A^HA + \lambda I)^{-1}A^Hy$: the component along $v_i$ is weighted by $\sigma_i^2/(\sigma_i^2+\lambda)$.  Not a spatial smoothness prior. |
| Generalized Tikhonov | $\tfrac12 \lVert Gx - b \rVert_2^2$ | Shrinkage toward a reference $b$, or, with $G$ a finite difference, a quadratic smoothness prior |
| Sparsity in a transform | $\lVert \Psi x \rVert_1$ | Few significant coefficients, for example of a wavelet transform[^lustig] |
| Total variation | $\sum_r \lVert (\nabla x)_r \rVert_2$ | Piecewise-constant images[^rof] |
| Locally low rank | $\sum_b \lVert B_b x \rVert_*$ | Image blocks, as matrices over a contrast or coefficient axis, of low rank |

The quadratic functionals keep the problem linear least squares, which
{class}`~bartorch.optim.CG` solves with {class}`~bartorch.optim.Tikhonov`
terms stacked beneath the encoding.  The others are not differentiable.

## Proximal operators

The data term $f(x) = \tfrac12\lVert Ax-y\rVert_2^2$ has gradient $A^H(Ax-y)$,
Lipschitz continuous with $L = \lambda_{\max}(A^HA)$.  A nonsmooth $g$ enters
through

$$
\operatorname{prox}_{\tau g}(v) = \arg\min_u \; \tfrac12 \lVert u - v \rVert_2^2 + \tau g(u),
$$

defined for proper, closed, convex $g$ and $\tau > 0$.  For the $\ell_1$ norm
of a complex array it is soft thresholding of the modulus with the phase kept,
zero where $v_i = 0$; thresholding a group jointly (the gradient components of
isotropic total variation, or a term's `joint_axes`) replaces $\lvert v_i\rvert$
by the group's $\ell_2$ norm.  For the indicator function of a convex set it is
the projection onto the set.

A term of {mod}`bartorch.priors` is $g(Gx)$.  For unitary $G$,
$\operatorname{prox}_{\tau g\circ G}(v) = G^H\operatorname{prox}_{\tau g}(Gv)$,
and BART's wavelet and locally low-rank terms apply their transform inside the
proximal operator and have $G = I$.  This is exact for the Haar wavelet on even
lengths, which is orthogonal, and for non-overlapping blocks.  BART's Daubechies
and CDF 4/4 transforms in symmetric extension produce more coefficients than
samples and are not orthogonal (`test_synthesis_is_not_the_adjoint_of_a_redundant_transform`
in `tests/test_wavelet_thresh.py`); with them, with overlapping blocks, and with
the random shifts both terms draw by default, the step approximates that
proximal operator.  For a general $G$ the solver has to apply $G$ separately.

## Iterations

| Algorithm | Problem and splitting | Transform $G$ | Per iteration | Convergence conditions |
| --- | --- | --- | --- | --- |
| CG | $\min \tfrac12\lVert Ax-y\rVert^2 + \tfrac{\lambda}{2}\lVert x\rVert^2 + \sum_i \tfrac{w_i}{2}\lVert G_ix-b_i\rVert^2$ | Any, inside the quadratic terms | One normal operator | Positive semidefinite normal operator |
| IST | $x \leftarrow \operatorname{prox}_{\tau g}(x - \tau\nabla f(x))$ | $I$ only | One normal operator, one proximal operator | Convex $f$, $g$; $0 < \tau \le 1/L$; objective error $O(1/k)$ |
| FISTA | IST with Nesterov momentum | $I$ only | As IST | As IST; objective error $O(1/k^2)$[^beck] |
| ADMM | $f(x) + \sum_j g_j(z_j)$ subject to $z_j = G_jx$ | Any | A CG solve with $A^HA + \rho\sum_j G_j^HG_j$, one proximal operator per term, a dual update | Convex terms, any $\rho > 0$[^boyd] |
| Primal-dual (Chambolle–Pock) | Saddle-point form with a dual variable per $g_j(G_jx)$ | Any | $A$, $G_j$, their adjoints and proximal operators; no inner solve | Convex terms, $\sigma\tau\lVert K\rVert^2 < 1$ for the stacked operator $K$[^chambolle] |
| POCS | Repeated projection onto convex sets | Inside the projections | One projection per set | Closed convex sets with nonempty intersection |

IST, FISTA and the primal-dual iteration take BART's default step of 0.95,
not divided by $L$ unless `eigen=True`.  ADMM's `maxiter` is BART's budget of
inner conjugate-gradient iterations over the whole run, not a count of outer
steps.  `apps.pics` chooses the iteration as BART's `italgo_choose` does: CG
without terms or with an $\ell_2$ term only, NIHT for the hard-thresholding
terms, ADMM when a total-variation-type or imaginary-part term appears, and
otherwise FISTA for a first term and ADMM once there is a second.

## References

[^pruessmann1999]: Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: sensitivity encoding for fast MRI. *Magn Reson Med* 42(5):952–962 (1999). [doi:10.1002/(SICI)1522-2594(199911)42:5\<952::AID-MRM16\>3.0.CO;2-S](https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S)

[^pruessmann2001]: Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in sensitivity encoding with arbitrary k-space trajectories. *Magn Reson Med* 46(4):638–651 (2001). [doi:10.1002/mrm.1241](https://doi.org/10.1002/mrm.1241)

[^hansen]: Hansen PC. *Rank-Deficient and Discrete Ill-Posed Problems: Numerical Aspects of Linear Inversion.* SIAM (1998). [doi:10.1137/1.9780898719697](https://doi.org/10.1137/1.9780898719697)

[^lustig]: Lustig M, Donoho D, Pauly JM. Sparse MRI: the application of compressed sensing for rapid MR imaging. *Magn Reson Med* 58(6):1182–1195 (2007). [doi:10.1002/mrm.21391](https://doi.org/10.1002/mrm.21391)

[^rof]: Rudin LI, Osher S, Fatemi E. Nonlinear total variation based noise removal algorithms. *Physica D* 60(1–4):259–268 (1992). [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)

[^beck]: Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM J Imaging Sci* 2(1):183–202 (2009). [doi:10.1137/080716542](https://doi.org/10.1137/080716542)

[^boyd]: Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. Distributed optimization and statistical learning via the alternating direction method of multipliers. *Found Trends Mach Learn* 3(1):1–122 (2011). [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)

[^chambolle]: Chambolle A, Pock T. A first-order primal-dual algorithm for convex problems with applications to imaging. *J Math Imaging Vis* 40(1):120–145 (2011). [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

Further reading: Parikh N, Boyd S. Proximal algorithms. *Found Trends Optim* 1(3):127–239 (2014). [doi:10.1561/2400000003](https://doi.org/10.1561/2400000003)
