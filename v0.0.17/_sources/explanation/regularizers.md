# Regularizers

```{admonition} TL;DR
:class: tldr

- A solver uses a regularizer only through its proximal operator; for an $\ell_1$ penalty that operator is soft thresholding.
- {mod}`bartorch.priors` builds each of BART's terms as $g(Gx)$; a term whose $G$ is not the identity, such as total variation, decides which solver you can use.
- An {class}`~bartorch.priors.ImplicitPrior` puts a denoiser in place of the proximal operator.
```

An undersampled or noisy acquisition does not determine the image on its own,
so you add a penalty that expresses what you know about it: few significant
wavelet coefficients, piecewise-constant intensity, low rank across a contrast
axis. You choose the term and its weight $\lambda$; the solver
({doc}`solvers`) does the rest. Lesson 5 of the
{doc}`Course <../gallery/course>` compares three terms and sweeps their
weights, and lesson 10 replaces the term with a denoising network.

## Capabilities

:::{container} capabilities

- **Builds every regularization term BART's `pics` accepts, from an object with axes rather than a `-R` string.**

  Code: {mod}`bartorch.priors`. Tests: *the terms are the ones barts parser knows*; *a term is the string the parser reads*; *an axis is an axis and not a bitmask*; *every term bart has can be asked for* (`test_solve.py`).
- **Applies a term's proximal operator on its own, with the bits BART's solvers get from it.**

  Code: {meth}`~bartorch.priors.Regularizer.prox`. Tests: *a term applied here is the one bart iterates with*; *the l1 term is a soft threshold at gamma times its weight* (`test_priors.py`).
- **Soft-thresholds a tensor element-wise or jointly over axes, and hard-thresholds it.**

  Code: {func}`~bartorch.soft_thresh`, {func}`~bartorch.hard_thresh`. Tests: *soft thresh is its closed form*; *joint soft thresh shrinks by the norm over the joint axes*; *hard thresh keeps what exceeds the threshold* (`test_wavelet_thresh.py`).
- **Reports each term's transform $G$ and the shape its proximal operator works on.**

  Code: {meth}`~bartorch.priors.Regularizer.transform_is_identity`, {meth}`~bartorch.priors.Regularizer.prox_shape`. Tests: *a wavelet term carries its transform inside the proximal operator*; *total variation thresholds the components of a gradient*; *the laplace term carries a real transform in front of it* (`test_priors.py`).
- **Builds a term once per image shape and hands the same operator to every solve.**

  Code: {meth}`~bartorch.priors.Regularizer.build`. Tests: *a term builds its operator once per shape* (`test_priors.py`); *a term is built once and handed over as it stands* (`test_solve.py`).
- **Cycle-spins the wavelet term by a random shift before every application, as BART's `pics` does, unless `randshift=False`.**

  Code: {class}`~bartorch.priors.Wavelet`. Tests: *cycle spinning makes the wavelet threshold a random one* (`test_priors.py`); *cycle spinning is on as it is for the tool* (`test_solve.py`).
- **Refuses a tensor that requires a gradient in BART's proximal operators, which have no backward pass.**

  Code: {meth}`~bartorch.priors.Regularizer.prox`. Tests: *the proximal operator refuses a gradient rather than dropping one* (`test_priors.py`).
- **Applies a denoiser in place of a term, on the image or on the codomain of a transform you give.**

  Code: {class}`~bartorch.priors.ImplicitPrior`. Tests: *a denoiser stands where a term would* (`test_optim_functional.py`); *the alternating direction step splits where the denoiser acts* (`test_priors.py`).

:::

## Division of responsibility

- **BART**: each term's proximal operator and transform, built by
  `opt_reg_configure`, including the wavelet, thresholding and low-rank code.
- **bartorch**: the term objects and their axes, the proximal operator as a
  function, and the conversion between a complex image and the real planes a
  network takes.
- **Your network, or deepinv's and MONAI's**: the denoiser behind an
  {class}`~bartorch.priors.ImplicitPrior`.

## Method

The regularized estimate is

$$
\hat{x} = \arg\min_x \; \tfrac12 \lVert Ax - y \rVert_2^2 + \lambda\, g(Gx),
$$

and a proximal solver alternates a step on the data term with the proximal
operator of the penalty,

$$
\operatorname{prox}_{\tau\lambda g}(v) = \arg\min_u \; \tfrac12 \lVert u - v \rVert_2^2 + \tau\lambda\, g(u),
$$

the point closest to $v$ that also keeps $g$ small. For $g = \lVert\cdot\rVert_1$
it is soft thresholding,
$\operatorname{prox}(v)_i = \frac{v_i}{\lvert v_i\rvert}\max(\lvert v_i\rvert - \tau\lambda, 0)$:
entries with modulus below $\tau\lambda$ are set to zero, the others are shrunk
by $\tau\lambda$, and the phase is kept. Applied to wavelet coefficients, it
keeps the few large ones that carry the edges and removes the many small ones
that carry the noise.

```{image} ../_static/figures/reg-proximal-operator.svg
:class: plot
:alt: Top left, soft thresholding as a function of its input for thresholds 0.5, 1 and 2, zero within the threshold and parallel to the identity outside it. Top right, the Haar coefficients of a noisy piecewise-constant signal, with the 33 that survive a threshold of 0.3 marked. Bottom, the noisy signal and the output of the wavelet term's proximal operator, which follows the noiseless steps.
:width: 100%
```

The figure is computed with {func}`~bartorch.soft_thresh`,
{func}`~bartorch.fwt` and a {class}`~bartorch.priors.Wavelet` term with the
Haar wavelet and no random shift. `term.prox(x, gamma)` thresholds at `gamma`
times the term's weight: the weight is part of the term, and `gamma` is the
solver's step.

### Proximal operators of the terms

For $\tfrac{\lambda}{2}\lVert x\rVert_2^2$ ({class}`~bartorch.priors.L2`) the
proximal operator is the scaling $v/(1+\tau\lambda)$. Total variation
thresholds the finite differences jointly, the components of each voxel's
gradient by their Euclidean norm (isotropic TV). The locally low-rank term
soft-thresholds the singular values of image blocks, and
{class}`~bartorch.priors.NonNegative` projects onto the set it names.
`joint_axes` makes any thresholding term shrink a group of entries by its norm,
for example across coils or contrasts. The wavelet term transforms, thresholds
and transforms back.

### Terms on a transformed variable

When $G$ is unitary, $\operatorname{prox}_{g\circ G}(v) = G^H \operatorname{prox}_g(Gv)$.
The wavelet and locally low-rank terms use this form: they apply their
transform inside the proximal operator and report $G = I$. The step is exact
for the Haar wavelet on even lengths and non-overlapping blocks without random
shifts, and an approximation with the redundant Daubechies and CDF wavelets,
overlapping blocks or cycle spinning. For a general $G$, such as the
finite differences of total variation, the Fourier transform of
{class}`~bartorch.priors.FourierL1` or the Laplacian of
{class}`~bartorch.priors.Laplace`, $g\circ G$ has no closed-form proximal
operator: the solver has to apply $G$ itself, which IST and FISTA do not and
ADMM and the primal-dual method do. Total generalized variation and the two
infimal convolutions also add auxiliary variables to $x$ and go to the same
two solvers. The table in {doc}`../api/priors` lists $G$ for every term, and
{doc}`solvers` turns it into the choice of solver.

### Implicit priors

Plug-and-play reconstruction replaces $\operatorname{prox}_{\tau\lambda g}$ with
a denoiser $D_\sigma$, called as `denoiser(x, sigma)` with no explicit $g$.
{class}`~bartorch.priors.ImplicitPrior` is accepted wherever a proximal term is; with
`spatial=` it hands the network real planes of order unity and converts the
result back, and with `transform=` it is applied to $Gx$, for a network trained
on contrast-weighted images while the solver works on subspace coefficients.
Unlike BART's proximal operators it is differentiable, so a solve containing
it can be trained end to end ({doc}`learned-reconstruction`).

## Examples

- {doc}`../auto_examples/03-regularization/01-regularized-reconstruction`:
  Tikhonov, wavelet and total-variation terms on one acquisition, and their
  weights.
- {doc}`../auto_examples/06-learning/01-plug-and-play`: a pretrained denoiser
  in place of the term.
- {doc}`../api/priors`: every term, its functional, its transform and the
  solvers that take it.
