# Learned reconstruction

```{admonition} TL;DR
:class: tldr

- A network enters a reconstruction as the proximal step of an iteration: trained on its own and applied in a fixed iteration (plug-and-play), or trained through a fixed number of iterations (unrolled), or through the fixed point (deep equilibrium).
- Complex multi-channel images are real channels to a network; the images stay on the host and a network runs on a device patch by patch, under mixed precision.
- An unrolled network is trained in stages: the denoiser alone, then one iteration at a time, then end to end with checkpointing.
- Without references, the acquired samples are split into a set to reconstruct from and a set to score on.
- A spread from randomized reconstructions becomes an error bar once it is calibrated on references to a coverage.
```

A reconstruction solves
$\min_x \tfrac12 \|A x - y\|_2^2 + g(x)$ for an image $x$ from data $y$
acquired through an encoding $A$ ({doc}`inverse-problems`).  A proximal
iteration touches the regularizer $g$ only through its proximal operator, a
map from an image to a less noisy one, and a learned reconstruction replaces
that map with a network.  The encoding, the data consistency step and the
iteration remain BART's.  This page states the ways the network is placed and
trained, how a network is applied to data larger than GPU memory, and what an
uncertainty estimate from it does and does not state.

## Where the network enters

| Scheme | What is trained | On what | Iterations at inference |
| --- | --- | --- | --- |
| Plug-and-play | A denoiser, once | Pairs of a clean and a noisy image | As many as convergence needs |
| Unrolled | The denoiser and the step's parameters | Data and references, through the iterations | The fixed number trained with |
| Deep equilibrium | As unrolled | As unrolled, differentiated at the fixed point | Until the fixed point |

A plug-and-play denoiser is independent of the acquisition, so one network
serves every protocol whose images resemble its training images, whatever the
acceleration, sampling pattern or coil array; it is not adapted to the
aliasing and g-factor noise a given undersampling produces.  An unrolled
network is trained on exactly those artefacts and needs few iterations, which
is what bounds the reconstruction time on the scanner, and is tied to the
encoding it was trained with.
{class}`~bartorch.priors.ImplicitPrior` places a network in the proximal
step of any block of {mod}`bartorch.optim`, {class}`~bartorch.learning.Unrolled`
repeats a block a fixed number of times, and
{class}`~bartorch.optim.FixedPoint` iterates it to its fixed point
({doc}`differentiation`).

A step may depend on the iteration.  In plug-and-play, the denoiser's noise
level $\sigma_k$ is decreased over the iterations, so that early iterations
remove the undersampling artefacts and later ones retain detail;[^dpir] the ADMM penalty follows it as
$\rho_k = \lambda / \sigma_k^2$, the weight at which the proximal step of
$\lambda\,\phi$ is a Gaussian denoiser of variance $\sigma_k^2$.  A noise level
or penalty given as a sequence is such a schedule.  In an unrolled network,
one network shared by the iterations is given the iteration index
(`steps=True` of {class}`~bartorch.learning.UNet`) and modulates its features
with it, which approaches the accuracy of a network per iteration at the
storage of one.  An iteration-dependent step has no fixed point, so
{class}`~bartorch.optim.FixedPoint` refuses it.

## Networks for complex, high-dimensional images

A network operates on real tensors of shape `(n, channels, *spatial)`.
{class}`~bartorch.learning.ComplexNet` lays out the real and imaginary parts
of every contrast, coefficient or channel of an image as its channels, so they
are denoised jointly.  Coefficient maps of a subspace basis differ in energy by
orders of magnitude; whitening (`normalize="whiten"`) decorrelates the
channels and scales them to unit variance before the call and reverts both
after it.  A time series is taken with a frame axis
(`frames=True`), convolved separately from the spatial axes and never
downsampled.

A 3D volume, or a series of them such as a cine or a fingerprinting
acquisition, rarely fits a network's activations in the memory of a scanner's
GPU.  The network is trained on patches, and
{class}`~bartorch.learning.Patchwise` applies it to the whole image a few
patches at a time: the image stays on the host, each group of patches is
copied to the device, passed through the network under mixed precision
(bfloat16 where the GPU supports it, float16 where it does not, such as a
T4) and copied back.  At inference the copies of one group run on a second
stream while the network computes on another, so the transfers are hidden
behind the computation.  What the device holds is the network and two groups
of patches.  A convolutional network sees zeros past a patch boundary, which
leaves seams at the boundaries of a fixed grid of patches; the grid is offset
at random at every call, so successive iterations place the seams differently.

## Training

An unrolled network of many iterations records every iteration's activations
during the backward pass.  Staged training bounds the memory:[^urman]

1. The denoiser is trained alone, on pairs of an image and a degraded copy of
   it, with the iteration index it is to be used at.
2. The unrolled network is trained one iteration at a time, each starting from
   the detached result of the previous one, with a loss on every iteration's
   image weighted to increase geometrically.
3. The whole network is trained end to end, with each iteration recomputed
   during the backward pass (checkpointing), so that memory holds one
   iteration's activations.

Per-iteration training requires the block's image to pass through that
iteration's denoiser, which holds for a proximal-gradient step and not for the
image of an ADMM step, its x-update.
{class}`~bartorch.learning.training.Reconstruction` runs each stage as a
Lightning module; the items stay where the dataset put them, and a network
inside decides where it runs.

Fully sampled references are rarely available for the data a learned
reconstruction is most needed for: a dynamic or high-dimensional acquisition
is undersampled because full sampling does not fit a breath-hold or a
reasonable scan time.  Self-supervision via data undersampling splits the acquired
samples $\Omega$ into disjoint sets $\Theta$ and $\Lambda$, reconstructs from
$\Theta$, and scores the reconstruction's k-space on $\Lambda$.[^ssdu]  A new split is drawn at every step
({func}`~bartorch.learning.split`), and the reconstruction at inference uses
all of $\Omega$.

## Uncertainty

Where the undersampling leaves the image underdetermined, a network fills in
what its training data suggest, and a hallucinated structure is not
distinguishable from anatomy in the image alone.  A spread is obtained by
repeating a randomized reconstruction and taking the voxel-wise variance ({func}`~bartorch.learning.moments`): with dropout active
in the network, from random subsets of the acquired samples, or with a random
patch grid.  Each spread measures one source of variability and none is the
reconstruction error or a posterior distribution.  Split conformal calibration
relates it to the error: on held-out images with references, the factor $q$ is
the empirical quantile of $|x - x_\mathrm{ref}| / s$ at the requested coverage
({func}`~bartorch.learning.calibrate`), and $q\,s$ is then an interval that
contains the error at that rate on new images from the same distribution.[^conformal]  The rate holds on average over voxels
and images, not at each voxel; how narrow the interval is where the
reconstruction is reliable depends on how well the spread follows the error.

## See also

- {doc}`differentiation`: backward passes through operators, solvers and
  unrolled iterations.
- {doc}`../api/learning`: the networks, iterations, splitting, uncertainty and
  training stages.
- {doc}`../examples/index`: the learned-reconstruction lessons.

## References

[^dpir]: Zhang K, Li Y, Zuo W, Zhang L, Van Gool L, Timofte R. Plug-and-play image restoration with deep denoiser prior. *IEEE Trans Pattern Anal Mach Intell* 44(10):6360–6376 (2022). [doi:10.1109/TPAMI.2021.3088914](https://doi.org/10.1109/TPAMI.2021.3088914)

[^urman]: Urman Y, Nishimura M, Abraham DR, Cao X, Setsompop K. Fully 3D unrolled magnetic resonance fingerprinting reconstruction via staged pretraining and implicit gridding. *Magn Reson Med* 96(5):2516–2529 (2026). [doi:10.1002/mrm.70500](https://doi.org/10.1002/mrm.70500)

[^ssdu]: Yaman B, Hosseini SAH, Moeller S, Ellermann J, Ugurbil K, Akcakaya M. Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data. *Magn Reson Med* 84(6):3172–3191 (2020). [doi:10.1002/mrm.28378](https://doi.org/10.1002/mrm.28378)

[^conformal]: Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. *Found Trends Mach Learn* 16(4):494–591 (2023). [doi:10.1561/2200000101](https://doi.org/10.1561/2200000101)
