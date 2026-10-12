# Model-based reconstruction

```{admonition} TL;DR
:class: tldr

- A forward model nonlinear in its unknowns, image times coils (`nlinv`) or parameter maps through a signal model (`moba`), is a {class}`~bartorch.nlop.NonlinearOperator` with a derivative and its adjoint.
- {class}`~bartorch.nlop.IRGNM` solves it by iteratively regularized Gauss-Newton, BART's own loop, with BlochSim signal models whose bounds keep every iterate physical.
- The `moba` and `mobafit` apps fit those models and answer in named maps; a command line they cannot express runs as BART's command.
```

You meet a nonlinear forward model when the coil sensitivities are unknowns
alongside the image, or when the image is a function of tissue parameters such
as $T_1$ or $T_2$. Model-based reconstruction fits the parameter maps directly
to k-space, $y_{c,e} = P_e F (S_c \cdot M_e(\theta))$, so every contrast
constrains the same few maps. Lesson 4 of the {doc}`Course <../gallery/course>`
estimates image and coils jointly, and lesson 9 fits a $T_2$ map to k-space.

## Capabilities

:::{container} capabilities

- **Represents a nonlinear operator with its value, its derivative at a point and the adjoint of that derivative, composable with linear operators by `@`.**

  Code: {mod}`bartorch.nlop`. Tests: *a derivative agrees with a finite difference*; *a Jacobian is a linear operator with the adjoint identity*; *a linear operator still composes with a nonlinear one* (`test_nlop_algebra.py`).
- **Runs iteratively regularized Gauss-Newton with the same arithmetic as BART's own loop, to the last bit.**

  Code: {class}`~bartorch.nlop.IRGNM`, {class}`~bartorch.nlop.IRGNMBlock`. Tests: *the loop without an inner solver is the first form to the last bit*; *the Python loop with conjugate gradients is the library to the last bit* (`test_nlop_irgnm.py`).
- **Solves each linearized problem with any solver of {mod}`bartorch.optim`, whose regularization terms then penalize the step.**

  Code: `IRGNM(inner=...)`. Tests: *an unregularized inner solver reaches the same answer*; *a threshold pulls the answer away from the unregularized one* (`test_nlop_irgnm.py`).
- **Estimates image and coil sensitivities jointly, as `nlinv` does, with the coils under a Sobolev weighting.**

  Code: {class}`~bartorch.nlop.NonlinearSense`, {func}`~bartorch.tools.nlinv`. Tests: *a Cartesian fit is the nlinv tool to the last bit*; *the coil weighting damps high spatial frequencies* (`test_nlop_mri.py`).
- **Wraps BlochSim signal models (inversion recovery, multi-echo decay, any simulated sequence) as nonlinear operators, without forming a Jacobian.**

  Code: {class}`~bartorch.nlop.SignalModel`, {func}`~bartorch.nlop.MultiEcho`, {func}`~bartorch.nlop.Bloch`. Tests: *the multi-echo decay is what BART computes*; *the adjoint is the adjoint of the derivative* (`test_nlop_blochsim.py`); *the Look-Locker recovery is BART's* (`test_signal_models.py`).
- **Keeps each bounded parameter inside its bounds at every iterate, and returns named maps in their own units.**

  Code: {meth}`~bartorch.nlop.SignalModel.initial`, {meth}`~bartorch.nlop.SignalModel.split`. Tests: *a bound keeps every iterate physical*; *a starting point and the maps it stands for round trip* (`test_nlop_blochsim.py`).
- **Fits a signal model voxel by voxel to images, or directly to k-space with known or jointly estimated coils.**

  Code: {func}`~bartorch.apps.mobafit`, {func}`~bartorch.apps.moba`. Tests: *a decay is fitted to the time it was made from*; *moba fits the decay behind known coils*; *moba estimates the coils with the decay* (`test_apps.py`).
- **Answers a `moba` or `mobafit` command line with the app, to the tolerance of a fit, and sends one the app cannot express to BART's command.**

  Code: {mod}`bartorch.cli`. Tests: *moba writes the command's maps to the tolerance of a fit*; *a moba the app cannot express goes to the command*; *a mobafit the app cannot express goes to the command* (`test_cli.py`).

:::

## Division of responsibility

- **BART**: the Gauss-Newton step and its conjugate-gradient inverse, the
  `noir` coil model, and the `moba` and `mobafit` commands a command line falls
  back to.
- **bartorch**: the operator algebra, the step written out in Python, and the
  apps that assemble `moba` and `mobafit` from them.
- **BlochSim**: the signal models, their derivatives and the bounded
  parameterisation.

## Method

```{image} ../_static/figures/model-irgnm.svg
:class: plot
:alt: Left, the eight echoes of one voxel with the decay fitted after Gauss-Newton steps 2, 6 and 16; right, the median relative T2 error over sixteen steps beside the Tikhonov weight, which halves each step
```

A $T_2$ decay fitted voxel by voxel by {class}`~bartorch.nlop.IRGNMBlock`
from a start of 80 ms, regularized towards the middle of the $T_2$ bound
(`scripts/figures/model.py`). The early, strongly weighted steps stay near
the centre; the error falls once $\alpha_k$ is small enough for the data to
dominate.

### Gauss-Newton step

Each step linearizes $F$ at the iterate $x_k$ and solves a Tikhonov-regularized
least-squares problem centred on $x_{\mathrm{ref}}$:

$$
x_{k+1} = x_k + (DF^H DF + \alpha_k I)^{-1}\left[DF^H (y - F(x_k)) - \alpha_k (x_k - x_{\mathrm{ref}})\right],
$$

with $DF = DF_{x_k}$. The weight decays geometrically,
$\alpha_{k+1} = (\alpha_k - \alpha_{\min})/q + \alpha_{\min}$, with $q = 2$ by
default.[^bakushinsky] The number of steps
is a regularization parameter: stopping early gives a smoother estimate.

### Sobolev weighting

In $S_c \cdot x$ any nonzero function can move from the coils to the image
without changing the data. `nlinv` solves for the coils as k-space
coefficients $\hat{s}$ with $S = \mathcal{F}^{-1}[(1 + a|k|^2)^{-b/2}\hat{s}]$,
so the penalty on $\hat{s}$ is a Sobolev norm of $S$ and the image receives the
high spatial frequencies.[^nlinv] `moba` uses the same weighting for its coils,
and `apps.moba(smooth=...)` applies it to a parameter map such as B1 or B0.

### Bounded parameters

A bounded parameter is solved for through a transformed variable: a scaled
sigmoid for a two-sided bound, a softplus for a one-sided one. Every iterate
then maps inside the bound, and a millisecond relaxation time and a
dimensionless fraction take steps of comparable size. The data term remains
nonconvex in $\theta$, so the result depends on the start.[^sumpf] Which
command lines the apps answer, and why the others go to BART, is listed in
{doc}`../guides/developer/internals/model-based`.

## Examples

- {doc}`../auto_examples/02-parallel-imaging/02-nonlinear-inversion`: image
  and coils by Gauss-Newton.
- {doc}`../auto_examples/05-model-based/01-quantitative-models`: a $T_2$ map
  by voxel-wise fit and by fit to k-space.
- {doc}`../auto_examples/08-workflows/02-subspace-t1-mapping`: the linear
  alternative, a subspace reconstruction followed by a fit.
- {doc}`../api/nlop` and {doc}`../api/apps`: the operators, the signal
  models, IRGNM and the apps.

## References

[^nlinv]: Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear inversion—joint estimation of coil sensitivities and image content. *Magn Reson Med* 60(3):674–682 (2008). [doi:10.1002/mrm.21691](https://doi.org/10.1002/mrm.21691)

[^sumpf]: Sumpf TJ, Uecker M, Boretius S, Frahm J. Model-based nonlinear inverse reconstruction for T2 mapping using highly undersampled spin-echo MRI. *J Magn Reson Imaging* 34(2):420–428 (2011). [doi:10.1002/jmri.22634](https://doi.org/10.1002/jmri.22634)

[^bakushinsky]: Bakushinsky AB, Kokurin MY. *Iterative Methods for Approximate Solution of Inverse Problems.* Springer (2004). [doi:10.1007/978-1-4020-3122-9](https://doi.org/10.1007/978-1-4020-3122-9)
