# Explanations

The concepts behind bartorch, one page each, from the call into BART to
learned reconstruction and performance. Exact interfaces are in {doc}`../api/index`, and
complete workflows in the {doc}`examples <../gallery/index>`.

| Page | Contents |
| --- | --- |
| {doc}`execution-model` | Tools, apps and operators over one embedded BART; errors, devices and threads |
| {doc}`data-layout` | Tensor axes as BART dimensions, Fourier conventions, trajectory units and file formats |
| {doc}`encoding` | MRI forward operator, operator algebra, the plan it is lowered into |
| {doc}`non-cartesian` | NUFFT, density compensation, Toeplitz normal |
| {doc}`regularizers` | Penalties and their proximal operators |
| {doc}`solvers` | Choice of iteration, step size and regularization weight |
| {doc}`nonlinear` | Nonlinear forward models, IRGNM, joint coil estimation and parameter maps |
| {doc}`learned-reconstruction` | Gradients through operators and solvers; networks inside a reconstruction |
| {doc}`performance` | Devices, precision, threads and the libraries behind BLAS, LAPACK and the FFT |

```{toctree}
:hidden:

execution-model
data-layout
encoding
non-cartesian
regularizers
solvers
nonlinear
learned-reconstruction
performance
```
