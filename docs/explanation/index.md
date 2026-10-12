# Explanation

The concepts behind bartorch, one page each, in the order data meets them:
from the call into BART, through the encoding and the inverse problem, to
learned reconstruction and performance. Each page opens with a summary, lists
what bartorch does with the code and the tests behind each item, and shows the
mechanism in one figure. Exact interfaces are in {doc}`../api/index`, and
complete workflows in the {doc}`examples <../auto_examples/index>`.

| Page | Concept |
| --- | --- |
| {doc}`execution-model` | Tools, apps and operators over one embedded BART; errors, devices and threads |
| {doc}`data-layout` | Tensor axes as BART dimensions, Fourier conventions, trajectory units and file formats |
| {doc}`encoding` | The MRI forward operator, its algebra, and the plan it is lowered into |
| {doc}`non-cartesian` | The NUFFT, density compensation and the Toeplitz normal |
| {doc}`regularizers` | Penalties and their proximal operators |
| {doc}`solvers` | Choosing an iteration, its step size and the regularization weight |
| {doc}`nonlinear` | Nonlinear forward models, IRGNM, joint coil estimation and parameter maps |
| {doc}`learned-reconstruction` | Gradients through operators and solvers, and networks inside a reconstruction |
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
