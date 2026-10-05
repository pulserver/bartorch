# API reference

Exact contracts of the public objects: parameters, defaults, shapes, units,
conventions and restrictions.  Concepts are in {doc}`../explanation/index`,
complete workflows in the {doc}`examples <../auto_examples/index>`.

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`functions` | `bartorch` | Fourier and non-uniform Fourier transforms, density compensation, k-space windows, wavelet transforms, thresholding, array functions, runtime settings |
| {doc}`tools` | `bartorch.tools` | BART's commands as functions — simulation, sampling, calibration, preprocessing, registration, metrics and low-rank completion — and corrections and rigid motion estimation outside the reconstruction |
| {doc}`linop` | `bartorch.linop` | Linear operators, operator algebra and MRI encoding operators |
| {doc}`nlop` | `bartorch.nlop` | Nonlinear operators, MRI and signal models, Gauss-Newton methods |
| {doc}`optim` | `bartorch.optim` | Iterative solvers, iteration blocks and data scaling |
| {doc}`priors` | `bartorch.priors` | Regularization terms, plug-and-play priors and BART's denoisers |
| {doc}`apps` | `bartorch.apps` | BART reconstruction pipelines assembled from operators and solvers |
| {doc}`learning` | `bartorch.learning` | Networks for complex images, unrolled iterations, patchwise execution, self-supervised splitting, uncertainty and training stages |
| {doc}`interop` | `bartorch.interop` | DeepInverse physics adapter |
| {doc}`io` | `bartorch.io` | CFL files, ISMRMRD raw data, DICOM and NIfTI images |
| {doc}`cli` | `bartorch.cli` | The `bartorch` command line |

```{toctree}
:hidden:

functions
tools
linop
nlop
optim
priors
apps
learning
interop
io
cli
```
