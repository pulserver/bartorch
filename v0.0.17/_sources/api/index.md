# API reference

Exact contracts of the public objects: parameters, defaults, shapes, units,
conventions and restrictions.  Concepts are in {doc}`../explanation/index`,
complete workflows in the {doc}`examples <../gallery/index>`.

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`functions` | `bartorch` | Fourier and non-uniform Fourier transforms, density compensation, k-space windows, wavelet transforms, thresholding, array functions, runtime settings |
| {doc}`tools` | `bartorch.tools` | BART commands as functions: simulation, sampling, calibration, preprocessing, registration, metrics, low-rank completion; corrections and rigid motion estimation |
| {doc}`linop` | `bartorch.linop` | Linear operators, operator algebra and MRI encoding operators |
| {doc}`nlop` | `bartorch.nlop` | Nonlinear operators, MRI and signal models, Gauss-Newton methods |
| {doc}`optim` | `bartorch.optim` | Iterative solvers, iteration blocks and data scaling |
| {doc}`priors` | `bartorch.priors` | Regularization terms, plug-and-play priors and BART's denoisers |
| {doc}`apps` | `bartorch.apps` | BART reconstruction pipelines assembled from operators and solvers |
| {doc}`learning` | `bartorch.learning` | Networks for complex images, unrolled iterations, patchwise execution, self-supervised splitting, uncertainty and training stages |
| {doc}`interop` | `bartorch.interop` | DeepInverse physics adapter |
| {doc}`io` | `bartorch.io` | CFL files, ISMRMRD raw data, DICOM and NIfTI images |
| {doc}`cli` | `bartorch.cli` | The `bartorch` command line |

## BART correspondence

bartorch accepts BART's command lines unchanged: `bartorch <command> ...` takes
the arguments `bart <command> ...` takes ({doc}`cli`).  In Python, arrays are
C-order tensors whose shape is BART's dimension vector reversed, coils first,
`(coils, z, y, x)` ({doc}`../explanation/data-layout`), and axes are indices,
never bitmasks.

| BART | bartorch |
| --- | --- |
| `bart <command> ...` | `bartorch <command> ...`, or `bartorch.tools.<command>(...)` |
| `bart fft -u -i 7` | `bartorch.ifft(x, (-3, -2, -1), unitary=True)` |
| `bart nufft` | `bartorch.nufft`, `linop.NUFFT` |
| `bart pics -R W:...` | `apps.pics(..., regularizers=priors.Wavelet(...))` |
| `bart pics -R T:...` | `apps.pics(..., regularizers=priors.TotalVariation(...))` |
| `bart pics -t traj` | `apps.pics(..., traj=traj)` |
| `bart moba`, `bart mobafit` | `apps.moba`, `apps.mobafit` |
| `bart ecalib`, `bart nlinv`, `bart traj`, `bart phantom` | `bartorch.tools.ecalib`, `.nlinv`, `.traj`, `.phantom` |

BART's `fft` is unnormalised unless `-u`; `linop.FFT` is unitary.

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
