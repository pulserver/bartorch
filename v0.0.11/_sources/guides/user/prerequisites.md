# Prerequisites and supported platforms

## Requirements

| Requirement | Version |
| --- | --- |
| Python | 3.10 or later; each wheel is tested on 3.10 and 3.14 |
| PyTorch | 2.2 or later (2.3 on macOS, the first to carry `libomp.dylib`; 2.7.1 on Linux, the first whose `libgomp` is named `libgomp.so.1`), CPU or CUDA build |
| NumPy, SciPy | NumPy 1.24 and SciPy 1.10 or later; MRI-NUFFT, a dependency, raises these to NumPy 2.2 and SciPy 1.13 |
| BlochSim, MRI-NUFFT | BlochSim 0.0.10 and MRI-NUFFT 1.0 or later, installed as dependencies |

## Platforms

| Platform | Distribution | Notes |
| --- | --- | --- |
| Linux x86-64, glibc 2.28 or later | Wheel | CPU build on PyPI; CUDA build attached to the GitHub release of each version |
| macOS 11 or later on Apple silicon | Wheel | CPU only; OpenMP is the runtime PyTorch installs (see {ref}`openmp-runtime`) |
| Linux aarch64 | Source distribution | BART and FINUFFT are compiled on installation |
| macOS on Intel | Source distribution | BART and FINUFFT are compiled on installation |
| Windows 10 or later on x86-64 | Wheel | CPU only; OpenMP is the runtime PyTorch installs |

A source installation needs the toolchain listed under {ref}`source-builds`.
Apple MPS devices are not supported; the device paths are CPU and CUDA.

## Optional components

| Extra | Installs | Needed for |
| --- | --- | --- |
| `mkl` | Intel MKL (Linux x86-64) | MKL as the source of BLAS, LAPACK and FFT routines, and of FINUFFT's FFT |
| `deepinv` | DeepInverse | {func}`bartorch.interop.to_deepinv`, and DeepInverse's denoisers for {class}`~bartorch.priors.ImplicitPrior` |
| `learning` | Lightning, TorchIO | The training stages and the complex-valued augmentation in `bartorch.learning` ({doc}`../../api/learning`) |
| `correct` | SimpleITK | Bias field and gradient nonlinearity correction in {mod}`bartorch.tools` |
| `motion` | SimpleITK | Rigid registration of navigator planes in {mod}`bartorch.tools` |
| `pyhysco` | PyHySCO (GPL-3.0-only) | {func}`bartorch.tools.correct_susceptibility` |

PyHySCO is not distributed with bartorch; it is imported only when
{func}`~bartorch.tools.correct_susceptibility` is called.  The spiral
deblurring of {func}`~bartorch.tools.deblur` uses Triton on a CUDA device when
it is installed.

The examples additionally need `brainweb-dl`, `matplotlib` and `cmap`; the
learned-regularization examples `lightning`, `torchio`, `monai` and `deepinv`,
and the tours `SimpleITK` ({doc}`../../auto_examples/index`).
