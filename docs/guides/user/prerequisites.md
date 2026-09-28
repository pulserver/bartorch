# Prerequisites and supported platforms

## Requirements

| Requirement | Version |
| --- | --- |
| Python | 3.10 or later; each wheel is tested on 3.10 and 3.14 |
| PyTorch | 2.2 or later, CPU or CUDA build |
| NumPy, SciPy | NumPy 1.24 and SciPy 1.10 or later; MRI-NUFFT, a dependency, raises these to NumPy 2.2 and SciPy 1.13 |
| TorchSim, MRI-NUFFT | TorchSim 0.0.5 and MRI-NUFFT 1.0 or later, installed as dependencies |

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
| `mkl` | Intel MKL (Linux x86-64) | MKL as the source of BLAS, LAPACK and FFT routines |
| `deepinv` | DeepInverse | {func}`bartorch.interop.to_deepinv` |

The executed examples additionally need `brainweb-dl`, `matplotlib` and `cmap`,
and the deep-learning example `lightning`, `torchio`, `monai` and `deepinv`.
