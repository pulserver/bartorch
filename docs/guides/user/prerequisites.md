# Requirements and supported platforms

## Requirements

| Requirement | Version |
| --- | --- |
| Python | 3.10 or later; the wheels are tested on 3.10 and 3.14 |
| PyTorch | CPU or CUDA build: 2.7.1 or later on Linux, 2.3 or later on macOS, 2.2 or later on Windows.  The library binds to the OpenMP runtime PyTorch carries ({ref}`openmp-runtime`), which sets the Linux and macOS floors |
| NumPy, SciPy | 2.2 and 1.13 or later |
| BlochSim, MRI-NUFFT | 0.0.10 and 1.0 or later |

pip installs all of these with bartorch; install PyTorch first to choose its
build ({doc}`installation`).

## Platforms

| Platform | Distribution | Device |
| --- | --- | --- |
| Linux x86-64, glibc 2.28 or later | Wheel | CPU; CUDA with the `cu12` or `cu13` extra |
| macOS 11 or later on Apple silicon | Wheel | CPU |
| Windows 10 or later on x86-64 | Wheel | CPU |
| Linux aarch64, macOS on Intel | Source distribution, compiled on installation ({ref}`source-builds`) | CPU |

Apple MPS devices are not supported.

## Optional dependencies

| Extra | Installs | Needed for |
| --- | --- | --- |
| `cu12`, `cu13` | `bartorch-cuda12`, `bartorch-cuda13` (Linux x86-64) | CUDA devices ({ref}`gpu-builds`) |
| `mkl` | Intel MKL (Linux x86-64) | MKL as the source of the BLAS, LAPACK and FFT routines, and of FINUFFT's FFT |
| `io` | ismrmrd, pydicom, nibabel | ISMRMRD raw data, DICOM and NIfTI in {mod}`bartorch.io` |
| `learning` | Lightning, TorchIO | {class}`~bartorch.learning.Reconstruction` and {class}`~bartorch.learning.RandomGain` ({doc}`../../api/learning`) |
| `correct` | SimpleITK | Bias field and gradient nonlinearity correction in {mod}`bartorch.tools` |
| `motion` | SimpleITK | Rigid registration of navigator planes in {mod}`bartorch.tools` |
| `pyhysco` | PyHySCO (GPL-3.0-only) | {func}`bartorch.tools.correct_susceptibility` |

Two packages are used when present and have no extra: DeepInverse
(`pip install deepinv`) for {func}`bartorch.interop.to_deepinv`, and Triton for
the spiral deblurring of {func}`~bartorch.tools.deblur` on a CUDA device.
PyHySCO is not distributed with bartorch and is imported only when
{func}`~bartorch.tools.correct_susceptibility` is called.  The packages the
examples import are listed on the {doc}`Examples <../../auto_examples/index>`
page.
