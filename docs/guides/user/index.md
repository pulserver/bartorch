# User guide

Installation and support.  The array layout, trajectory units and
Fourier-transform conventions are in {doc}`../../explanation/data-layout`, the
other concepts in {doc}`../../explanation/index`, and complete reconstructions
in the {doc}`Examples <../../gallery/index>`.  A build from source or from a
checkout is described in {doc}`../developer/installation`.

## Supported platforms

| Platform | Distribution | Device |
| --- | --- | --- |
| Linux x86-64, glibc 2.28 or later | Wheel | CPU; CUDA with the `cu12` or `cu13` extra |
| macOS 11 or later on Apple silicon | Wheel | CPU |
| Windows 10 or later on x86-64 | Wheel | CPU |
| Linux aarch64, macOS on Intel | Source distribution, compiled on installation ({ref}`source-builds`) | CPU |

Apple MPS devices are not supported.

| Requirement | Version |
| --- | --- |
| Python | 3.10 or later; the wheels are tested on 3.10 and 3.14 |
| PyTorch | CPU or CUDA build: 2.7.1 or later on Linux, 2.3 or later on macOS, 2.2 or later on Windows.  The library binds to the OpenMP runtime PyTorch carries ({ref}`openmp-runtime`), which sets the Linux and macOS floors |
| NumPy, SciPy | 2.2 and 1.13 or later |
| BlochSim, MRI-NUFFT | 0.0.10 and 1.0 or later |
| SimpleITK | 2.3 or later |
| ismrmrd, pydicom, nibabel | 1.14, 3.0 and 5.0 or later |

pip installs all of these with bartorch; install PyTorch first to choose its
build.

## Installation

### PyTorch build

Install PyTorch first, with the command the
[PyTorch installation selector](https://pytorch.org/get-started/locally/) gives
for your device: the CPU build, or the build for your CUDA version.  If PyTorch
is missing, pip installs the default build from PyPI with bartorch.  The
version floors are listed under *Supported platforms*.

### Package and extras

```bash
python -m pip install bartorch
python -m pip install 'bartorch[mkl,io]'   # with extras, comma-separated
```

The wheel contains the compiled library with BART and FINUFFT embedded; you
need no BART or FINUFFT installation and no compiler.

Check the installation by building a phantom:

```python
import bartorch
import bartorch.tools as bt

print(bartorch.__version__, bartorch.bart_version())
print(bartorch.build_info())
print(bartorch.backend_sources())
print(bt.phantom(32).shape)
```

{func}`bartorch.build_info` reports the compiler, the OpenMP and CUDA
configuration of the library, and the FINUFFT version and FFT compiled into it.
{func}`bartorch.backend_sources` reports the library serving each BLAS and
LAPACK routine, BART's FFT and FINUFFT's FFT: MKL when the `mkl` extra is
installed, otherwise the routines PyTorch links, and SciPy's for the rest.

### Optional dependencies

| Extra | Installs | Needed for |
| --- | --- | --- |
| `cu12`, `cu13` | `bartorch-cuda12`, `bartorch-cuda13` (Linux x86-64) | CUDA devices ({ref}`gpu-builds`) |
| `mkl` | Intel MKL (Linux x86-64) | MKL as the source of the BLAS, LAPACK and FFT routines, and of FINUFFT's FFT |
| `learning` | Lightning, TorchIO | {class}`~bartorch.learning.Reconstruction` and {class}`~bartorch.learning.RandomGain` ({doc}`../../api/learning`) |
| `pyhysco` | PyHySCO (GPL-3.0-only) | {func}`bartorch.tools.correct_susceptibility` |

Two packages are used when present and have no extra: DeepInverse
(`pip install deepinv`) for {func}`bartorch.interop.to_deepinv`, and Triton for
the spiral deblurring of {func}`~bartorch.tools.deblur` on a CUDA device.
PyHySCO is not distributed with bartorch and is imported only when
{func}`~bartorch.tools.correct_susceptibility` is called.  The packages the
examples import are listed on the {doc}`Examples <../../gallery/index>` page.

(gpu-builds)=
### GPU builds

A CUDA device needs a CUDA build of PyTorch and bartorch's library compiled for
the same CUDA major version, a separate package installed by an extra.  Both
exist for Linux x86-64 only.

| PyTorch build (`torch.version.cuda`) | bartorch |
| --- | --- |
| 12.x | `python -m pip install 'bartorch[cu12]'` |
| 13.x | `python -m pip install 'bartorch[cu13]'` |

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'bartorch[cu12]'
```

bartorch loads the CUDA build in place of its own library when it matches
PyTorch's CUDA major version and the installed bartorch release, and raises an
`ImportError` naming the extra to install otherwise.
{func}`bartorch.cuda_available` and `cuda=ON` in {func}`bartorch.build_info`
report the library loaded.  The CUDA build contains BART's kernels and
cuFINUFFT for compute capabilities 7.5, 8.0 and 9.0, with the 8.0 code serving
8.6 and 8.9 and PTX for newer devices, and links the CUDA runtime, cuFFT and
cuBLAS installed with PyTorch's CUDA build.

### Command-line interface

The `bartorch` command accepts the command lines of BART's `bart` executable
and operates on CFL files:

```sh
bartorch pics -l1 -r0.01 -i30 kspace sensitivities image
bartorch --list
```

A script written for `bart` runs with the command name replaced, or with a
`bart` symbolic link to `bartorch` earlier on the `PATH`.  The commands that
run as {mod}`bartorch.apps` pipelines are listed in {doc}`../../api/cli`.

## Getting help

**Issue reports.** Bugs, documentation defects and feature requests go to the
[issue tracker](https://github.com/pulserver/bartorch/issues/new/choose), whose
forms ask for the information below.  Search the existing issues for the error
message and the function name first.  A bug report contains:

- the observed and the expected behaviour, with the complete traceback;
- a minimal script on synthetic data, {func}`bartorch.tools.phantom` or a
  random tensor with a fixed seed, rather than acquired data;
- the shapes, dtypes and devices of the inputs, the axis and trajectory
  conventions assumed, and the options used, including regularization terms
  and Fourier-transform normalization;
- `bartorch.__version__`, the output of {func}`bartorch.build_info` (which
  includes the BART and FINUFFT versions compiled in), the Python and PyTorch
  versions, the operating system and the installation command;
- for a build from a checkout, `git rev-parse HEAD` and
  `git -C external/bart rev-parse HEAD`;
- on a CUDA device, the GPU model, the driver version, `torch.version.cuda`,
  `torch.cuda.is_available()` and {func}`bartorch.cuda_available`.

If importing bartorch or loading its library fails, report that error; the
functions above need the library.  A numerical discrepancy report states the
reference compared with (an independent implementation, an explicit sum, a
closed form) and the relative error observed.  A performance report states the
problem size, the thread count, whether the timing includes warm-up and device
synchronization, and the peak memory.  Share only data you may publish, and
remove patient identifiers, credentials and private paths from logs.

**Questions and proposals.** The repository has no GitHub Discussions;
questions and proposals go to the issue tracker.  File a usage question the
documentation does not answer as a documentation issue, and a proposal as a
feature request stating the reconstruction problem and the interface proposed.
Discuss a change to the public API in an issue before opening a pull request.

**Security.** Report a vulnerability privately, through the **Report a
vulnerability** button under the repository's
[Security](https://github.com/pulserver/bartorch/security) tab, not as a public
issue.  The policy, including the supported versions, is
[`SECURITY.md`](https://github.com/pulserver/bartorch/blob/main/SECURITY.md).
