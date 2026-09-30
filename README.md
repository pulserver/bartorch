[![Tests](https://github.com/pulserver/bartorch/actions/workflows/tests.yml/badge.svg)](https://github.com/pulserver/bartorch/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/pulserver/bartorch/branch/main/graph/badge.svg)](https://codecov.io/gh/pulserver/bartorch)
[![CodeFactor](https://www.codefactor.io/repository/github/pulserver/bartorch/badge)](https://www.codefactor.io/repository/github/pulserver/bartorch)
[![Docs](https://github.com/pulserver/bartorch/actions/workflows/docs.yml/badge.svg)](https://pulserver.github.io/bartorch/)
[![Lint](https://github.com/pulserver/bartorch/actions/workflows/lint.yml/badge.svg)](https://github.com/pulserver/bartorch/actions/workflows/lint.yml)
[![Typos](https://github.com/pulserver/bartorch/actions/workflows/typos.yml/badge.svg)](https://github.com/pulserver/bartorch/actions/workflows/typos.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/bartorch.svg)](https://pypi.org/project/bartorch/)
[![Downloads](https://img.shields.io/pypi/dm/bartorch.svg)](https://pypistats.org/packages/bartorch)
[![Python](https://img.shields.io/pypi/pyversions/bartorch.svg)](https://pypi.org/project/bartorch/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Wheels](https://img.shields.io/badge/wheels-Linux%20x86--64%20%7C%20macOS%20arm64%20%7C%20Windows%20x86--64-2f6f9f)](https://github.com/pulserver/bartorch/blob/main/.github/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f6f9f.svg)](https://github.com/pulserver/bartorch/blob/main/LICENSE)
[![Source](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/pulserver/bartorch)
[![Stars](https://img.shields.io/github/stars/pulserver/bartorch?style=flat&logo=github&color=2f6f9f)](https://github.com/pulserver/bartorch/stargazers)
[![FAIR checklist badge](https://fairsoftwarechecklist.net/badge.svg)](https://fairsoftwarechecklist.net/v0.2?f=31&a=32112&i=32322&r=133)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/bartorch/main/docs/_static/bartorch-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/pulserver/bartorch/main/docs/_static/bartorch-logo.svg" alt="bartorch" width="440">
  </picture>
</p>

bartorch makes BART, the Berkeley Advanced Reconstruction Toolbox, available in
Python on PyTorch tensors, in host memory or on a CUDA device.  BART is compiled
into the package and runs in the Python process, so a reconstruction written in
Python uses BART's parallel imaging, compressed sensing, non-Cartesian and
model-based methods without writing files or calling an executable, and can be
placed inside a larger PyTorch computation such as the training of an unrolled
network.

BART's encoding operators are composable Python objects, its iterative
algorithms are solver classes and differentiable iteration blocks, its
reconstructions (`pics`, `moba`, `mobafit`, POCSENSE) are pipelines assembled
from both, and its remaining commands are Python functions.  The arithmetic is
BART's except where bartorch substitutes a component: FINUFFT and cuFINUFFT
compute every non-uniform FFT, the FFT and BLAS/LAPACK routines come from MKL,
PyTorch's linked library or SciPy, and the MRI encoding operators run on
bartorch's own executor built from BART's operators.

## Features

- BART's reconstructions (`pics`, `moba`, `mobafit`, POCSENSE) as pipelines of
  operators and solvers, and its calibration, sampling and simulation commands
  (`ecalib`, `nlinv`, `traj`, `phantom`, ...) as functions of tensors.
- Corrections outside the reconstruction (EPI ghosting, bias field, gradient
  nonlinearity, off-resonance and susceptibility distortion) and rigid motion
  estimation from navigators.
- The `bartorch` command line, which accepts the arguments of `bart`.
- MRI encoding operators — Cartesian, non-Cartesian and wave-encoded SENSE,
  subspace models, B0 off-resonance correction — composed with
  `@` and `+` into single BART operators.
- BART's regularization terms and its CG, IST, FISTA, ADMM and primal-dual
  iterations, as solvers and as differentiable iteration blocks.
- Nonlinear operators, iteratively regularized Gauss-Newton, and quantitative
  signal models from TorchSim.
- Adapters for unrolled networks, plug-and-play denoisers and DeepInverse.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/bartorch/main/docs/_static/architecture-dark.svg">
    <img src="https://raw.githubusercontent.com/pulserver/bartorch/main/docs/_static/architecture.svg" alt="bartorch architecture: PyTorch tensors enter the command-style and composable interfaces, which call the bartorch C ABI through ctypes; the ABI runs the embedded BART, whose non-uniform Fourier transforms and linear algebra are served by substituted backends" width="760">
  </picture>
</p>

## Quick start

```bash
pip install bartorch
```

```python
import bartorch.tools as bt
from bartorch import apps, priors

kspace = bt.phantom(128, coils=8, kspace=True)  # (coils, z, y, x)
maps = bt.ecalib(kspace, maps=1)
image = apps.pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005), solver="fista")
```

The same reconstruction can be assembled from an encoding operator and a
solver; [Interfaces and execution](https://pulserver.github.io/bartorch/latest/explanation/execution-model.html)
describes which interface fits which task.

## Documentation

<https://pulserver.github.io/bartorch/> has the
[user guide](https://pulserver.github.io/bartorch/latest/guides/user/index.html)
(supported platforms, installation, reporting issues), the
[developer guide](https://pulserver.github.io/bartorch/latest/guides/developer/index.html),
conceptual [explanations](https://pulserver.github.io/bartorch/latest/explanation/index.html),
executed [examples](https://pulserver.github.io/bartorch/latest/auto_examples/index.html)
and the [API reference](https://pulserver.github.io/bartorch/latest/api/index.html),
with a version switcher between the development version and the releases.

## Citation

bartorch has no publication or archival DOI.  Work that uses it should cite
BART and the methods it applied (ESPIRiT, compressed sensing, nonlinear
inversion, ...), and report the bartorch version or commit and the pinned
BART revision; [Contributors and citation](https://pulserver.github.io/bartorch/latest/misc/contributors.html)
gives the references and what else a reproducible report records.

## License

bartorch is MIT-licensed.  The embedded BART (BSD-3-Clause) and FINUFFT
(Apache-2.0), the libraries FINUFFT's build compiles in with it, and the
vendored pocketfft (BSD-3-Clause) and BlocksRuntime (MIT or NCSA) keep their
own licenses; see
[License and third-party notices](https://pulserver.github.io/bartorch/latest/misc/license.html).
bartorch is an independent project, not affiliated with or endorsed by the BART
developers, the PyTorch Foundation or The Linux Foundation.  The logo combines
BART's mark with the PyTorch logo's flame; PyTorch, the PyTorch logo and any
related marks are trademarks of The Linux Foundation.
