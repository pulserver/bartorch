[![Tests](https://github.com/pulserver/bartorch/actions/workflows/tests.yml/badge.svg)](https://github.com/pulserver/bartorch/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/bartorch.svg)](https://pypi.org/project/bartorch/)
[![Docs: stable](https://img.shields.io/badge/docs-stable-2f6f9f)](https://pulserver.github.io/bartorch/stable/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f6f9f.svg)](https://github.com/pulserver/bartorch/blob/main/LICENSE)

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/bartorch/main/docs/_static/bartorch-logo-dark.svg">
  <img src="https://raw.githubusercontent.com/pulserver/bartorch/main/docs/_static/bartorch-logo.svg" alt="bartorch" width="440">
</picture></p>

bartorch puts BART, the Berkeley Advanced Reconstruction Toolbox, inside your
Python process. You hand it PyTorch tensors, on the CPU or on a CUDA device,
and BART works on them where they are: no `.cfl` files, no `bart` executable,
no subprocess. What comes back is a tensor, so a BART reconstruction can sit
inside any PyTorch code, including the training loop of an unrolled network.

You use BART in one of three ways, from the most ready-made to the most
flexible:

- **Tools.** Every BART command is a Python function: `bt.ecalib(kspace)`,
  `bt.phantom(128)`, `bt.traj(...)`. If you know BART's command line, you
  already know them.
- **Apps.** BART's reconstructions (`pics`, `moba`, `mobafit`, POCSENSE) are
  functions that take k-space and return an image.
- **Operators and solvers.** You write the encoding yourself, `A = P F S` or
  anything else, compose it with `@` and `+`, and solve it with one of BART's
  iterations. This is where a reconstruction BART has no command for lives,
  and it is differentiable.

The three are the same library underneath: `apps.pics` *is* an encoding
operator and a solver, built for you.

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/bartorch/main/docs/_static/architecture-dark.svg">
  <img src="https://raw.githubusercontent.com/pulserver/bartorch/main/docs/_static/architecture.svg" alt="bartorch architecture" width="760">
</picture></p>

## How it works

1. You pass a tensor. A C-order tensor of shape `(coils, z, y, x)` holds the
   same bytes as a BART array of dimensions `[x, y, z, coils]`, so bartorch
   reverses the shape and hands BART the tensor's memory. Nothing is copied.
2. BART runs in the same process, compiled into the package. When BART
   reports an error, you get a Python exception (`bartorch.BartError`) with
   BART's message, and your interpreter keeps running.
3. A few parts of BART are swapped for faster ones: every non-uniform FFT is
   FINUFFT (cuFINUFFT on a card), and the FFT, BLAS and LAPACK come from MKL,
   PyTorch's own library or SciPy. The numbers are BART's everywhere else.
4. A CUDA tensor stays on its card. The operators and solvers run there
   directly, and a tool runs on the card when that has been checked to be
   safe; otherwise its result is copied back to the card for you.
5. An operator applied to a tensor that requires a gradient is recorded by
   autograd, and its backward pass is its adjoint. You can train through a
   reconstruction.

## Tools: BART's commands

```python
import bartorch.tools as bt

kspace = bt.phantom(128, coils=8, kspace=True)  # (coils, z, y, x)
compressed = bt.cc(kspace, p=4)
maps = bt.ecalib(compressed, maps=1)
```

- Options are keyword arguments, under BART's long name or its one-letter
  flag.
- Where BART takes a bitmask of dimensions, you pass tensor axes:
  `bartorch.fft(x, axes=(-2, -1))`.
- Every command BART has is here, as a function or, for the few with a
  reason not to be, listed as such in the documentation.

## Apps: BART's reconstructions

```python
from bartorch import apps, priors

image = apps.pics(compressed, maps, regularizers=priors.Wavelet((-1, -2), 0.005))
```

- A regularization term is an object from `bartorch.priors`, not a `-R`
  string; its axes are tensor axes.
- `apps.pics` gives the same bits as BART's `pics` command, and the
  documentation's tests hold it to that.

## Operators and solvers: your own reconstruction

```python
import torch

from bartorch import linop, optim, priors

A = linop.CartesianSense(maps, (128, 128), pattern)  # A = P F S
image = optim.FISTA(priors.Wavelet((-1, -2), 0.005), maxiter=100)(kspace, A)

B = A @ linop.Diagonal(phase)  # one more factor, still one BART operator
loss = (B(x) - kspace).abs().pow(2).sum()
loss.backward()  # x.grad is 2 B^H (B x - y)
```

- `A(x)` applies the operator, `A.H(y)` its adjoint, and `A.H @ A` its
  normal. A composition is handed to BART as one operator, so a solver
  iterating on it never comes back to Python.
- `A.plan` tells you how BART will run what you composed: which transform,
  which factors were fused, and whether it fell back to a slower path.
- An operator you write in Python (a forward and an adjoint) joins a
  composition like any other.

## Command line

The `bartorch` command takes the same arguments as `bart`, so a script written
for BART runs with the name changed and no BART installation:

```bash
bartorch phantom -k -s 8 kspace
bartorch ecalib -m 1 kspace maps
bartorch pics -R W:3:0:0.005 kspace maps image
```

## Install

```bash
pip install bartorch
pip install "bartorch[cu12]"  # with CUDA 12; use cu13 for CUDA 13
```

Wheels are built for Linux x86-64, macOS arm64 and Windows x86-64; the CUDA
build is Linux x86-64 only. More functionality and faster backends come as
optional dependencies, listed in
[Installation](https://pulserver.github.io/bartorch/stable/guides/user/installation.html).

## Learn more

- [Course](https://pulserver.github.io/bartorch/stable/auto_examples/index.html):
  from k-space to an image, then parallel imaging, regularization,
  non-Cartesian and model-based reconstruction, and learned regularization.
  Every lesson opens in Colab.
- [Documentation](https://pulserver.github.io/bartorch/stable/), for every
  function, operator and option.

## How to cite

bartorch has no publication of its own. Please cite BART
([Uecker et al., doi:10.5281/zenodo.592960](https://doi.org/10.5281/zenodo.592960))
and the papers behind the methods you used (ESPIRiT, compressed sensing,
nonlinear inversion, ...), and report the bartorch version.
[Contributors and citation](https://pulserver.github.io/bartorch/stable/misc/contributors.html)
lists the references.

## License

MIT. The embedded BART (BSD-3-Clause), FINUFFT (Apache-2.0) and the other
bundled code keep their own licenses; see
[License and third-party notices](https://pulserver.github.io/bartorch/stable/misc/license.html).
bartorch is an independent project, not affiliated with or endorsed by the BART
developers, the PyTorch Foundation or The Linux Foundation. The logo combines
BART's mark with the PyTorch logo's flame; PyTorch, the PyTorch logo and any
related marks are trademarks of The Linux Foundation.
