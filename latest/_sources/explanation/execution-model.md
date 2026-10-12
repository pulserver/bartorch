# Execution model

```{admonition} TL;DR
:class: tldr

- BART is compiled into the package and runs in your Python process, on your tensors, with no files and no subprocess.
- You reach it through tools (BART's commands), apps (its reconstructions) or operators and solvers (your own reconstruction), all over the same library.
- An error inside BART raises {class}`~bartorch.BartError`, and your interpreter keeps running.
```

You call BART as you would call any PyTorch function: you pass tensors, on the
CPU or on a CUDA device, and you get tensors back. Choose the level that fits
the task. A standard reconstruction is one call to an app. An encoding BART has
no command for, a solver inside an outer loop, or a gradient through the
reconstruction needs the operators and solvers. Calibration, trajectories and
simulation are tools. Lesson 1 of the {doc}`Course <../gallery/course>`
calls a command on a tensor, and lesson 6 assembles `pics` by hand.

## Capabilities

:::{container} capabilities

- **Runs every BART command as a function of tensors, in this process.**

  Code: {mod}`bartorch.tools`. Tests: *every command is curated, derived or private* (`test_tools.py`); *fft matches numpy on two axes* (`test_core.py`).
- **Turns an error or a failed assertion inside BART into {class}`~bartorch.BartError`, with BART's message, and leaves the library ready for the next call.**

  Code: `bartorch_command` in `src/csrc/abi/api.c`. Tests: *a failing tool raises with BART's message*; *a failed assertion inside BART reaches the caller*; *a failed command leaves the next one free to run* (`test_core.py`).
- **Hands BART an operator's tensors without copying them, and copies a command's inputs because some commands write into theirs.**

  Code: `_dispatch.py`, {func}`~bartorch.set_copy_inputs`. Tests: *inputs are left untouched by a tool that writes into them*; *scratch inputs skip the copy* (`test_core.py`).
- **Runs an app with the same result as BART's command: bit for bit on a Cartesian grid, to round-off along a trajectory.**

  Code: {func}`~bartorch.apps.pics`. Tests: *the app is the tool to the last bit* (`test_apps.py`); *an assembled pics is the tool to the last bit* (`test_solve.py`).
- **Runs a solver as one call into BART, however many iterations it takes.**

  Code: {mod}`bartorch.optim`. Tests: *the solve is one call into the library*; *a BART operator is handed over as itself* (`test_solve.py`).
- **Keeps a CUDA tensor on its card, ordered against PyTorch's stream without synchronising.**

  Code: `src/csrc/abi/cuda.c`, `_ON_DEVICE` in `_dispatch.py`. Tests: *work is ordered against torch's stream without synchronising*; *every tool kept on the card answers there and agrees with the host*; *a tool that is not kept on the card still answers on it* (`test_cuda.py`).
- **Sets one thread count for BART, its FFT and FINUFFT, on the OpenMP runtime PyTorch has already loaded.**

  Code: {func}`~bartorch.set_num_threads`, `cmake/openmp.cmake`. Tests: *one thread count covers BART and the transform* (`test_finufft.py`); *threaded torch and threaded FINUFFT share one OpenMP runtime*; *loading modifies no installed file* (`test_openmp.py`).
- **Reads a `bart` command line, so a BART script runs with the name changed.**

  Code: {mod}`bartorch.cli`. Tests: *the app route answers what the command answers*; *an argument the reader cannot express goes to the command* (`test_cli.py`).

:::

## Division of responsibility

- **BART**: the commands, operators and iterations, compiled from the pinned
  revision of `pulserver/bart`.
- **bartorch**: the tensor boundary, error handling, device placement, and the
  substitutions, which change a result only within their own accuracy: FINUFFT
  for every non-uniform transform ({doc}`non-cartesian`), and the FFT, BLAS and
  LAPACK from MKL, PyTorch's library or SciPy.

## Method

```{image} ../_static/architecture.svg
:class: only-light
:alt: PyTorch tensors enter the tools, apps and operators, which call the bartorch C ABI through ctypes; the ABI runs the embedded BART, whose non-uniform transforms and linear algebra come from substituted backends
:width: 100%
```

```{image} ../_static/architecture-dark.svg
:class: only-dark
:alt: PyTorch tensors enter the tools, apps and operators, which call the bartorch C ABI through ctypes; the ABI runs the embedded BART, whose non-uniform transforms and linear algebra come from substituted backends
:width: 100%
```

### Tensor transfer

A tensor crosses as its data pointer and its shape reversed into a BART
dimension vector ({doc}`data-layout`). Arrays BART allocates come from
PyTorch, on the device of the call. One Python thread calls into BART at a
time; the parallelism is inside the call, on OpenMP threads.

### GPU execution

Operators and solvers work on device memory directly, with cuFFT, cuBLAS and
cuFINUFFT. A few commands allocate host temporaries and mix them with their
inputs; those are given host copies, and their results come back on the card.

(openmp-runtime)=
### OpenMP runtime

BART and FINUFFT link the OpenMP runtime PyTorch ships, and the wheels carry
none of their own, because LLVM's runtime ends the process when a second copy
starts. On Linux this needs PyTorch 2.7.1 or later.

## Examples

- {doc}`../auto_examples/01-basics/01-tensors-and-commands`: BART commands on
  tensors.
- {doc}`../auto_examples/03-regularization/02-operators-and-solvers`: `pics`
  assembled from an operator and a solver.
- {doc}`../api/functions`: threads, input copies and the CUDA settings.
