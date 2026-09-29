# Preparing data

Raw data exported from a scanner, simulated in another package or written by
BART's command line reaches bartorch as arrays whose axis order, trajectory
units and file format follow someone else's convention.  This page gives the
conversions into bartorch's; {doc}`../../explanation/data-layout` states those
conventions and the reasons for them.

## Arranging k-space as a tensor

A tensor is in C order and lists BART's dimensions in reverse, so the readout
is the last axis, the phase-encoding directions precede it, and the receive
coils are the fourth axis from the end.  Multichannel Cartesian k-space for a
BART command is therefore `(coils, z, y, x)`, with `z` a singleton for a 2D
acquisition:

```python
import torch

# ksp: complex NumPy array of shape (x, y, coils), as loaded from a MATLAB file
kspace = torch.from_numpy(ksp.T.copy()).to(torch.complex64)[:, None]  # (coils, 1, y, x)
```

The MRI operators of {mod}`bartorch.linop` take the same k-space without the
singleton, `(coils, y, x)`, and put batch axes such as slices or averages in
front of the coils.  An axis argument is a tensor axis index:
`bartorch.fft(kspace, axes=(-2, -1))` transforms `y` and `x`.

## Converting a trajectory to grid units

bartorch's trajectories are in grid units: multiples of $1/\mathrm{FOV}$, so
that a readout of $N$ samples at the Nyquist rate spans $-N/2$ to $N/2$.  A
trajectory in another convention is rescaled before it is passed on:

| Trajectory given in | Grid units |
| --- | --- |
| Cycles per metre, $k$ | $k \cdot \mathrm{FOV}$, with $\mathrm{FOV}$ in metres |
| Radians per metre, $k$ | $k \cdot \mathrm{FOV} / (2\pi)$ |
| Fraction of the sampling bandwidth, $k \in [-0.5, 0.5)$ | $k \cdot N$ |
| Radians, $k \in [-\pi, \pi)$, as in `torchkbnufft` | $k \cdot N / (2\pi)$ |

$N$ is the matrix size along the axis.  The components are the last axis of
the tensor: `(spokes, samples, 3)` for a BART command, whose $k_z$ is zero
throughout for a 2D trajectory, and `(shots, samples, 2)` or
`(shots, samples, 3)` for {class}`bartorch.linop.NUFFT` and
{class}`bartorch.linop.NoncartesianSense`.  {func}`bartorch.tools.traj`
generates trajectories in grid units directly.

## Reading and writing CFL files

{func}`bartorch.io.readcfl` and {func}`bartorch.io.writecfl` exchange NumPy
arrays in BART's dimension order, so the axes are reversed at that boundary:

```python
import numpy as np
import torch
from bartorch.io import readcfl, writecfl

tensor = torch.from_numpy(np.ascontiguousarray(readcfl("kspace").T))  # kspace.hdr, kspace.cfl
writecfl("result", tensor.detach().cpu().numpy().T)
```

The `bartorch` command line reads and writes CFL files itself, so a script
written for BART's `bart` executable runs unchanged with the command name
replaced ({doc}`installation`).

## Checking a Fourier convention

{func}`bartorch.fft` is centred and unnormalized in both directions, which
differs from NumPy's inverse transform by a factor of $N$.  When results are
compared with another package, state the centring and the normalization of
each; the table in {doc}`../../explanation/data-layout` lists them for every
transform in bartorch.
