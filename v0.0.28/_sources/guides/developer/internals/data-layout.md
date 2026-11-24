# Array layouts and file conventions

The layouts behind {doc}`../../../explanation/data-layout`. The code is
`src/bartorch/_dispatch.py` (commands), `src/bartorch/_layout.py` (operators),
`axes_flags` in `src/bartorch/_operator.py` (axis arguments) and
`src/bartorch/io/` (files); the tests are `tests/test_core.py`,
`tests/test_flags.py`, `tests/test_io.py` and `tests/test_cfl.py`.

## Commands

A command places data along BART's dimensions, reversed. A coil axis is
therefore always the fourth from last and a set of sensitivity maps the fifth
from last, and these axes stay as singletons where BART expects them:
{func}`bartorch.tools.phantom` returns coil k-space of shape `(coils, 1, y, x)`
for a two-dimensional acquisition. A returned shape is the one BART produces,
with its trailing singleton dimensions, the leading tensor axes, removed.

| Data | Tensor shape | BART dimensions |
| --- | --- | --- |
| Cartesian coil k-space | `(coils, z, y, x)` | `[x, y, z, coils]` |
| Radial trajectory | `(spokes, samples, 3)` | `[3, samples, spokes]` |
| Radial coil samples | `(coils, spokes, samples, 1)` | `[1, samples, spokes, coils]` |

A radial acquisition places the readout samples on BART's dimension 1 and the
spokes on dimension 2; dimension 0 of the samples is a singleton matching the
three trajectory components.

Array inputs are converted to contiguous `complex64`. A command works on a copy
of each input unless {func}`bartorch.set_copy_inputs` is `False`, because some
BART commands write into their inputs.

## Axis arguments

An axis argument is converted to BART's bitmask or dimension number at the
boundary. An axis argument whose array is not an argument of the call counts
from the last axis and accepts negative indices only, because the number of
leading axes is not known. A set of indices that are not axes, such as coil
channels or parameter maps, is a tuple and is not reversed.

## Operators

The MRI operators of {mod}`bartorch.linop` use a compact layout: batch axes
first, then coils, then encoding axes, then the axes a transform reads.

| Array | Shape |
| --- | --- |
| Image | `(*batches, [sets,] *encoding, [z,] y, x)` |
| Coil sensitivity maps | `([sets,] coils, [z,] y, x)` |
| Trajectory | `(*encoding, shots, samples, ndim)` |
| Non-Cartesian samples | `(*batches, coils, *encoding, shots, samples)` |
| Cartesian samples | `(*batches, coils, *encoding, [z,] y, x)` |
| Samples of a table of phase encodes | `(*batches, coils, [frames,] shots, readout)` |
| Table of sampled phase encodes | `([frames,] shots, d)` |
| NUFFT and FFT samples | `(*batches, *encoding, shots, samples)` |

A batch axis indexes independent volumes sharing the trajectory or the
sampling pattern, such as slices or averages; each item is encoded separately.
An encoding axis is one the trajectory has in front of its shots, such as time
frames, echoes or cardiac phases. A two-dimensional problem has no `z` axis. A
NUFFT or FFT without sensitivities treats coils as a batch axis. A subspace
basis `(coeffs, frames)` contracts the last encoding axis, so the image carries
coefficients where the samples carry frames. A sampling pattern broadcasts over
the samples of one coil: a pattern of shape `(y, 1)` selects phase encodes for
every coil, readout sample and batch item.

No single BART dimension vector describes such an array, because BART's roles
put coils before any encoding axis while the tensor has them after the batches.
One block does: an operator is built for one batch item, each of its axes placed
on a BART dimension in memory order, and `bartorch_linop_blocks` applies it
block after block. A block is contiguous in both layouts, so nothing is copied.
{doc}`encoding` states the cost of each kind of axis.

## Arrays from MATLAB and CFL files

An array stored column-major in BART's axis order, as MATLAB writes k-space of
shape `(x, y, coils)`, is a tensor after a transpose, with the singleton `z`
restored for a command:

```python
kspace = torch.from_numpy(ksp.T.copy()).to(torch.complex64)[:, None]  # (coils, 1, y, x)
```

{func}`bartorch.io.readcfl` returns a NumPy array whose shape is BART's
dimension vector in Fortran order, trailing singletons dropped, and
{func}`bartorch.io.writecfl` takes one. `array.T` converts between the two
orders without moving data:

```python
tensor = torch.from_numpy(readcfl("kspace").T)       # kspace.hdr, kspace.cfl
writecfl("result", tensor.detach().cpu().numpy().T)
```

## Image geometry

The affine $A$ maps a voxel index to a position,
$(p, 1)^T = A\,(i_x, i_y, i_z, 1)^T$, with $i_x$, $i_y$, $i_z$ the last,
second-to-last and third-to-last tensor axes. Positions are millimetres in RAS,
increasing towards the subject's right, anterior and superior. The columns of
$A$ are the steps between neighbouring voxels, so their norms are the voxel
sizes, and its last column is the position of the first voxel.

MRD and DICOM use LPS, which differs from RAS in the sign of the first two
coordinates (`src/bartorch/io/_geometry.py`). In MRD, $x$ runs along an
acquisition's `read_dir`, $y$ along `phase_dir`, and `position` is the centre of
the field of view; with more than one slice, $z$ steps by the mean distance
between consecutive slices, and a single slice takes the slab thickness. In
DICOM, `ImageOrientationPatient` gives the directions of increasing column and
row index and `ImagePositionPatient` the centre of the first pixel.
