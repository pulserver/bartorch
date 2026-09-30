# Data layout and conventions

```{admonition} TL;DR
:class: tldr

- A C-order tensor of shape `(a, b, c)` and a BART array of dimensions `[c, b, a]` occupy the same memory, so bartorch reverses the shape at the boundary and copies nothing: BART's readout dimension is the last tensor axis and its coil dimension the fourth from last.
- BART's commands keep BART's positional dimensions; the MRI operators of {mod}`bartorch.linop` use a compact layout of batch axes, coils and encoding axes in front of the spatial axes.
- Axis arguments are tensor axis indices, never BART bitmasks or dimension numbers.
- Image geometry is a `(4, 4)` affine from voxel indices `(x, y, z)`, the last three tensor axes reversed, to RAS millimetres, as in NIfTI.
- Trajectories are in grid units, multiples of $1/\mathrm{FOV}$, so that a fully sampled readout of $N$ samples spans $-N/2$ to $N/2$.
- {func}`bartorch.fft` is centred and unnormalized in both directions unless asked otherwise; {class}`bartorch.linop.FFT` is centred and unitary; the NUFFT carries a $1/\sqrt{N}$ scaling and a negative exponent in its forward transform.
```

Two reconstructions of the same k-space by two software packages agree only if
they agree on where each axis of the data lies in memory, on the units of the
trajectory, and on the centring and scaling of the Fourier transform.  None of
these is fixed by the physics, and each package chooses differently: BART
stores arrays in column-major order with a fixed meaning for each dimension,
PyTorch and NumPy index in row-major (C) order, and NUFFT libraries measure
k-space in radians, in cycles per metre or in fractions of the sampling
bandwidth.  This page states the conventions bartorch adopts and the reason for
each.  {doc}`../guides/user/conventions` shows how to bring data acquired
elsewhere into them.

## Array order and BART's dimensions

BART represents every array as sixteen dimensions in column-major (Fortran)
order, and assigns each a meaning in `misc/mri.h`: dimension 0 is the readout,
1 and 2 the phase-encoding directions, 3 the receive coils, 4 the sets of
sensitivity maps, 5 the echo times, 6 the subspace coefficients, 10 the time
frames, 13 the slices and 15 a batch, among others.  A PyTorch tensor is
row-major: its last axis varies fastest in memory.

The two orders describe the same memory when the shape is reversed.  A C-order
tensor of shape `(a, b, c)` and a BART array of dimensions `[c, b, a]` are the
same bytes, so bartorch hands BART the tensor's data pointer and its reversed
shape, and no operator or solver copies its input to change the order.  BART's
dimension $d$ is therefore tensor axis $-(d + 1)$: the readout is the last
axis, the coils the fourth from last.

| Data | Tensor shape | BART dimensions |
| --- | --- | --- |
| Cartesian coil k-space | `(coils, z, y, x)` | `[x, y, z, coils]` |
| Radial trajectory | `(spokes, samples, 3)` | `[3, samples, spokes]` |
| Radial coil samples | `(coils, spokes, samples, 1)` | `[1, samples, spokes, coils]` |

A radial acquisition places the readout samples along BART's dimension 1 and
the spokes along dimension 2, with dimension 0 of the samples a singleton that
corresponds to the three trajectory components.

## Axis arguments

Where BART takes a bitmask of dimensions or a dimension number, bartorch takes
indices into the tensor's shape, negative indices included:
`bartorch.fft(x, axes=(-2, -1))` transforms the two in-plane axes.  No public
argument takes a BART bitmask, and a set of indices that are not axes, such as
coil channels or parameter maps, is also given as a tuple.  Regularization is
given as {mod}`bartorch.priors` terms rather than as BART's `-R` strings:
`apps.pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))`.  An
axis argument whose array is not an argument of the call counts from the last
axis and accepts negative indices only, because the number of leading axes is
not known.

## Commands

The functions of {mod}`bartorch.tools` and of the `bartorch` namespace that run
a BART command place data along BART's dimensions, reversed.  A coil axis is
therefore always the fourth from last and a set of sensitivity maps the fifth
from last, and these axes are kept as singletons in inputs and outputs where
BART expects them: {func}`bartorch.tools.phantom` returns coil k-space of shape
`(coils, 1, y, x)` for a two-dimensional acquisition.  Returned shapes are
those BART produces, with BART's trailing singleton dimensions, the leading
tensor axes, removed.

Array inputs are converted to contiguous `complex64`.  A command works on a
copy of each input unless {func}`bartorch.set_copy_inputs` is set to `False`,
because some BART commands write into their inputs.  The corrections and the
motion estimation of {mod}`bartorch.tools` run no BART command; their shapes
and units are stated in each object's documentation.

## Operators

A positional layout of sixteen dimensions fits BART's command-line tools,
which exchange files, but forces a tensor to carry singleton axes for every
dimension the problem does not use.  The MRI operators of {mod}`bartorch.linop`
use a compact layout instead: batch axes first, then coils, then the encoding
axes, then the spatial axes.

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

A **batch axis** indexes independent volumes that share the trajectory or the
sampling pattern, such as slices or averages; each item is encoded separately.
An **encoding axis** is an axis the trajectory has in front of its shots, such
as time frames, echoes or cardiac phases.  A two-dimensional problem has no `z`
axis.  A NUFFT or FFT without sensitivities treats coils as a batch axis.  A
subspace basis of shape `(coeffs, frames)` contracts the last encoding axis, so
the image carries coefficients where the samples carry frames.  A sampling
pattern broadcasts over the samples of one coil: a pattern of shape `(y, 1)`
selects phase encodes for every coil, readout sample and batch item.
{doc}`encoding` states the cost of each kind of axis.

## Trajectories

A trajectory holds the k-space coordinates $k_x, k_y, k_z$ in **grid units**,
multiples of $1/\mathrm{FOV}$ of the image being encoded.  In these units the
Nyquist sampling interval is one, and a fully sampled readout of $N$ samples
spans $-N/2$ to $N/2$.  A coordinate $k$ in cycles per metre is $k \cdot \mathrm{FOV}$ in
grid units.

A trajectory given to a command has three components, and a $k_z$ that is zero
throughout makes the transform two-dimensional; the operators of
{mod}`bartorch.linop` also accept two components, $k_x, k_y$.
{func}`bartorch.tools.traj` generates trajectories in grid units.

## Image geometry

Where an image lies in the scanner is a `(4, 4)` affine matrix $A$ mapping a
voxel index to a position, $(p, 1)^T = A\,(i_x, i_y, i_z, 1)^T$, with
$i_x$, $i_y$, $i_z$ the last, second-to-last and third-to-last axes of the
image tensor.  Positions are in millimetres in RAS coordinates, which increase
towards the subject's right, anterior and superior; this is the convention of
NIfTI and of `nibabel` and `torchio`.  The columns of $A$ are the steps
between neighbouring voxels, so their norms are the voxel sizes, and its last
column is the position of the first voxel.

MRD and DICOM state positions in the patient coordinate system, LPS, which
differs from RAS in the sign of the first two coordinates.  In MRD the readout
direction is $x$, the phase-encoding direction $y$, and an acquisition's
`position` is the centre of the field of view; in DICOM
`ImageOrientationPatient` gives the directions of increasing column and row
index and `ImagePositionPatient` the centre of the first pixel.  The readers
and writers of {mod}`bartorch.io` convert between these and the affine, so an
image read from DICOM and written to NIfTI keeps its position.

## Fourier transform conventions

| Function or operator | Centring | Normalization |
| --- | --- | --- |
| {func}`bartorch.fft`, {func}`bartorch.ifft` | Centred unless `uncentred=True` | Unnormalized in both directions unless `unitary=True` |
| {class}`bartorch.linop.FFT` | Centred unless `centred=False` | Unitary |
| {func}`bartorch.nufft`, {class}`bartorch.linop.NUFFT` | Zero frequency at the trajectory origin | $1/\sqrt{N}$ for $N$ image voxels; negative exponent in the forward transform |

A centred transform places the zero frequency, and the image centre, at index
$\lfloor n/2 \rfloor$ of an axis of length $n$: {func}`bartorch.fft` equals
`numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.ifftshift(x)))` over the
transformed axes.  Because the inverse transform is unnormalized as well,
`bartorch.ifft(bartorch.fft(x, axes), axes)` is $N x$ for $N$ transformed
samples, where NumPy's pair returns $x$.  A comparison of reconstructions from
different software states the centring and the normalization of each.

## CFL files

BART's file format is a pair: a `.hdr` text header listing the dimensions and a
`.cfl` file of `complex64` values in column-major order.
{func}`bartorch.io.readcfl` and {func}`bartorch.io.writecfl` exchange NumPy
arrays whose shape is BART's dimension vector, so the axes are reversed at that
boundary; `array.T` converts between the two orders without moving data.

## Differentiation

Applying an operator, a nonlinear operator or a solver of {mod}`bartorch.optim`
to a tensor that requires a gradient records the operation for autograd; the
BART commands of {mod}`bartorch.tools` and of the `bartorch` namespace record
nothing.  {doc}`differentiation` describes the backward pass of each.
