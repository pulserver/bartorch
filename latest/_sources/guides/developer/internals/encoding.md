# Encoding form and lowering

The rules behind {doc}`../../../explanation/encoding`: what the planner lowers,
what the plan reports, how the normal operator is applied, and how encoding
and batch axes are told apart.  The code is `src/bartorch/linop/_form.py` (the
form and the plan), `src/bartorch/linop/_plan.py` (matching and lowering) and
`src/csrc/ops/sense.c` (the executor); the tests are `tests/test_plan.py`,
`tests/test_sense.py` and `tests/test_linop_mri.py`.

## Sensitivity encoding

A receive coil measures the transverse magnetization weighted by its
sensitivity and integrated against the phase of the gradients.  With $x(r)$
the object, $S_c(r)$ the sensitivity of coil $c$ and $k_j$ the k-space position
of sample $j$,

$$
y_{c,j} = \int S_c(r)\, x(r)\, e^{-2\pi i\, k_j \cdot r}\, \mathrm{d}r ,
$$

which on a voxel grid is $A = PFS$: $S$ maps the image to one coil image per
channel, $F$ is the Fourier transform of each, and $P$ selects the acquired
samples.  Undersampling a phase-encoding direction by $R$ folds $R$ voxels
$\mathrm{FOV}/R$ apart onto each other; with $C \ge R$ coils whose
sensitivities at those voxels are linearly independent the fold is
invertible, and the SNR of the unfolded image is that of the fully sampled one
divided by $g\sqrt{R}$, where the g-factor $g \ge 1$ depends on the coil
geometry voxel by voxel.  {func}`~bartorch.tools.ecalib` estimates the
sensitivities by ESPIRiT from the ACS region, {func}`~bartorch.tools.nlinv`
jointly with the image, and {func}`~bartorch.tools.ncalib` from non-Cartesian
samples.

## Factors of the form

| Symbol | Name | Instances |
| --- | --- | --- |
| $I$ | Image-side factor | Coil sensitivities; spatial weights of the terms of a contraction, such as the field-map phase of a time segment |
| $T$ | Transform | FFT on the image grid; NUFFT along a trajectory; wave encoding, an FFT with a point spread function between the readout and phase-encode transforms |
| $O$ | K-space-side factor | Sampling pattern or table of sampled phase encodes; density weights; subspace basis; sample weights of the terms of a contraction; a slice phase |
| $\sum_a$ | Contraction | Subspace coefficients; time segments; shots; slices of a simultaneous multislice acquisition |

## Lowering

Composing with `@` and `+` records a description and builds nothing
(`_Composition`); the shapes are worked out from the operands.  The first
operation that needs the handle -- an application, `.H`, a solve, or reading
`.plan` -- offers the whole description to `_plan.describe` and `_plan.lower`.

- A product of {class}`~bartorch.linop.Diagonal` operators around one encoding
  is a `Chain`; a sum of chains is a `Sum`.
- A sum whose terms share one encoding is a contraction, and is lowered into
  the form's own sum over terms (`contraction="segments"`).  Terms on two
  different encodings are not a contraction, and stay a sum.
- Along a trajectory, a contraction whose k-space weights vary along the
  samples is lowered into a subspace over the samples
  (`contraction="subspace"`), which has a normal kernel per pair of terms.
- A term that picks one set of sensitivities whole, with a phase per set in
  k-space, is a slice (`contraction="slices"`): the coil images keep the sets,
  the transform runs once per slice, and the slices are added after the
  k-space factor.
- An image weight that differs between sets of sensitivities is applied to
  each set before the sensitivities sum them, inside the coil loop.
- Weights that vary along the coils cannot be cut into coil slabs, so such a
  sum is built as BART's sum of chains (`contraction="chained"`).
- Anything that is not element-wise on either side of one of the three
  transforms has no description and is built as it stands.  An operator around
  the encoding that is not a diagonal is chained to the lowered encoding, and
  the plan reported is the encoding's.

## Plan fields

`A.plan` is read back from the library after the operator is built, so it
reports what ran.

| Field | Reports |
| --- | --- |
| `transform` | `"fft"`, `"nufft"`, `"wave"` or `"none"` |
| `image`, `kspace` | The element-wise factors on each side of the transform |
| `contraction`, `terms` | `"subspace"`, `"segments"`, `"slices"`, `"chained"` or `None`, and the number of terms |
| `normal` | `"kernel"`, `"transform"` or `"applications"` |
| `coil_batch`, `streamed` | Coils per slab (0 is every coil at once, BART's own chain), and what is walked a slab at a time |
| `executor` | `"slab"` where the executor took the form, `"chain"` where BART's chain of operators runs instead |
| `fused` | `executor == "slab"` and `contraction != "chained"` |
| `cartesian` | Trajectory axes found on the image grid and transformed by an FFT, the NUFFT then over the others: `("z",)` for a stack of stars |
| `items` | Items along encoding axes the trajectory varies along and the image varies along too, each with its own transform and normal kernel |

## Normal operator

`plan.normal` is one of three.

- **`"kernel"`, on a grid.**  With a pattern $P$,
  $A^H A = \sum_c \overline{S_c} F^H P F S_c$.  With a subspace basis $\Phi$
  the pattern becomes a kernel over pairs of coefficients,
  $K_{aa'}(k) = \sum_t \overline{\Phi_{at}}\, \Phi_{a't}\, P_t(k)$, computed
  once at build time, so the cost does not grow with the frames.  A wave
  pattern that varies along the readout has no such kernel, because the wave
  point spread function sits between the readout and phase-encode transforms.
- **`"kernel"`, along a trajectory.**  The transform's normal
  $Q = \mathrm{NUFFT}^H W^H W\, \mathrm{NUFFT}$ is a convolution with a point
  spread function ({doc}`nufft`).  The full normal
  $\sum_c \overline{S_c}\, Q\, S_c$ is not translation invariant and is applied
  coil by coil.  With a subspace basis the kernel of $Q$ is taken over pairs of
  coefficients.
- **`"transform"`** where there is no k-space factor and the transform's own
  normal is the whole of it, and **`"applications"`** -- the forward followed
  by the adjoint -- for a Cartesian or wave contraction over segments or
  slices, for a wave pattern along the readout, and with `toeplitz=False`.

## Encoding and batch axes

| Axis | Definition | Examples | Cost |
| --- | --- | --- | --- |
| Encoding axis | Indexed by the trajectory or the sampling pattern | Frames, echoes, cardiac phases | Joins the samples of one transform when the image does not vary along it; one transform per item, under the one coil loop, when it does (`plan.items`) |
| Batch axis | Not indexed by the trajectory; items share it and are encoded independently | Slices with their own sensitivities, averages, repetitions | One transform plan for all items |

Where a basis contracts an encoding axis, the image carries coefficients and
every sample the trajectory indexes belongs to one point set.  A batch the
sensitivities vary along is written `(batch, 1, coils, y, x)`, a batch axis in
front of a sets axis, and is a dimension of the operator rather than a block.
Under BART's own commands, the NUFFT substitution refuses an image that varies
along a trajectory axis, with the reason.
