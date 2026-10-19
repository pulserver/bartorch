# Encoding operators

```{admonition} TL;DR
:class: tldr

- Cartesian, non-Cartesian and wave SENSE are one expression, $y = \sum_a O \cdot T(I \cdot x_a)$, applied by one executor a slab of coils at a time.
- Subspace, time segmentation, multi-shot phase and SMS are compositions written with `@` and `+`, lowered into that expression when first used.
- `A.plan` reports the form an operator was built into; `plan.fused` is false where a composition fell back to a chain.
```

The encoding operator $A$ maps the image you estimate to the samples the
scanner acquired, and every reconstruction you assemble yourself starts from
one. You build the standard encodings with a constructor, extend them with
diagonals, sums and operators of your own, and hand the result to a solver or
to autograd as one BART operator. Lesson 6 of the
{doc}`Course <../gallery/course>` builds the SENSE operator, checks its
adjoint and extends it with an operator written in Python.

## Capabilities

:::{container} capabilities

- **Builds Cartesian, non-Cartesian and wave SENSE as one encoding form, applied a slab of coils at a time.**

  Code: {func}`~bartorch.linop.CartesianSense`, {class}`~bartorch.linop.NoncartesianSense`, {func}`~bartorch.linop.WaveSense`, `src/csrc/ops/sense.c`. Tests: *every encoding reports its transform*; *the slab loop is what ran the encoding* (`test_plan.py`); *a cartesian reconstruction is the same whatever the slab* (`test_sense.py`).
- **Contracts subspace coefficients with a temporal basis inside the encoding, with a normal kernel that does not grow with the frames.**

  Code: `basis=` of each constructor. Tests: *a subspace encoding is the model written out*; *the collapsed normal does not grow with the frames* (`test_linop_mri.py`); *a fused subspace encoding is the frames written out* (`test_plan.py`).
- **Corrects off-resonance by time segmentation around any encoding.**

  Code: {func}`~bartorch.linop.FieldCorrected`. Tests: *more segments is a better approximation*; *it wraps any encoding*; *a segmented nufft is the sum of its segments* (`test_linop_mri.py`).
- **Lowers a sum of terms written with `@` and `+` into one encoding, applied in one call however many terms it has.**

  Code: `linop/_plan.py`. Tests: *composing builds a description and nothing else*; *a sum written by hand is the contraction the planner lowers*; *a composition costs one application however many terms* (`test_plan.py`).
- **Lowers a phase per shot and a phase per slice (SMS) the same way.**

  Code: `linop/_plan.py`, `src/csrc/ops/sense.c`. Tests: *a shot phase is a contraction over the shots*; *a phase per slice is summed over on the far side*; *an echo phase with a basis is a contraction over the frames* (`test_plan.py`).
- **Reports the plan an operator was built into, including a fallback to BART's chain of operators.**

  Code: {attr}`~bartorch.linop.LinearOperator.plan`. Tests: *a plan reads as a sentence*; *a form the loop cannot take says so*; *a grid contraction is fused and a coil varying one is not* (`test_plan.py`).
- **Composes, adds, scales, slices, stacks and takes adjoints and normals of operators, and the result stays one BART operator with a correct adjoint.**

  Code: {class}`~bartorch.linop.LinearOperator`, {func}`~bartorch.linop.concatenate`, {func}`~bartorch.linop.hstack`, {func}`~bartorch.linop.block_diag`, {func}`~bartorch.linop.block`. Tests: *every combination stays one bart operator*; *the algebra keeps the adjoint honest* (`test_linop.py`); *each stays one bart operator* (`test_linop_combine.py`).
- **Takes an operator written in Python, by its forward and adjoint, into a BART composition and a BART solver.**

  Code: {class}`~bartorch.linop.LinearOperator`. Tests: *an operator written here chains with one of barts*; *an operator written here is solved by barts conjugate gradients* (`test_linop.py`).
- **Applies the normal operator through a kernel where one exists, and as the forward followed by the adjoint otherwise.**

  Code: {meth}`~bartorch.linop.LinearOperator.gram`, `toeplitz=`. Tests: *a pattern collapses into a kernel*; *a fused normal is the kernel written out*; *without toeplitz the normal is the two applications* (`test_plan.py`).

:::

## Division of responsibility

- **BART**: the operators the encodings are made of, its chains and sums of
  operators, the FFT on a device (cuFFT), and the solvers that iterate on the result.
- **bartorch**: the encoding executor, the planner that lowers a composition
  into it, and the callbacks that carry a Python operator into BART.
- **Other libraries**: FINUFFT for the NUFFT ({doc}`non-cartesian`), and
  mri-nufft for the time-segmentation coefficients
  {func}`~bartorch.linop.FieldCorrected` fits from a field map.

## Method

```{image} ../_static/encoding.svg
:class: only-light
:alt: The image or its coefficients x[a] is multiplied by the image-side factor I, transformed by T, multiplied by the k-space-side factor O, and summed over a to give the samples y
:width: 100%
```

```{image} ../_static/encoding-dark.svg
:class: only-dark
:alt: The image or its coefficients x[a] is multiplied by the image-side factor I, transformed by T, multiplied by the k-space-side factor O, and summed over a to give the samples y
:width: 100%
```

### Encoding form

For coil $c$, frame $t$ and sample $k$,

$$
y[c, t, k] = \sum_a O[a, t](k)\; T_t\!\left( I[c, a, t](r)\, x[a](r) \right)\!(k).
$$

$I$ holds the coil sensitivities and any spatial weight; $T$ is the FFT, the
NUFFT or the wave transform;[^wave] $O$ holds the sampling pattern, density
weights, a subspace basis[^tamir] or a sample weight; the sum over $a$ runs over
subspace coefficients, time segments[^sutton] or slices. With $I = S$, $T = F$,
$O = P$ and one term this is SENSE, $A = PFS$.[^pruessmann] Each application
is one call into the library, and the coil images are never held for all coils
at once.

### Operator algebra

`A @ B` applies `B` first; `A + B` and `A - B` need equal shapes; `c * A`
scales by a number, and `*` between two operators is refused. `A.H` is the
adjoint, `A.T` the adjoint without conjugation, `A.gram()` is $A^H A$ and
`A.cogram()` is $A A^H$. `A[0]` restricts the output as tensor indexing does.
{func}`~bartorch.linop.concatenate` (vertical stacking) applies every operator
to one input, {func}`~bartorch.linop.hstack` splits the input and adds the
outputs, and {func}`~bartorch.linop.block_diag` gives each operator its own
part of both. An operator of your own defines `forward` and `adjoint`, and each
application crosses into Python once. Check any adjoint with the dot-product
test, $\langle A x, y \rangle = \langle x, A^H y \rangle$ for random $x$, $y$:

```python
lhs = torch.vdot(A(x).flatten(), y.flatten())
rhs = torch.vdot(x.flatten(), A.H(y).flatten())
```

A product of diagonals around one encoding, or a sum of such products sharing
the encoding, is lowered into the form when the operator is first applied,
solved, or asked for `.plan`. Anything else is built as BART's chain of
operators, with the same result at a higher cost; `plan.contraction` is then
`"chained"` or `plan.executor` is `"chain"`. The planner's rules and every
field of the plan are in {doc}`../guides/developer/internals/encoding`.

### Normal operator

A solver applies $A^H A$ once per iteration. On a grid with a pattern it is
$\sum_c \overline{S_c} F^H P F S_c$, with a subspace kernel
$K_{aa'} = \sum_t \overline{\Phi_{at}} \Phi_{a't} P_t$ computed once. Along a
trajectory the transform's own normal is a convolution with a point spread
function ({doc}`non-cartesian`). `plan.normal` names which applies:
`"kernel"`, `"transform"`, or `"applications"`, the forward followed by the
adjoint.

## Examples

- {doc}`../auto_examples/03-regularization/02-operators-and-solvers`: the SENSE
  operator built by hand, its adjoint checked, and an operator written in Python.
- {doc}`../auto_examples/07-tours/05-spiral-deblurring`: off-resonance correction
  of a spiral scan with {func}`~bartorch.linop.FieldCorrected`.
- {doc}`../auto_examples/08-workflows/02-subspace-t1-mapping`: a subspace
  encoding along a radial trajectory, and its plan.
- {doc}`../api/linop`: every operator and combinator.

## References

[^pruessmann]: Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: sensitivity encoding for fast MRI. *Magn Reson Med* 42(5):952–962 (1999). [doi:10.1002/(SICI)1522-2594(199911)42:5\<952::AID-MRM16\>3.0.CO;2-S](https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S)

[^wave]: Bilgic B, Gagoski BA, Cauley SF, Fan AP, Polimeni JR, Grant PE, Wald LL, Setsompop K. Wave-CAIPI for highly accelerated 3D imaging. *Magn Reson Med* 73(6):2152–2162 (2015). [doi:10.1002/mrm.25347](https://doi.org/10.1002/mrm.25347)

[^tamir]: Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. T2 shuffling: sharp, multicontrast, volumetric fast spin-echo imaging. *Magn Reson Med* 77(1):180–195 (2017). [doi:10.1002/mrm.26102](https://doi.org/10.1002/mrm.26102)

[^sutton]: Sutton BP, Noll DC, Fessler JA. Fast, iterative image reconstruction for MRI in the presence of field inhomogeneities. *IEEE Trans Med Imaging* 22(2):178–188 (2003). [doi:10.1109/TMI.2002.808360](https://doi.org/10.1109/TMI.2002.808360)
