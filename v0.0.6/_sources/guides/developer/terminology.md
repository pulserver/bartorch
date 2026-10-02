# Terminology and conventions

The vocabulary used in code, docstrings and documentation.  Definitions and
derivations are in {doc}`../../explanation/index`; this page fixes the terms.
The readers are MR scientists, so an established MR term is used rather than a
description of what it denotes: *coil sensitivity maps*, not "the coils'
weights"; *autocalibration region*, not "the fully sampled middle".

## Acquisition and reconstruction

| Term | Meaning |
| --- | --- |
| Readout, frequency-encoding direction | The k-space direction sampled during one ADC window; the last tensor axis of Cartesian k-space. |
| Phase-encoding direction | A k-space direction stepped between readouts; `y`, and `z` in 3D. |
| Acceleration factor $R$ | The ratio of the phase encodes of a fully sampled acquisition to those acquired. |
| Autocalibration (ACS) region | The fully sampled block at the centre of k-space from which coil sensitivity maps are estimated. |
| Coil sensitivity maps | The complex receive sensitivities $S_c(r)$ of the channels of an array. |
| SENSE | The encoding model $A = PFS$ and its least-squares or regularized inversion. |
| g-factor | The voxel-wise noise amplification of a parallel-imaging reconstruction beyond the $\sqrt{R}$ loss of the shorter acquisition. |
| Density compensation | Weighting of non-Cartesian samples by the inverse of the local sampling density. |
| Off-resonance, $B_0$ inhomogeneity | A deviation of the precession frequency from the reference frequency, in hertz; its map is the field map. |
| Signal model | The dependence of the signal of a voxel on tissue parameters such as $T_1$ and $T_2$ and on the sequence. |

## Arrays

| Term | Meaning |
| --- | --- |
| C-order shape | The shape of a PyTorch tensor.  The last axis varies fastest. |
| BART dimension vector | BART's `dims[]`, the reverse of the C-order shape of the same memory.  BART's dimension 0 is the tensor's last axis. |
| Axis index | An index into a C-order shape, negative indices counting from the end.  Public arguments take axis indices. |
| BART bitmask | BART's set of dimensions as bits of an integer.  No public argument takes one; axis indices are converted at the boundary. |
| Batch axis | An axis of independent items that share the operator's trajectory or pattern, each transformed separately. |
| Encoding axis | An axis the trajectory or sampling pattern indexes, such as frames or echoes; its samples belong to one transform. |
| Sets | ESPIRiT's multiple sensitivity maps, an axis the image carries and the samples do not. |

## Sampling

| Term | Meaning |
| --- | --- |
| Trajectory | The k-space coordinates `kx, ky, kz` of every sample, in grid units ($1/\mathrm{FOV}$). |
| Non-uniform FFT (NUFFT) | The discrete Fourier transform between a Cartesian image grid and samples at arbitrary k-space positions, computed approximately to a tolerance; not "gridding", which names one algorithm for it. |
| Sampling pattern | A binary mask on the Cartesian grid, one where a sample was acquired. |
| Density weights | Density-compensation or data weights as a diagonal in k-space applied to the samples of a non-Cartesian transform, on the forward pass and conjugated on the adjoint. |
| Subspace basis | A matrix `(coeffs, frames)` mapping coefficients to the frames of a signal series. |

## Operators

| Term | Meaning |
| --- | --- |
| Forward operator, encoding operator | $A$, mapping the unknown to the data.  "Encoding operator" is the MRI forward operator. |
| Domain, codomain | `ishape` and `oshape`: the spaces an operator maps from and to. |
| Forward, adjoint, normal | $Ax$, $A^H y$ and $A^H A x$. |
| Linear operator | {class}`~bartorch.linop.LinearOperator`, complex-linear unless its page states that it is only real-linear. |
| Nonlinear operator | {class}`~bartorch.nlop.NonlinearOperator`: a map with a derivative $DF_x$ at every point and the derivative's adjoint. |
| Encoding form | The expression every MRI encoding here reduces to: image-side factor, transform, k-space-side factor and contraction; reported by `A.plan`. |
| Lowering | Matching a composition of operators against the encoding form and building it as one encoding. |

## Regularization and solvers

| Term | Meaning |
| --- | --- |
| Functional | A map from the image to a real number. |
| Regularization functional, term | A functional added to the data term; a {mod}`bartorch.priors` object. |
| Transform $G$ | The linear operator in $g(Gx)$; the identity for a term whose transform is inside its proximal operator. |
| Proximal operator | $\operatorname{prox}_{\tau g}(v) = \arg\min_u \tfrac12\lVert u - v\rVert^2 + \tau g(u)$. |
| Solver | A configured algorithm called as `solver(y, A)`, running BART's iteration to its stopping rule. |
| Iteration block | One step of a solver as a `torch.nn.Module`, with `start`, `forward` and `output`. |
| Unrolled network | A fixed number of iteration blocks, whose parameters may be learned ({class}`~bartorch.learning.Unrolled`). |
| Explicit unrolling | Differentiation by recording every step of the iteration. |
| Implicit differentiation | Differentiation through the optimality or fixed-point condition, without recording the iterations. |

## Interfaces and backends

| Term | Meaning |
| --- | --- |
| BART command, BART tool | A program of BART's command line, such as `pics`. |
| `bartorch.tools` function | A BART command called in-process on tensors, or a correction or motion-estimation function with no BART command behind it. |
| App | A {mod}`bartorch.apps` pipeline re-expressing a BART command with operators and solvers. |
| Reference command | A BART reconstruction command answered publicly by an app, kept private in `bartorch._reference` so that the test suite can hold the app to it. |
| CLI | The `bartorch` console command, which accepts `bart` command lines and runs an app or the command. |
| Substitution | A component compiled in BART's place: the FINUFFT and cuFINUFFT NUFFT, the point spread function, the FFT and BLAS/LAPACK routing. |
| Backend | The library that performs a computation for BART: FINUFFT, cuFINUFFT, MKL, the BLAS/LAPACK PyTorch links, SciPy's, cuFFT, cuBLAS. |
| Decline, refusal | A configuration the substitution cannot serve, reported as an error with its reason rather than computed by another method. |

## Writing conventions

Units are stated where a quantity is physical: times in milliseconds for
signal models, frequencies in hertz and readout times in seconds for field
correction, trajectories in grid units.  Fourier-transform centring and
normalization are stated for every transform.  A check or a comparison is
described by what it compared and to what tolerance, not as a general
guarantee.
