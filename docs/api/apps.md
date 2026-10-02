# Reconstruction pipelines

`bartorch.apps` contains selected BART reconstruction applications
re-expressed with this package's operators and solvers.  An app takes tensors
and Python arguments, performs the application's preprocessing — sampling
pattern, modulation into BART's uncentred convention, data scaling — in Python,
and then runs an encoding from {mod}`bartorch.linop` under a solver from
{mod}`bartorch.optim` or {mod}`bartorch.nlop`.  The `bartorch` command line
runs an app in place of the BART command where one exists and expresses
every option given.  The BART commands themselves are not public; the table
states how each app relates to its command.
{doc}`../explanation/execution-model` compares apps with the composable
interface they are assembled from.

```{eval-rst}
.. currentmodule:: bartorch.apps
```

| Object | BART command | Relation to the command |
| --- | --- | --- |
| {obj}`~bartorch.apps.pics` | `pics` | Identical output on a Cartesian grid; agrees to floating-point round-off along a trajectory |
| {obj}`~bartorch.apps.nlinv_pics` | `nlinv`, `pics`, `homodyne` | {obj}`~bartorch.apps.nlinv_maps`, then a wavelet `pics` solve, then, for Cartesian k-space only, {obj}`~bartorch.apps.partial_fourier`; Cartesian or along a trajectory |
| {obj}`~bartorch.apps.nlinv_maps` | `nlinv` | One set of coil sensitivity maps from `nlinv -m 1` on the low-resolution centre of k-space, normalized to unit root sum of squares; Cartesian or along a trajectory |
| {obj}`~bartorch.apps.partial_fourier` | `homodyne` | `homodyne -I -C` along each axis acquired on one side of k-space only, the side and the acquired fraction read from the sampling mask |
| {obj}`~bartorch.apps.pocsense` | none | The POCSENSE projections swept by BART's `pocs` iteration |
| {obj}`~bartorch.apps.mobafit` | `mobafit` | Same Gauss-Newton method over a TorchSim model; returns named parameter maps in physical units |
| {obj}`~bartorch.apps.moba` | `moba` | Same Gauss-Newton method over a TorchSim model inside the encoding, with the coils known or estimated jointly under Sobolev weighting; returns named parameter maps in physical units, not held to the command's output |
