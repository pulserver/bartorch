# Reconstruction pipelines

`bartorch.apps` contains selected BART reconstruction applications
re-expressed with this package's operators and solvers.  An app takes tensors
and Python arguments, performs the application's preprocessing — sampling
pattern, modulation into BART's uncentred convention, data scaling — in Python,
and then runs an encoding from {mod}`bartorch.linop` under a solver from
{mod}`bartorch.optim` or {mod}`bartorch.nlop`.  The `bartorch` command line
runs an app in place of the BART command where one exists.
{doc}`../explanation/execution-model` compares apps with the corresponding
{mod}`bartorch.tools` commands.

```{eval-rst}
.. currentmodule:: bartorch.apps
```

| Object | BART command | Relation to the command |
| --- | --- | --- |
| {obj}`~bartorch.apps.pics` | `pics` | Identical output on a Cartesian grid; agrees to floating-point round-off along a trajectory |
| {obj}`~bartorch.apps.pocsense` | none | The POCSENSE projections swept by BART's `pocs` iteration |
| {obj}`~bartorch.apps.mobafit` | `mobafit` | Same Gauss-Newton method over a TorchSim model; returns named parameter maps in physical units |
| {obj}`~bartorch.apps.moba` | `moba` | Same Gauss-Newton method over a TorchSim model inside the encoding, with the coils known or estimated jointly under Sobolev weighting; returns named parameter maps in physical units, not held to the command's output |
