# Command line

`bartorch.cli` implements the `bartorch` console command, which accepts the
command lines of BART's `bart` executable and operates on CFL files:

```sh
bartorch pics -l1 -r0.01 -i30 kspace sensitivities image
bartorch ecalib -m1 kspace maps
bartorch --list
```

A command for which {mod}`bartorch.apps` has a pipeline -- `pics`, `mobafit`
and `moba` -- is parsed into a call of that app; every other command runs as
the BART command, in the same process, through the entry point
{mod}`bartorch.tools` uses.  The options are read from BART's own command
declarations, and an option the app does not express sends the command line to
the BART command.

Either route writes the output files of the BART command, in its array layout
and units.  A `pics` command line routed to the app writes the same values
bit for bit.  `mobafit` and `moba` are routed for the models whose parameters
the TorchSim model of the app represents exactly -- `-T`, `-I`, `-L`, `-D`,
`-M` with `--init`, and `-G` with `-m 0`, `1`, `3` or `4` for `mobafit`, and
`-T` and `-L` with `-l2` on a Cartesian grid for `moba` -- and the fitted maps
are converted to BART's coefficients: relaxation rates in 1/s for times in
seconds, off-resonance in Hz, water and fat as complex amplitudes, stacked
along `COEFF_DIM`.  Their values agree with the
BART command to the accuracy of the fit, since the two minimize the same
objective in different parameterizations.  `-i`, which counts Gauss-Newton
steps in BART's parameterization, sends either command to BART.
`bartorch <command> --help` prints BART's help text for the command.

```{eval-rst}
.. currentmodule:: bartorch.cli
```

| Object | Description |
| --- | --- |
| {obj}`~bartorch.cli.main` | Run one command line; returns its exit code |
| {obj}`~bartorch.cli.route` | Whether a command line runs as an app or as the BART command |
| {obj}`~bartorch.cli.read` | Read a CFL pair as a tensor in this package's C-order layout |
