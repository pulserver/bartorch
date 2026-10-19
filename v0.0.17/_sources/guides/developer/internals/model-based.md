# Model-based command routing

`bartorch moba ...` and `bartorch mobafit ...` run as the apps
{func}`~bartorch.apps.moba` and {func}`~bartorch.apps.mobafit` when the
command line reads into an app call exactly, and as BART's command otherwise.
`cli.route` makes the decision after reading the input arrays; the adapters
are `_moba` and `_mobafit` in `src/bartorch/cli/_apps.py`. An adapter raises
`Unsupported` for any argument it does not translate, so an argument is never
ignored: the whole command line goes to the command.

## Agreement with the command

The apps fit BlochSim's models in BlochSim's bounded parameterisation and
reach the minimum the command reaches by a different path, so they agree with
the command to the tolerance of a fit rather than to the bit. The adapters
convert the named maps back into BART's coefficients: rates in 1/s from times
in ms, the same stacking along `COEFF_DIM`, and the data scaling the command
applies (`--scale_data`, `--normalize_scaling`, `--scale_psf`; under `-G` the
coil images scaled to a norm of a hundred). The tolerances are held in
`tests/test_cli.py`.

With the coils estimated, only the product of the amplitude and the coils is
determined by the data; the two routes split it by a smooth factor, so the
tests compare the product.

## Models routed to the app

| Command | Flags | App model |
| --- | --- | --- |
| `mobafit` | `-T`, `-I`, `-L`, `-G` with `-m` 0, 1, 3 or 4, `-D`, `-M` with `--init` | Multi-echo decay, inversion recovery, Look-Locker recovery, water and fat, diffusion decay, Lorentzian lines |
| `moba` | `-T`, `-L` with `-l2` | Multi-echo decay, Look-Locker recovery |
| `moba` | `-P` with `-l2` | Look-Locker model in the flip angle, B1 under a Sobolev weighting |
| `moba` | `-G` with `-m` 0, 1, 3 or 4, without `-r` | Water and fat over the coil images, B0 under a Sobolev weighting |
| `moba` | `-D` with `-m` 6 or 7 | Inversion recovery of a multi-echo gradient-echo train |
| `moba` | `--bloch` with `-l2` | BlochSim FLASH or IR-FLASH simulation in `(R1, M0, R2, B1)` |

## Command lines sent to BART

| Argument | Reason |
| --- | --- |
| `-i` | Counts Gauss-Newton steps over BART's own scaled coefficients; a bounded parameterisation needs more, so the count does not translate into the app's `iterations` |
| `moba -C` | Counts FISTA iterations, or under `-G` conjugate-gradient steps stopped at a tolerance the app's are not |
| `moba` without `-l2` (`-l1`, the default) | A wavelet term on the maps, which the app does not carry |
| `-G -m 2` | Separate water and fat decays, which the bounded fit does not separate |
| Any other `-m` | A model on another layout, which the app does not have |
| `moba -G -r` | The proximal iteration `-r` puts around the gradient-echo fit |
| `mobafit -P`, `--fB0-init`, `-M` without `--init`, a start the bounded model has no image of, `--scale`, a covariance output | No app counterpart, no neutral start, a value outside the bounds, a preconditioning, an output the app does not compute |
| `--bloch` balanced SSFP, spin or spoke averaging, a slice profile, a held R1 or M0, a B1 map beside a fitted B1 | A train or a held parameter the app does not play; a B1 map beside a held B1 is taken |
| A trajectory (`-t`) | The adapters read Cartesian k-space only |
| An input the reader cannot load, or an array with more dimensions than BART has | The command reports it |

A missing input file is reported before either route runs (`cli._missing`),
so the message is the same whichever route would have served the command.
