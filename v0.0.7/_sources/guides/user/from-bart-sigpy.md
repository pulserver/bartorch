# From BART and sigpy

```python
import bartorch.tools as bt
from bartorch import apps, linop, optim, priors

kspace = bt.phantom(128, coils=8, kspace=True)  # (coils, z, y, x)
maps = bt.ecalib(kspace, maps=1)                # bart ecalib -m1
image = apps.pics(kspace, maps,                 # bart pics -R W:3:0:0.005
                  regularizers=priors.Wavelet((-1, -2), 0.005))

A = linop.CartesianSense(maps, (1, 128, 128))   # sigpy.mri.linop.Sense
x = optim.CG(maxiter=20)(kspace, A)             # sigpy.app.LinearLeastSquares
```

Arrays are torch tensors in C order, with the coils first: `(coils, z, y, x)`,
the reverse of BART's dimension vector and the layout sigpy uses
({doc}`../../explanation/data-layout`). Axes are given as indices, never as
BART bitmasks.

| BART | sigpy | bartorch |
| --- | --- | --- |
| `bart fft -u -i 7` | `sp.ifft(x, axes=(-3, -2, -1))` | `bartorch.ifft(x, (-3, -2, -1), unitary=True)` |
| `bart ecalib` | `mr.app.EspiritCalib` | `bartorch.tools.ecalib` |
| `bart pics -R W:...` | `mr.app.L1WaveletRecon` | `apps.pics(..., regularizers=priors.Wavelet(...))` |
| `bart pics -R T:...` | `mr.app.TotalVariationRecon` | `apps.pics(..., regularizers=priors.TotalVariation(...))` |
| `bart pics -t traj` | `mr.app.SenseRecon(coord=...)` | `apps.pics(..., traj=traj)` |
| `bart nufft` | `sp.nufft` | `bartorch.nufft`, `linop.NUFFT` |
| `bart nlinv` | — | `bartorch.tools.nlinv` |
| `bart moba`, `bart mobafit` | — | `apps.moba`, `apps.mobafit` |
| `bart traj`, `bart phantom` | `mr.radial`, `sp.shepp_logan` | `bartorch.tools.traj`, `bartorch.tools.phantom` |
| — | `A.H`, `A * B`, `A + B` | `A.H`, `A @ B`, `A + B` |
| — | `sp.alg.ConjugateGradient` | `optim.CG` |
| — | `sp.alg.GradientMethod(prox=...)` | `optim.FISTA`, `optim.IST` |
| `bart <command> ...` | — | `bartorch <command> ...`, the same arguments |

BART's `fft` is unnormalised unless `-u`; `linop.FFT` is unitary. Every other
command is a function of `bartorch.tools` ({doc}`../../api/tools`).
