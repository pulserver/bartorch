# Learning

`bartorch.learning` holds neural networks for complex, high-dimensional images
and what places them in this package's iterations: a network for images
carrying frames or subspace coefficients, patchwise execution on a device of an
image held on the host, unrolled iterations, the partition of the acquired
samples for training without a reference, and a voxel-wise uncertainty.  A
network used as a regularizer is a {obj}`~bartorch.priors.ImplicitPrior`,
whose `sigma` may be a schedule over the iterations and which with `step=True`
hands the network the iteration index.  {doc}`../explanation/differentiation`
describes how gradients pass through an unrolled iteration and the memory each
backward strategy uses.

```{eval-rst}
.. currentmodule:: bartorch.learning
```

## Networks

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.UNet` | Residual UNet on 2D or 3D real channels, optionally over a frame axis, conditioned on the iteration index, the noise level and a class |
| {obj}`~bartorch.learning.ComplexNet` | A real network applied to complex images, their leading axes and real and imaginary parts as its channels, normalized by the peak or whitened |
| {obj}`~bartorch.learning.Patchwise` | A network applied patch by patch on a device, with a randomly shifted grid and mixed precision, to an image held elsewhere |

| Precision on the device | `Patchwise(dtype="auto")` |
| --- | --- |
| Card with bfloat16 (Ampere and later: A40, RTX 4000 Ada) | bfloat16 |
| Card without it (Turing: T4) | float16 |
| Host | single precision, no mixed precision |

## Iterations as networks

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.Unrolled` | A fixed number of applications of an iteration block from {mod}`bartorch.optim` or {mod}`bartorch.nlop`, with shared or per-iteration parameters |

| `Unrolled` setting | Graph recorded | Memory | Gradient |
| --- | --- | --- | --- |
| default | The whole stack | Iterations × one step | End to end |
| `detach=True` | One iteration at a time | One step | Of each step alone (greedy training with {meth}`Unrolled.steps`) |
| `checkpoint=True` | The states between iterations | States + one step | End to end; each step is computed twice |

## Training without a reference

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.split` | Partition of the acquired samples into a set reconstructed from and a set held out, for self-supervised training |

## Uncertainty

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.moments` | Voxel-wise mean and variance of a randomized reconstruction repeated |
| {obj}`~bartorch.learning.calibrate` | The factor turning a spread into an interval with a stated coverage, by split conformal calibration |

## Complex values as channels

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.as_real` | Real and imaginary parts on a new leading axis |
| {obj}`~bartorch.learning.as_complex` | Inverse of {obj}`~bartorch.learning.as_real` |

## Training stages

`bartorch.learning.training` imports `lightning` and `torchio`, which
`pip install bartorch[learning]` installs.

```{eval-rst}
.. currentmodule:: bartorch.learning.training
```

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.training.Reconstruction` | A `LightningModule` training a denoiser, an unrolled network greedily, or an unrolled network end to end, supervised or self-supervised, with the batch left on the host |
| {obj}`~bartorch.learning.training.RandomGain` | A `torchio` transform multiplying every image of a subject by one random complex gain |

The examples of {doc}`../auto_examples/06-learning/index` train networks built
from these objects.
