"""Neural networks for complex images, and their place in this package's iterations.

:class:`UNet` is a residual network for 2D and 3D images, factorised over a
frame axis and conditioned on the iteration, the noise level or a class;
:class:`ComplexNet` lays complex images out as its channels and
:class:`Patchwise` runs it patch by patch on a device while the image stays on
the host.  :class:`Unrolled` applies one of :mod:`bartorch.optim`'s iteration
blocks a fixed number of times, and a network used as a regularizer is
:class:`bartorch.priors.ImplicitPrior`.  :func:`split` partitions acquired
samples for self-supervised training, and :func:`moments` and
:func:`calibrate` give a voxel-wise uncertainty.

:mod:`bartorch.learning.training` holds the training stages, on ``lightning``
and ``torchio``; nothing else here imports a training library.
"""

from __future__ import annotations

from bartorch.learning.channels import ComplexNet, as_complex, as_real
from bartorch.learning.nets import UNet
from bartorch.learning.patches import Patchwise
from bartorch.learning.splitting import split
from bartorch.learning.uncertainty import calibrate, moments
from bartorch.learning.unrolled import Unrolled

__all__ = [
    "ComplexNet",
    "Patchwise",
    "UNet",
    "Unrolled",
    "as_complex",
    "as_real",
    "calibrate",
    "moments",
    "split",
]
