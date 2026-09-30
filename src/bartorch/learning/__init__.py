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

:class:`Reconstruction` trains a network in one of the three stages of an
unrolled network, on ``lightning``, and :class:`RandomGain` is a ``torchio``
augmentation.  Both are imported the first time one of them is asked for, so
importing this package imports no training library.
"""

from __future__ import annotations

from bartorch.learning._channels import ComplexNet, as_complex, as_real
from bartorch.learning._nets import UNet
from bartorch.learning._patches import Patchwise
from bartorch.learning._splitting import split
from bartorch.learning._uncertainty import calibrate, moments
from bartorch.learning._unrolled import Unrolled

#: Names defined in ``_training``, which imports ``lightning`` and ``torchio``.
_TRAINING = ("RandomGain", "Reconstruction")

__all__ = [
    "ComplexNet",
    "Patchwise",
    "RandomGain",
    "Reconstruction",
    "UNet",
    "Unrolled",
    "as_complex",
    "as_real",
    "calibrate",
    "moments",
    "split",
]


def __getattr__(name: str):
    if name in _TRAINING:
        from bartorch.learning import _training

        return getattr(_training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_TRAINING})
