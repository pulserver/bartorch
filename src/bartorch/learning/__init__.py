"""Adapters between neural networks and this package's iterations.

:class:`Unrolled` applies one of :mod:`bartorch.optim`'s iteration blocks a
fixed number of times, with the differentiation strategies a deep stack
requires.  :func:`as_real` and :func:`as_complex` convert between complex
tensors and the leading real channel axis of convolutional networks and
``torchio`` images.  A network used as a regularizer is
:class:`bartorch.priors.ImplicitPrior`.

Training loops, datasets, augmentation, networks, losses and metrics are
``lightning``'s, ``torchio``'s, ``monai``'s, ``deepinv``'s and
``torchmetrics``'s; this subpackage imports none of them.
"""

from __future__ import annotations

from bartorch.learning.channels import as_complex, as_real
from bartorch.learning.unrolled import Unrolled

__all__ = ["Unrolled", "as_complex", "as_real"]
