"""Operators that filter, differentiate, decompose or multiply by a matrix.

Each is one of BART's constructors.  What they have in common with everything
else in :mod:`bartorch.linop` is that they map a tensor to a tensor, so each is
a class and each composes with ``@``, ``+`` and the rest of the algebra into a
single BART operator.
"""

from __future__ import annotations

import torch

from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, axes_flags, dims
from bartorch.linop._base import LinearOperator

__all__ = ["Convolve", "Gradient", "Matrix", "Sobolev"]

#: numpy's names for what BART calls CONV_CYCLIC, TRUNCATED, VALID and EXTENDED.
_CONV_TYPE = {"wrap": 0, "same": 1, "valid": 2, "full": 3}

#: BART's conv_mode.
_CONV_MODE = {"symmetric": 0, "causal": 1, "anticausal": 2}


class Matrix(LinearOperator):
    """Multiplication by a matrix along one axis, BART's ``linop_matrix``.

    The contracted axis has size one in the codomain and the produced axis
    size one in the domain.  The matrix is broadcast over its axes of size
    one and may vary along the others, for instance one matrix per slice.

    Parameters
    ----------
    matrix : tensor
        Same rank as the operator: the domain's size along the contracted
        axis, the codomain's along the produced axis.
    oshape, ishape : tuple of int
        Codomain and domain, C order, with the same rank.

    Examples
    --------
    A basis ``(k, t)`` applied to ``(1, t, y, x)``, giving ``(k, 1, y, x)``:

    >>> B = Matrix(basis[:, :, None, None], (k, 1, y, x), (1, t, y, x))
    """

    def __init__(self, matrix: torch.Tensor, oshape: Shape, ishape: Shape):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        if len(self.oshape) != len(self.ishape):
            raise ValueError(
                f"a matrix keeps the axes it has: {self.ishape} and {self.oshape} differ in rank"
            )
        self.matrix = as_operand(matrix, tuple(matrix.shape), "matrix")
        if len(self.matrix.shape) != len(self.ishape):
            raise ValueError(
                f"the matrix needs one axis per axis of the operator: "
                f"{tuple(self.matrix.shape)} against {self.ishape}"
            )
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_matrix,
            DIMS,
            dims(self.oshape),
            dims(self.ishape),
            dims(tuple(self.matrix.shape)),
            self.matrix.data_ptr(),
            device=self.matrix.device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=(self.matrix,))


class Convolve(LinearOperator):
    """Convolution with a fixed kernel, BART's ``linop_conv``.

    Parameters
    ----------
    kernel : tensor
        Same rank as ``shape``.
    shape : tuple of int
        Domain, C order.
    axes : int or tuple of int
        Axes to convolve along.
    mode : {"wrap", "same", "valid", "full"}, default='same'
        Boundary handling, under numpy's names: circular, truncated to the
        input size, only where the kernel fits, or extended.
    direction : {"symmetric", "causal", "anticausal"}, default=None
        Kernel alignment.  ``None`` is ``"causal"`` for ``"valid"`` and
        ``"full"``, the only direction BART accepts for those modes, and
        ``"symmetric"`` otherwise.

    Raises
    ------
    ValueError
        For an unknown mode or direction, a non-causal ``"valid"`` or
        ``"full"`` convolution, or a kernel whose rank differs from the
        domain's.
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        shape: Shape,
        axes,
        mode: str = "same",
        direction: str | None = None,
    ):
        if mode not in _CONV_TYPE:
            raise ValueError(f"mode {mode!r} is not one of {sorted(_CONV_TYPE)}")
        if direction is None:
            direction = "causal" if mode in ("valid", "full") else "symmetric"
        if direction not in _CONV_MODE:
            raise ValueError(f"direction {direction!r} is not one of {sorted(_CONV_MODE)}")
        if mode in ("valid", "full") and "causal" != direction:
            raise ValueError(
                f"BART plans a {mode!r} convolution only as a causal one, not {direction!r}"
            )

        self.ishape = tuple(shape)
        self.kernel = as_operand(kernel, tuple(kernel.shape), "kernel")
        self.axes, self.mode, self.direction = axes, mode, direction

        if len(self.kernel.shape) != len(self.ishape):
            raise ValueError(
                f"the kernel needs one axis per axis of the operator: "
                f"{tuple(self.kernel.shape)} against {self.ishape}"
            )

        ndim = len(self.ishape)
        convolved = {a % ndim for a in (axes if isinstance(axes, (tuple, list)) else (axes,))}
        widths = tuple(self.kernel.shape)
        self.oshape = tuple(
            _conv_size(mode, n, widths[axis]) if axis in convolved else n
            for axis, n in enumerate(self.ishape)
        )
        if any(n < 1 for n in self.oshape):
            raise ValueError(f"a {widths} kernel leaves nothing of {self.ishape} in {mode!r} mode")
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_conv,
            DIMS,
            axes_flags(self.axes, len(self.ishape)),
            _CONV_TYPE[self.mode],
            _CONV_MODE[self.direction],
            dims(self.oshape),
            dims(self.ishape),
            dims(tuple(self.kernel.shape)),
            self.kernel.data_ptr(),
            device=self.kernel.device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=(self.kernel,))


def _conv_size(mode: str, n: int, width: int) -> int:
    if mode in ("wrap", "same"):
        return n
    return n - width + 1 if "valid" == mode else n + width - 1


class _Differences(LinearOperator):
    """``linop_grad``: the differences stacked in BART's own order."""

    def __init__(self, ishape, oshape, axes):
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        self.axes = axes
        super().__init__()

    def _create(self) -> Built:
        ndim = len(self.ishape)
        # The differences go on the first BART dimension the shape does not
        # use, which read back in C order is a new leading axis.
        ptr = self._under_lock(
            library().bartorch_linop_grad,
            DIMS,
            dims(self.ishape),
            ndim,
            axes_flags(self.axes, ndim),
        )
        return Built(ptr, self.ishape, self.oshape)


def Gradient(shape: Shape, axes) -> LinearOperator:  # noqa: N802  (it is a constructor)
    """Forward finite differences along ``axes``, BART's ``linop_grad``.

    The boundary is circular: at the last index along an axis the difference
    is the first entry minus the last.  Constants are the null space.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    axes : int or tuple of int
        Axes to difference along, each at most once.

    Returns
    -------
    LinearOperator
        Codomain ``(len(axes), *shape)``.  The components are in ascending
        axis order, whatever the order of ``axes``.

    Raises
    ------
    ValueError
        When an axis is named twice, or the new axis would exceed BART's
        dimension count.
    """
    from bartorch.linop._shape import Flip

    ishape = tuple(shape)
    ndim = len(ishape)
    chosen = tuple(a % ndim for a in (axes if isinstance(axes, (tuple, list)) else (axes,)))
    if len(set(chosen)) != len(chosen):
        raise ValueError(
            f"{tuple(axes) if not isinstance(axes, int) else axes} names an axis twice"
        )
    if ndim >= DIMS:
        raise ValueError(f"the new axis would take this past BART's {DIMS} dimensions")

    oshape = (len(chosen), *ishape)
    out: LinearOperator = _Differences(ishape, oshape, axes)
    if len(chosen) > 1:
        out = Flip(oshape, axes=0) @ out
    return out


class Sobolev(LinearOperator):
    """Image from Sobolev-weighted k-space coefficients, BART's ``linop_noir_weights``.

    The coefficients are multiplied by ``c (1 + a |k|^2)^(-b/2)`` and
    transformed by the centred unitary inverse FFT along ``axes``.  ``k_i``
    is the frequency index along axis ``i``, from ``-(n_i // 2)``, divided by
    ``n_i``.  Penalising the coefficients rather than the image is the
    Sobolev-norm regularization NLINV applies to coil sensitivities and
    ``moba`` to B1 and B0 maps.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    axes : int or tuple of int
        Axes weighted and transformed.
    a, b : float
        Width and order of the weighting (NLINV's ``a`` and ``b``).
    c : float, default=1.0
        Weight at the centre of k-space.

    Examples
    --------
    A map on a 64x64 grid under ``moba``'s B1 weighting, started from a
    constant:

    >>> W = Real((64, 64)) @ Sobolev((64, 64), (0, 1), 440.0, 20.0)
    >>> coefficients = W.H(torch.ones(64, 64, dtype=torch.complex64))
    """

    def __init__(self, shape: Shape, axes, a: float, b: float, c: float = 1.0):
        self._shape = tuple(shape)
        self.axes = axes
        self.a, self.b, self.c = float(a), float(b), float(c)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_sobolev,
            DIMS,
            dims(self._shape),
            axes_flags(self.axes, len(self._shape)),
            self.a,
            self.b,
            self.c,
        )
        return Built(ptr, self._shape, self._shape)
