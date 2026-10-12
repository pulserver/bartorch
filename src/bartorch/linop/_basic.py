"""Elementary linear operators, each one BART constructor."""

from __future__ import annotations

from collections.abc import Callable

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError
from bartorch._lib import DIMS, library
from bartorch._operator import (
    Built,
    Shape,
    as_operand,
    axes_flags,
    broadcast_flags,
    callback,
    dims,
)
from bartorch.linop._base import LinearOperator

__all__ = [
    "ComponentDiagonal",
    "Conj",
    "Diagonal",
    "FFT",
    "Identity",
    "MultiplySum",
    "Zero",
]


class FFT(LinearOperator):
    """Unitary Fourier transform along ``axes``, BART's ``linop_fft``.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    axes : int or tuple of int
        Axes to transform; negative indices count from the end.
    inverse : bool, default=False
        Apply the inverse transform as the forward.
    centred : bool, default=True
        Zero frequency at index ``n // 2`` (BART's ``fftc``); False is the
        uncentred transform.  ``centered`` is accepted as a keyword alias.

    Examples
    --------
    >>> F = FFT((8, 16), axes=(-1, -2))
    >>> F(x).shape
    torch.Size([8, 16])
    """

    # The shape a constructor was given stays private: an operator answers for
    # its domain and codomain through ishape and oshape.

    def __init__(self, shape: Shape, axes, inverse: bool = False, centred: bool = True, **kwargs):
        # ``centered`` is accepted as a spelling of ``centred``.
        if "centered" in kwargs:
            centred = kwargs.pop("centered")
        if kwargs:
            raise TypeError(f"unexpected arguments {sorted(kwargs)}")
        self._shape = tuple(shape)
        self.axes = axes
        self.inverse = bool(inverse)
        self.centred = bool(centred)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(
            library().bartorch_linop_fft,
            DIMS,
            dims(self._shape),
            flags,
            int(self.inverse),
            int(self.centred),
        )
        return Built(ptr, self._shape, self._shape)


class _SettableDiagonal(LinearOperator):
    """A BART ``cdiag`` whose values can be rewritten after it was built.

    ``linop_gdiag_set_diag`` writes into the operator that exists and drops its
    cached normal, so every composition and gram standing on this one answers
    for the new values from the next application.  The values are copied, so
    the tensor passed to :meth:`set` need not outlive the call.
    """

    #: The attribute holding the values, named for what the subclass calls them.
    _values: str

    def set(self, values: torch.Tensor) -> None:
        """Replace the values in place, without rebuilding the operator.

        Compositions, normals and solvers built over this operator apply the
        new values from their next application.  The values are copied.

        Parameters
        ----------
        values : tensor
            Same shape as the values the operator was built with.

        Raises
        ------
        BartError
            When BART refuses to write the values.
        """
        current = getattr(self, self._values)
        made = as_operand(values, tuple(current.shape), self._values)
        if "_h" not in self.__dict__:
            # Not built yet, so there is nothing to write into: the constructor
            # will read this when it is.
            setattr(self, self._values, made)
            return
        failed = self._under_lock(
            library().bartorch_linop_set_diagonal,
            self._h.ptr,
            DIMS,
            dims(tuple(current.shape)),
            made.data_ptr(),
            device=made.device,
        )
        if failed:
            raise BartError("BART would not write the diagonal; see the log for its message")
        setattr(self, self._values, made)


class Diagonal(_SettableDiagonal):
    """Pointwise multiplication by ``diag``, BART's ``linop_cdiag``.

    The adjoint multiplies by ``conj(diag)``.

    Parameters
    ----------
    diag : tensor
        The diagonal, with the operator's rank; each axis is the operator's
        size or one, and size-one axes are broadcast.
    shape : tuple of int
        Domain and codomain, C order.
    """

    _values = "diag"

    def __init__(self, diag: torch.Tensor, shape: Shape):
        self._shape = tuple(shape)
        self.diag = as_operand(diag, tuple(diag.shape), "diag")
        super().__init__()

    def _create(self) -> Built:
        flags = broadcast_flags(tuple(self.diag.shape), self._shape)
        ptr = self._under_lock(
            library().bartorch_linop_cdiag,
            DIMS,
            dims(self._shape),
            flags,
            self.diag.data_ptr(),
            device=self.diag.device,
        )
        return Built(ptr, self._shape, self._shape, keep=(self.diag,))


class ComponentDiagonal(LinearOperator):
    """Separate real scalings of the real and imaginary parts, BART's ``linop_rdiag``.

    ``y = Re(d) Re(x) + i Im(d) Im(x)`` (BART's ``md_zrmul``).  A real ``d``
    therefore zeroes the imaginary part; multiplication by a real diagonal is
    :class:`Diagonal`.

    The operator is linear over the reals only, like :class:`Conj` and
    :class:`~bartorch.linop.Real`: it is self-adjoint under the real inner
    product and fails a complex dot test on its own.

    Parameters
    ----------
    diag : tensor
        Real part scales the real part, imaginary part scales the imaginary
        part.  Each axis is the operator's size or one, and size-one axes are
        broadcast.
    shape : tuple of int
        Domain and codomain, C order.
    """

    def __init__(self, diag: torch.Tensor, shape: Shape):
        self._shape = tuple(shape)
        self.diag = as_operand(diag, tuple(diag.shape), "diag")
        super().__init__()

    def _create(self) -> Built:
        flags = broadcast_flags(tuple(self.diag.shape), self._shape)
        ptr = self._under_lock(
            library().bartorch_linop_rdiag,
            DIMS,
            dims(self._shape),
            flags,
            self.diag.data_ptr(),
            device=self.diag.device,
        )
        return Built(ptr, self._shape, self._shape, keep=(self.diag,))


class Sampling(_SettableDiagonal):
    """Multiplication by a sampling pattern, broadcast over the axes where it is one.

    Parameters
    ----------
    pattern : tensor
        Binary sampling mask, one at acquired positions and zero elsewhere.
    shape : tuple of int
        The k-space shape, C order.
    """

    _values = "pattern"

    def __init__(self, pattern: torch.Tensor, shape: Shape):
        self._shape = tuple(shape)
        self.pattern = as_operand(pattern, tuple(pattern.shape), "pattern")
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_sampling,
            dims(self._shape),
            dims(tuple(self.pattern.shape)),
            self.pattern.data_ptr(),
            device=self.pattern.device,
        )
        return Built(ptr, self._shape, self._shape, keep=(self.pattern,))


class MultiplySum(LinearOperator):
    """Multiply by a tensor and sum over the axes the codomain reduces, BART's ``linop_fmac``.

    With sensitivities of shape ``(coils, y, x)``, ``ishape=(1, y, x)`` and
    ``oshape=(coils, y, x)`` this is the coil model: image to coil images,
    and the adjoint is the coil combination with conjugate sensitivities.

    Parameters
    ----------
    tensor : tensor
        Multiplier, broadcast against the domain and codomain.
    ishape, oshape : tuple of int
        Domain and codomain, C order.  An axis of size one in ``oshape``
        and larger in the product is summed over.
    """

    def __init__(self, tensor: torch.Tensor, ishape: Shape, oshape: Shape):
        self.tensor = as_operand(tensor, tuple(tensor.shape), "tensor")
        # ``Operator._build`` sets these again from what _create returns; a
        # concrete operator may name them itself so that _create can read them.
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_fmac,
            DIMS,
            dims(self.oshape),
            dims(self.ishape),
            dims(tuple(self.tensor.shape)),
            self.tensor.data_ptr(),
            device=self.tensor.device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=(self.tensor,))


class Identity(LinearOperator):
    """Identity on ``shape``, BART's ``linop_identity``.

    ``A ** 0`` returns one; ``A + 0.1 * Identity(A.ishape)`` adds a multiple
    of the identity.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    """

    def __init__(self, shape: Shape):
        self._shape = tuple(shape)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(library().bartorch_linop_identity, DIMS, dims(self._shape))
        return Built(ptr, self._shape, self._shape)


class Zero(LinearOperator):
    """Zero operator, BART's ``linop_null``.

    Parameters
    ----------
    oshape : tuple of int
        Codomain, C order.
    ishape : tuple of int, default=None
        Domain, C order; the same as ``oshape`` when left out.
    """

    def __init__(self, oshape: Shape, ishape: Shape | None = None):
        self.oshape = tuple(oshape)
        self.ishape = tuple(oshape if ishape is None else ishape)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_null, DIMS, dims(self.oshape), DIMS, dims(self.ishape)
        )
        return Built(ptr, self.ishape, self.oshape)


class Conj(LinearOperator):
    """Complex conjugation, BART's ``linop_zconj``.

    Conjugation is antilinear: linear over the reals, not over the complex
    numbers, so the operator fails a complex dot test on its own.
    :meth:`LinearOperator.conj` and :attr:`LinearOperator.T` are built from it.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    """

    def __init__(self, shape: Shape):
        self._shape = tuple(shape)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(library().bartorch_linop_zconj, DIMS, dims(self._shape))
        return Built(ptr, self._shape, self._shape)


class _Callback(LinearOperator):
    """A linear operator from Python functions, applied through BART; see
    :meth:`LinearOperator.from_callbacks`.

    Each function receives a view of BART's buffer, without a copy, and returns a
    tensor; every application crosses into Python.

    Parameters
    ----------
    oshape, ishape : tuple of int
        Codomain and domain shapes, C order.
    forward, adjoint : callable
        Maps from ``ishape`` to ``oshape`` and back.
    normal : callable, optional
        ``adjoint(forward(x))`` in one function, where a cheaper form exists;
        without one BART composes the two.
    """

    def __init__(
        self,
        oshape: Shape,
        ishape: Shape,
        forward: Callable[[torch.Tensor], torch.Tensor],
        adjoint: Callable[[torch.Tensor], torch.Tensor],
        normal: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        self.forward_fn, self.adjoint_fn, self.normal_fn = forward, adjoint, normal
        super().__init__()

    def _create(self) -> Built:
        ishape, oshape = self.ishape, self.oshape
        fwd = callback(self.forward_fn, ishape, oshape, "forward")
        adj = callback(self.adjoint_fn, oshape, ishape, "adjoint")
        nrm = (
            callback(self.normal_fn, ishape, ishape, "normal")
            if self.normal_fn is not None
            else _marshal.null_apply()
        )
        ptr = self._under_lock(
            library().bartorch_linop_callback,
            DIMS,
            dims(oshape),
            DIMS,
            dims(ishape),
            fwd,
            adj,
            nrm,
            None,
            _marshal.null_release(),
        )
        keep = (fwd, adj, nrm, self.forward_fn, self.adjoint_fn, self.normal_fn)
        return Built(ptr, ishape, oshape, keep=keep)
