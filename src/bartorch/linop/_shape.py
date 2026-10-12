"""Operators that rearrange, reduce or restrict a shape, each one BART constructor.

BART counts its dimensions the other way round and always at
:data:`~bartorch._lib.DIMS` of them, so a C-order shape, an axis, a set of
axes and a per-axis vector each have to be turned around on the way in.  The
helpers at the top of this file do that in one place; no constructor below
does its own arithmetic on an index.
"""

from __future__ import annotations

import math

from bartorch import _marshal
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, axes_flags, dims
from bartorch.linop._base import LinearOperator

__all__ = [
    "Extract",
    "Flip",
    "Hankel",
    "Mean",
    "Pad",
    "Permute",
    "Real",
    "Repeat",
    "Reshape",
    "Resize",
    "Roll",
    "ScaledSum",
    "Sum",
    "Transpose",
]

#: What :class:`Pad` and :class:`Roll` fill the new entries with, under numpy's
#: names for the same things.  BART's own are PAD_VALID, PAD_SAME, PAD_CYCLIC,
#: PAD_SYMMETRIC, PAD_REFLECT and PAD_CAUSAL, in that order.
_PADDING = {"constant": 0, "wrap": 2, "symmetric": 3, "reflect": 4}


def _padding(mode: str) -> int:
    if mode not in _PADDING:
        raise ValueError(f"padding mode {mode!r} is not one of {sorted(_PADDING)}")
    return _PADDING[mode]


def _axis(axis: int, ndim: int) -> int:
    """A C-order axis index as BART's, counting from the other end."""
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of range for {ndim} axes")
    return ndim - 1 - (axis % ndim)


def _reduced(shape: Shape, axes) -> tuple[int, ...]:
    """``shape`` with the given axes set to one, the shape BART reduces to."""
    ndim = len(shape)
    keep = {a % ndim for a in (axes if isinstance(axes, (tuple, list)) else (axes,))}
    return tuple(1 if i in keep else n for i, n in enumerate(shape))


class Real(LinearOperator):
    """Real part, BART's ``linop_zreal``.

    The output is complex with zero imaginary part.  Like
    :class:`~bartorch.linop.Conj` the operator is linear over the reals only:
    a projection, self-adjoint under the real inner product, that fails a
    complex dot test on its own.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    """

    def __init__(self, shape: Shape):
        self._shape = tuple(shape)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(library().bartorch_linop_zreal, DIMS, dims(self._shape))
        return Built(ptr, self._shape, self._shape)


class Sum(LinearOperator):
    """Sum over ``axes``, BART's ``linop_sum``.

    The summed axes are kept with size one.  The adjoint is :class:`Repeat`.
    :meth:`~bartorch.linop.LinearOperator.pinv` uses the solver: BART's
    closed form on this operator is :class:`ScaledSum`'s.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    axes : int or tuple of int
        Axes to sum.
    """

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self.ishape = self._shape
        self.oshape = _reduced(self._shape, axes)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_sum, DIMS, dims(self._shape), flags)
        return Built(ptr, self.ishape, self.oshape)


class ScaledSum(LinearOperator):
    """Sum over ``axes``, divided by the square root of how many were summed.

    BART's ``linop_scaled_sum``.  The normal operator is an orthogonal
    projection, so the spectral norm is one, and
    :meth:`~bartorch.linop.LinearOperator.pinv` is BART's closed form rather
    than an iteration.  The summed axes are kept with size one.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    axes : int or tuple of int
        Axes to sum.
    """

    # Checked against an exact damped solve in tests/test_linop_shape.py, for
    # several sizes and dampings.  BART's routine is right for this operator
    # and wrong for the plain sum it is shared with, so this is where the flag
    # goes and nowhere else.
    _exact_pinv = True

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self.ishape = self._shape
        self.oshape = _reduced(self._shape, axes)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_scaled_sum, DIMS, dims(self._shape), flags)
        return Built(ptr, self.ishape, self.oshape)


class Mean(LinearOperator):
    """Average over ``axes``, BART's ``linop_avg``.

    :class:`Sum` divided by the number of summed elements; the averaged axes
    are kept with size one.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    axes : int or tuple of int
        Axes to average.
    """

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self.ishape = self._shape
        self.oshape = _reduced(self._shape, axes)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_avg, DIMS, dims(self._shape), flags)
        return Built(ptr, self.ishape, self.oshape)


class Repeat(LinearOperator):
    """Repeat along ``axes`` up to ``shape``, BART's ``linop_repmat``.

    The adjoint of :class:`Sum`.  The domain is ``shape`` with ``axes`` set
    to one.

    Parameters
    ----------
    shape : tuple of int
        Codomain, C order.
    axes : int or tuple of int
        Axes to repeat along.
    """

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self.oshape = self._shape
        self.ishape = _reduced(self._shape, axes)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_repmat, DIMS, dims(self._shape), flags)
        return Built(ptr, self.ishape, self.oshape)


class Flip(LinearOperator):
    """Reverse ``axes``, BART's ``linop_flip``.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    axes : int or tuple of int
        Axes to reverse.
    """

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_flip, DIMS, dims(self._shape), flags)
        return Built(ptr, self._shape, self._shape)


class _Hankelization(LinearOperator):
    """``linop_hankelization``: the window on the first BART dimension going spare."""

    def __init__(self, ishape, oshape, axis: int, window: int):
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        self.axis, self.window = axis, window
        super().__init__()

    def _create(self) -> Built:
        ndim = len(self.ishape)
        ptr = self._under_lock(
            library().bartorch_linop_hankel,
            DIMS,
            dims(self.ishape),
            _axis(self.axis, ndim),
            ndim,
            self.window,
        )
        return Built(ptr, self.ishape, self.oshape)


def Hankel(shape: Shape, axis: int, window: int) -> LinearOperator:  # noqa: N802
    """Sliding window along ``axis`` (Hankel matrix), BART's ``linop_hankelization``.

    ``torch.Tensor.unfold(axis, window, 1)`` as an operator: an axis of
    length ``n`` becomes ``n - window + 1`` positions and the window is
    appended as the last axis.  The adjoint sums each sample over every
    window containing it.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    axis : int
        Axis to slide along.
    window : int
        Window length, from 1 to the length of the axis.

    Returns
    -------
    LinearOperator
        Codomain ``shape`` with ``axis`` shortened to ``n - window + 1`` and
        a trailing axis of length ``window``.

    Raises
    ------
    ValueError
        When the window is empty or longer than the axis.

    Examples
    --------
    >>> H = Hankel((64, 8), axis=0, window=16)
    >>> H.oshape
    (49, 8, 16)
    """
    from bartorch.linop._shape import Permute  # noqa: PLC0415  (itself, after definition)

    ishape = tuple(shape)
    ndim = len(ishape)
    axis %= ndim
    window = int(window)

    if window < 1:
        raise ValueError(f"a window of {window} has nothing in it")
    if window > ishape[axis]:
        raise ValueError(
            f"a window of {window} does not fit in axis {axis}, which is {ishape[axis]} long"
        )
    if ndim >= DIMS:
        raise ValueError(f"the window axis would take this past BART's {DIMS} dimensions")

    # BART puts the window on a dimension of its own, and the first one going
    # spare reads back as a leading axis in C order.  torch.unfold puts it
    # last, so one permute moves it there.
    positions = list(ishape)
    positions[axis] = ishape[axis] - window + 1
    leading = (window, *positions)

    out: LinearOperator = _Hankelization(ishape, leading, axis, window)
    order = (*range(1, ndim + 1), 0)
    return Permute(leading, order) @ out


class Reshape(LinearOperator):
    """Reshape, BART's ``linop_reshape``.

    The identity on the buffer, C order; both shapes hold the same number of
    elements.

    Parameters
    ----------
    oshape, ishape : tuple of int
        Codomain and domain, C order.
    """

    def __init__(self, oshape: Shape, ishape: Shape):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        if math.prod(self.oshape) != math.prod(self.ishape):
            raise ValueError(
                f"cannot reshape {self.ishape} to {self.oshape}: "
                f"{math.prod(self.ishape)} elements against {math.prod(self.oshape)}"
            )
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_reshape, DIMS, dims(self.oshape), DIMS, dims(self.ishape)
        )
        return Built(ptr, self.ishape, self.oshape)


class Resize(LinearOperator):
    """Crop or zero-fill about the centre of each axis, BART's ``resize -c``.

    Each axis is cropped or zero-filled symmetrically about index ``n // 2``,
    which keeps the centre of a centred Fourier transform in place.

    Parameters
    ----------
    oshape, ishape : tuple of int
        Codomain and domain, C order, with the same number of axes.
    """

    def __init__(self, oshape: Shape, ishape: Shape):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        if len(self.oshape) != len(self.ishape):
            raise ValueError(
                f"resizing keeps the axes it has: {self.ishape} and {self.oshape} differ in rank"
            )
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_resize, DIMS, dims(self.oshape), dims(self.ishape)
        )
        return Built(ptr, self.ishape, self.oshape)


class Extract(LinearOperator):
    """Take the block of ``oshape`` that starts at ``start``, BART's ``linop_extract``.

    The adjoint zero-fills outside the block.  ``A[key]`` builds one.

    Parameters
    ----------
    start : sequence of int
        Offset of the block, one per axis of ``ishape``.
    oshape, ishape : tuple of int
        Block shape and domain, C order, with the same rank.
    """

    def __init__(self, start, oshape: Shape, ishape: Shape):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        self.start = tuple(int(s) for s in start)
        if len(self.oshape) != len(self.ishape):
            raise ValueError(
                f"extracting keeps the axes it has: {self.ishape} and {self.oshape} differ in rank"
            )
        for axis, (begin, width, full) in enumerate(zip(self.start, self.oshape, self.ishape)):
            if begin < 0 or begin + width > full:
                raise ValueError(
                    f"axis {axis}: a block of {width} from {begin} does not fit in {full}"
                )
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_extract,
            DIMS,
            _marshal.padded_offsets(self.start, len(self.ishape), "start"),
            dims(self.oshape),
            dims(self.ishape),
        )
        return Built(ptr, self.ishape, self.oshape)


class Transpose(LinearOperator):
    """Swap two axes, BART's ``linop_transpose``.

    ``torch.transpose`` as an operator; :class:`Permute` is the general case.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    axis0, axis1 : int
        Axes to exchange.
    """

    def __init__(self, shape: Shape, axis0: int, axis1: int):
        self._shape = tuple(shape)
        ndim = len(self._shape)
        self.axis0, self.axis1 = axis0 % ndim, axis1 % ndim
        if self.axis0 == self.axis1:
            raise ValueError(f"axis {axis0} cannot be exchanged with itself")
        self.ishape = self._shape
        swapped = list(self._shape)
        swapped[self.axis0], swapped[self.axis1] = swapped[self.axis1], swapped[self.axis0]
        self.oshape = tuple(swapped)
        super().__init__()

    def _create(self) -> Built:
        ndim = len(self.ishape)
        ptr = self._under_lock(
            library().bartorch_linop_transpose,
            DIMS,
            _axis(self.axis0, ndim),
            _axis(self.axis1, ndim),
            dims(self.ishape),
        )
        return Built(ptr, self.ishape, self.oshape)


class Permute(LinearOperator):
    """Rearrange the axes, BART's ``linop_permute``.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    order : sequence of int
        Permutation of ``range(len(shape))``, as for ``torch.permute``: axis
        ``i`` of the output is axis ``order[i]`` of the input.
    """

    def __init__(self, shape: Shape, order):
        self.ishape = tuple(shape)
        ndim = len(self.ishape)
        self.order = tuple(int(o) % ndim for o in order)
        if sorted(self.order) != list(range(ndim)):
            raise ValueError(f"{tuple(order)} is not a permutation of {ndim} axes")
        self.oshape = tuple(self.ishape[o] for o in self.order)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_permute,
            DIMS,
            _marshal.padded_order(self.order, len(self.ishape)),
            dims(self.ishape),
        )
        return Built(ptr, self.ishape, self.oshape)


class Roll(LinearOperator):
    """Shift along one axis, BART's ``linop_shift``.

    With ``mode="wrap"`` this is ``torch.roll`` and is unitary.  With another
    mode the samples shifted out are dropped and the vacated ones are filled
    by that mode.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    shift : int
        Shift in samples, towards higher indices when positive.
    axis : int, default=-1
        Axis to shift along.
    mode : {"wrap", "constant", "symmetric", "reflect"}, default='wrap'
        Fill of the vacated samples, under numpy's padding names.
    """

    def __init__(self, shape: Shape, shift: int, axis: int = -1, mode: str = "wrap"):
        self._shape = tuple(shape)
        self.shift = int(shift)
        self.axis = axis % len(self._shape)
        self.mode = mode
        self._pad = _padding(mode)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_shift,
            DIMS,
            dims(self._shape),
            _axis(self.axis, len(self._shape)),
            self.shift,
            self._pad,
        )
        return Built(ptr, self._shape, self._shape)


class Pad(LinearOperator):
    """Make each axis longer, BART's ``linop_padding``.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    before : int or sequence of int
        Samples added at the start of each axis; an int applies to every axis.
    after : int or sequence of int, default=None
        Samples added at the end of each axis; ``None`` takes ``before``.
    mode : {"constant", "wrap", "symmetric", "reflect"}, default='constant'
        Fill under numpy's padding names; ``"constant"`` is zeros.
    """

    def __init__(self, shape: Shape, before, after=None, mode: str = "constant"):
        self.ishape = tuple(shape)
        ndim = len(self.ishape)
        self.before = (before,) * ndim if isinstance(before, int) else tuple(before)
        after = self.before if after is None else after
        self.after = (after,) * ndim if isinstance(after, int) else tuple(after)
        if len(self.before) != ndim or len(self.after) != ndim:
            raise ValueError(f"before and after need one entry per axis, {ndim} of them")
        if any(b < 0 for b in self.before) or any(a < 0 for a in self.after):
            raise ValueError("padding cannot be negative; crop with Resize or Extract")
        self.mode = mode
        self._pad = _padding(mode)
        self.oshape = tuple(n + b + a for n, b, a in zip(self.ishape, self.before, self.after))
        super().__init__()

    def _create(self) -> Built:
        ndim = len(self.ishape)
        ptr = self._under_lock(
            library().bartorch_linop_padding,
            DIMS,
            dims(self.ishape),
            self._pad,
            _marshal.padded_offsets(self.before, ndim, "before"),
            _marshal.padded_offsets(self.after, ndim, "after"),
        )
        return Built(ptr, self.ishape, self.oshape)
