"""The basic nonlinear operators, each one BART constructor.

:class:`Multiply` is the two-input product every model is built out of,
broadcast the way BART broadcasts.  Everything here works on ``complex
float``, as BART's nonlinear operators do: :class:`Abs` returns a real
quantity in a complex tensor with a zero imaginary part, as ``zabs`` does.
"""

from __future__ import annotations

import torch

from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, axes_flags, dims
from bartorch.nlop._base import NonlinearOperator, _built

__all__ = [
    "Abs",
    "Add",
    "Constant",
    "Divide",
    "Exp",
    "Inverse",
    "Log",
    "Multiply",
    "Phase",
    "Power",
    "RootSumOfSquares",
    "SmoothAbs",
    "Sqrt",
    "SumOfSquares",
    "Weighted",
]


def _broadcast(a: Shape, b: Shape) -> Shape:
    """The shape BART's tensor product gives two operands, which is torch's rule."""
    if len(a) != len(b):
        raise ValueError(f"{a} and {b} do not have the same rank; BART broadcasts axis by axis")
    out = []
    for axis, (m, n) in enumerate(zip(a, b)):
        if m != n and 1 != m and 1 != n:
            raise ValueError(f"axis {axis} is {m} and {n}, which do not broadcast")
        out.append(max(m, n))
    return tuple(out)


class _Elementwise(NonlinearOperator):
    """One input to one output of the same shape, through one BART constructor."""

    #: The library function, and whatever it takes after the dimensions.
    _fn: str = ""

    def __init__(self, shape: Shape):
        self._shape = tuple(shape)
        super().__init__()

    def _extra(self) -> tuple:
        return ()

    def _create(self) -> Built:
        ptr = self._under_lock(
            getattr(library(), self._fn), DIMS, dims(self._shape), *self._extra()
        )
        return Built(ptr, self._shape, self._shape)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._shape})"


class Exp(_Elementwise):
    """Elementwise ``exp(x)``, BART's ``zexp``.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    """

    _fn = "bartorch_nlop_zexp"

    def _bundle(self):
        from bartorch.nlop._bundle import diagonal

        return diagonal(self, Exp(self._shape))


class Log(_Elementwise):
    """Elementwise principal ``log(x)``, BART's ``zlog``.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    """

    _fn = "bartorch_nlop_zlog"

    def _bundle(self):
        from bartorch.nlop._bundle import diagonal

        return diagonal(self, Inverse(self._shape))


class Sqrt(_Elementwise):
    """Elementwise principal ``sqrt(x)``, BART's ``zsqrt``.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    """

    _fn = "bartorch_nlop_zsqrt"

    def _bundle(self):
        from bartorch.nlop._base import _chain
        from bartorch.nlop._bundle import diagonal, scaled

        # `zsqrt_apply` fills the diagonal with 0.5 and divides it by the
        # value, so it is half the inverse of the root and not of the point.
        half = _chain(_chain(Sqrt(self._shape), Inverse(self._shape)), scaled(self._shape, 0.5))
        return diagonal(self, half)


class Abs(_Elementwise):
    """Elementwise ``|x|``, BART's ``zabs``.

    The result is complex with zero imaginary part.  The derivative is
    undefined at zero; :class:`SmoothAbs` is finite there.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    """

    _fn = "bartorch_nlop_zabs"

    def _composition(self):
        """``nlop_zabs_create`` is ``zrss`` over no axes (``someops.c:626``)."""
        return RootSumOfSquares(self._shape, ())

    def _bundle(self):
        from bartorch.nlop._bundle import of_composition

        return of_composition(self, self._composition())


class SmoothAbs(_Elementwise):
    """Elementwise ``sqrt(|x|^2 + eps)``, BART's ``smo_abs``.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    eps : float, default=1e-12
        Offset that keeps the derivative finite at zero.
    """

    _fn = "bartorch_nlop_smo_abs"

    def __init__(self, shape: Shape, eps: float = 1e-12):
        self.eps = float(eps)
        super().__init__(shape)

    def _extra(self) -> tuple:
        return (self.eps,)

    def _composition(self):
        """``nlop_smo_abs_create`` is ``zrss`` over no axes, regularised (``someops.c:621``)."""
        return RootSumOfSquares(self._shape, (), self.eps)

    def _bundle(self):
        from bartorch.nlop._bundle import of_composition

        return of_composition(self, self._composition())

    def __repr__(self) -> str:
        return f"SmoothAbs({self._shape}, eps={self.eps})"


class Inverse(_Elementwise):
    """Elementwise ``1 / x``, BART's ``zinv``.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    eps : float, default=0.0
        Positive selects BART's regularized form ``conj(x) / (|x|^2 + eps)``.
    """

    _fn = "bartorch_nlop_zinv"

    def __init__(self, shape: Shape, eps: float = 0.0):
        self.eps = float(eps)
        super().__init__(shape)

    def _extra(self) -> tuple:
        return (self.eps,)

    def _bundle(self):
        from bartorch.nlop._base import _chain
        from bartorch.nlop._bundle import diagonal, scaled

        # `zinv_reg_fun` squares the value and negates it, so the diagonal is
        # minus the square of the regularised inverse rather than of `1 / x`.
        squared = Multiply(self._shape, self._shape)._dup(0, 1)
        made = _chain(_chain(Inverse(self._shape, self.eps), squared), scaled(self._shape, -1.0))
        return diagonal(self, made)

    def __repr__(self) -> str:
        return f"Inverse({self._shape}, eps={self.eps})"


class Power(_Elementwise):
    """Elementwise ``x ** exponent``, BART's ``zspow``.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    exponent : complex
        Scalar exponent.
    """

    _fn = "bartorch_nlop_zspow"

    def __init__(self, shape: Shape, exponent: complex):
        self.exponent = complex(exponent)
        super().__init__(shape)

    def _extra(self) -> tuple:
        return (self.exponent.real, self.exponent.imag)

    def _bundle(self):
        from bartorch.nlop._base import _chain
        from bartorch.nlop._bundle import diagonal, scaled

        # `zspow_fun` divides the value by the point rather than raising the
        # point to one power less, which differs where the point is zero.
        # BART divides with `md_zdiv` and `Divide` multiplies by the inverse,
        # so this is that diagonal to within one rounding rather than to the
        # bit.
        ratio = _chain(Power(self._shape, self.exponent), Divide(self._shape))._dup(0, 1)
        return diagonal(self, _chain(ratio, scaled(self._shape, self.exponent)))

    def __repr__(self) -> str:
        return f"Power({self._shape}, {self.exponent})"


class Add(_Elementwise):
    """Elementwise ``x + value``, BART's ``zsadd``.

    An affine map, hence a nonlinear operator; its derivative is the identity.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.
    value : complex
        Scalar added to every entry.
    """

    _fn = "bartorch_nlop_zsadd"

    def __init__(self, shape: Shape, value: complex):
        self.value = complex(value)
        super().__init__(shape)

    def _extra(self) -> tuple:
        return (self.value.real, self.value.imag)

    def _bundle(self):
        from bartorch.linop._basic import Identity
        from bartorch.nlop._base import FromLinear
        from bartorch.nlop._bundle import linear

        one = FromLinear(Identity(self._shape))
        return linear(self, one, one)

    def __repr__(self) -> str:
        return f"Add({self._shape}, {self.value})"


class _Reduction(NonlinearOperator):
    """One input reduced along ``axes``, which become one."""

    _fn: str = ""

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self._flags = axes_flags(axes, len(self._shape))
        self._out = tuple(
            1 if self._flags >> (len(self._shape) - 1 - axis) & 1 else size
            for axis, size in enumerate(self._shape)
        )
        super().__init__()

    def _extra(self) -> tuple:
        return ()

    def _create(self) -> Built:
        ptr = self._under_lock(
            getattr(library(), self._fn), DIMS, dims(self._shape), self._flags, *self._extra()
        )
        return Built(ptr, self._shape, self._out)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._shape}, axes={self.axes})"


class SumOfSquares(_Reduction):
    """Sum of squared magnitudes along ``axes``, BART's ``zss``.

    The reduced axes are kept with size one; the result is complex with zero
    imaginary part.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    axes : sequence of int
        Axes summed over.
    """

    _fn = "bartorch_nlop_zss"

    def _composition(self):
        """``nlop_zss_create``: the point against its own conjugate, summed, made real.

        The product contracts onto the reduced shape, as
        ``md_ztenmul`` does with a smaller output (``someops.c:539``).
        """
        from bartorch.linop._basic import Conj
        from bartorch.linop._shape import Real
        from bartorch.nlop._base import FromLinear, _chain
        from bartorch.nlop._bundle import _TenMul

        product = _chain(
            FromLinear(Conj(self._shape)),
            _TenMul(self._out, self._shape, self._shape),
            output=0,
            input=0,
        )
        return _chain(product._dup(0, 1), FromLinear(Real(self._out)))

    def _bundle(self):
        from bartorch.nlop._bundle import of_composition

        return of_composition(self, self._composition())


class RootSumOfSquares(_Reduction):
    """Root sum of squared magnitudes along ``axes``, BART's ``zrss``.

    The reduced axes are kept with size one; the result is complex with zero
    imaginary part.

    Parameters
    ----------
    shape : tuple of int
        Domain, C order.
    axes : sequence of int
        Axes summed over; ``()`` gives the elementwise magnitude.
    eps : float, default=0.0
        Positive computes ``sqrt(sum(|x|^2) + eps)``, whose derivative is
        finite where the sum is zero.
    """

    _fn = "bartorch_nlop_zrss"

    def __init__(self, shape: Shape, axes, eps: float = 0.0):
        self.eps = float(eps)
        super().__init__(shape, axes)

    def _extra(self) -> tuple:
        return (self.eps,)

    def _composition(self):
        """``nlop_zrss_reg_create``: the sum of squares, offset where asked, rooted."""
        from bartorch.nlop._base import _chain

        made = SumOfSquares(self._shape, self.axes)
        if self.eps:
            made = _chain(made, Add(self._out, self.eps))
        return _chain(made, Sqrt(self._out))

    def _bundle(self):
        from bartorch.nlop._bundle import of_composition

        return of_composition(self, self._composition())


class Multiply(NonlinearOperator):
    """Pointwise product of two inputs, BART's ``tenmul``.

    The derivative by either input is multiplication by the other.  ``nlinv``'s
    model is this product of an image ``(1, y, x)`` and coil profiles
    ``(coils, y, x)``.

    Parameters
    ----------
    a, b : tuple of int
        Shapes of the two inputs, C order and of equal rank; axes of size one
        broadcast.

    Raises
    ------
    ValueError
        If the ranks differ or an axis does not broadcast.
    """

    def __init__(self, a: Shape, b: Shape):
        self._a, self._b = tuple(a), tuple(b)
        self._out = _broadcast(self._a, self._b)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_tenmul,
            DIMS,
            dims(self._out),
            dims(self._a),
            dims(self._b),
        )
        return _built(ptr, (self._a, self._b), (self._out,))

    def _bundle(self):
        from bartorch.nlop._bundle import of_product

        return of_product(self, self._out, self._a, self._b)

    def __repr__(self) -> str:
        return f"Multiply({self._a}, {self._b})"


class Divide(NonlinearOperator):
    """Elementwise quotient of two inputs, BART's ``zdiv``.

    Input 0 is the dividend and input 1 the divisor.

    Parameters
    ----------
    shape : tuple of int
        Shape of both inputs and of the output, C order.
    eps : float, default=0.0
        Positive multiplies by the regularized inverse of :class:`Inverse`
        instead of dividing.
    """

    def __init__(self, shape: Shape, eps: float = 0.0):
        self._shape = tuple(shape)
        self.eps = float(eps)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(library().bartorch_nlop_zdiv, DIMS, dims(self._shape), self.eps)
        return _built(ptr, (self._shape, self._shape), (self._shape,))

    def _composition(self):
        """``nlop_zdiv_reg_create``: the divisor inverted, then multiplied in."""
        from bartorch.nlop._base import _chain

        return _chain(
            Inverse(self._shape, self.eps),
            Multiply(self._shape, self._shape),
            output=0,
            input=1,
        )

    def _bundle(self):
        from bartorch.nlop._bundle import of_composition

        return of_composition(self, self._composition())

    def __repr__(self) -> str:
        return f"Divide({self._shape}, eps={self.eps})"


class Weighted(NonlinearOperator):
    """Weighted sum ``a * x + b * z`` of two inputs, BART's ``zaxpbz``.

    Parameters
    ----------
    shape : tuple of int
        Shape of both inputs and of the output, C order.
    a, b : float, default=1.0
        Real weights of input 0 (``x``) and input 1 (``z``).
    """

    def __init__(self, shape: Shape, a: float = 1.0, b: float = 1.0):
        self._shape = tuple(shape)
        self.a, self.b = float(a), float(b)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_zaxpbz, DIMS, dims(self._shape), self.a, self.b
        )
        return _built(ptr, (self._shape, self._shape), (self._shape,))

    def _bundle(self):
        from bartorch.nlop._base import _combine
        from bartorch.nlop._bundle import linear, scaled

        # `a` and `b` are real, so the adjoint scales by them rather than by
        # their conjugates.
        adjoint = _combine(scaled(self._shape, self.a), scaled(self._shape, self.b))._dup(0, 1)
        return linear(self, Weighted(self._shape, self.a, self.b), adjoint)

    def __repr__(self) -> str:
        return f"Weighted({self._shape}, a={self.a}, b={self.b})"


def Phase(shape: Shape) -> NonlinearOperator:  # noqa: N802
    """Elementwise ``x / |x|``, BART's ``zphsr``.

    Built as :class:`Abs` chained into the divisor of :class:`Divide`, with
    the two inputs tied: BART's ``nlop_zphsr_create`` calls
    ``nlop_dup(x, 0, 0)``, which fails BART's own ``a < b`` assertion.

    Parameters
    ----------
    shape : tuple of int
        Domain and codomain, C order.

    Returns
    -------
    NonlinearOperator
        One input and one output of ``shape``.
    """
    shape = tuple(shape)
    from bartorch.nlop._base import _chain

    return _chain(Abs(shape), Divide(shape), output=0, input=1)._dup(0, 1)


class Constant(NonlinearOperator):
    """Operator of no inputs returning a fixed tensor, BART's ``nlop_const``.

    :meth:`~bartorch.nlop.NonlinearOperator.partial` fixes an input of another
    operator to a constant in one step.

    Parameters
    ----------
    value : torch.Tensor
        The output.
    shape : tuple of int, default=None
        Output shape, C order; ``value``'s shape by default.
    """

    def __init__(self, value: torch.Tensor, shape: Shape | None = None):
        self.value = as_operand(value, tuple(shape) if shape is not None else value.shape, "value")
        super().__init__()

    def _create(self) -> Built:
        shape = tuple(self.value.shape)
        ptr = self._under_lock(
            library().bartorch_nlop_const,
            DIMS,
            dims(shape),
            self.value.data_ptr(),
        )
        return _built(ptr, (), (shape,))

    def _bundle(self):
        from bartorch.nlop._bundle import Bundle

        # No input, so no tangent to carry and no cotangent to return.
        return Bundle(self, Constant(torch.zeros_like(self.value)), None)

    def __repr__(self) -> str:
        return f"Constant({tuple(self.value.shape)})"
