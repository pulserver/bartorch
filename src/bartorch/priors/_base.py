"""The regularizer base class."""

from __future__ import annotations

import abc
import weakref
from collections.abc import Iterable

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import DIMS, library
from bartorch._operator import as_operand, axes_flags

__all__ = ["Regularizer"]


class Regularizer(abc.ABC):
    r"""Regularization term :math:`g(G x)` whose proximal operator and transform BART builds.

    :math:`G` is a linear transform and :math:`g` a functional on its codomain,
    both from BART's ``opt_reg_configure``.  :math:`G` is the identity for a
    term on the image itself and for a term whose transform is inside its
    proximal operator, such as :class:`~bartorch.priors.Wavelet`; it is a
    proper operator for a term such as :class:`~bartorch.priors.TotalVariation`,
    whose functional acts on finite differences.  Only the alternating-direction
    and primal-dual iterations apply :math:`G`; see
    :meth:`transform_is_identity`.

    Both are built per image shape and cached, so solving twice with the same
    term builds nothing the second time.

    Attributes
    ----------
    kind : str
        BART's letter for the term, as ``-R`` takes it.
    weight : float
        The term's weight :math:`\lambda`, relative to data divided by
        :func:`~bartorch.optim.data_scaling`.
    axes : tuple of int
        Axes the transform acts over, as indices into the image's shape.
    joint_axes : tuple of int
        Axes along which entries are thresholded jointly, as one :math:`\ell_2`
        norm.
    count : int
        For an NIHT term, the percentage of entries kept; zero otherwise.

    Examples
    --------
    >>> term = priors.Wavelet((-1, -2), 0.01)
    >>> z = term.prox(x, 0.5)                    # BART's proximal operator, step 0.5
    >>> term.transform_is_identity((64, 64))     # G = I: every proximal solver takes it
    True
    >>> priors.TotalVariation((-1, -2), 0.01).transform_is_identity((64, 64))
    False
    """

    kind: str = ""
    weight: float = 0.0
    axes: tuple[int, ...] = ()
    joint_axes: tuple[int, ...] = ()
    count: int = 0
    #: Whether the term extends the optimization variable with auxiliary ones,
    #: which BART counts
    #: across the whole set of terms, so that it cannot be built alone.
    _extends: bool = False

    def build(self, shape: tuple[int, ...], item: int = 0) -> int:
        """Handle of BART's operator for this term over an image of C-order ``shape``.

        Built on first use for each shape and cached.  A term that draws random
        shifts is built once per batch ``item``, so that each item draws the
        sequence it would draw alone.  The handle is owned by the term and freed
        with it.

        Parameters
        ----------
        shape : tuple of int
            Image shape.
        item : int, default=0
            Batch item.

        Returns
        -------
        int
            The handle.

        Raises
        ------
        ~bartorch.BartError
            BART could not build the term over ``shape``.
        """
        shape = tuple(shape)
        if not hasattr(self, "_handles"):
            self._handles = {}
        key = shape if 0 == item or 1 != self._options()[2] else (shape, int(item))
        if key in self._handles:
            return self._handles[key]

        _ensure_ready()
        xflags, jflags = self._flags(len(shape))
        block, family, shift_mode = self._options()
        out = _marshal.out_pointer()
        with _lock:
            code = library().bartorch_prox_create(
                self.kind.encode(),
                xflags,
                jflags,
                float(self.weight),
                int(self.count),
                int(block),
                family.encode(),
                shift_mode,
                _marshal.padded_dims(shape),
                _marshal.by_reference(out),
            )
        if code != 0:
            said = library().bartorch_solve_error(code).decode(errors="replace")
            raise BartError(f"{self!r} could not be built: {said}")

        handle = out.value
        self._handles[key] = handle
        weakref.finalize(self, _release, handle)
        return handle

    def prox_shape(self, image_shape: tuple[int, ...]) -> tuple[int, ...]:
        """C-order shape of the domain of the proximal operator, for an image of ``image_shape``.

        The image's shape when :math:`G` is the identity, wavelet terms included;
        otherwise the codomain of :math:`G`, which for total variation carries the
        gradient components on an axis of their own in front.

        Parameters
        ----------
        image_shape : tuple of int
            Image shape.

        Returns
        -------
        tuple of int
        """
        image_shape = tuple(image_shape)
        handle = self.build(image_shape)
        out = _marshal.wide_dim_vector()

        with _lock:
            rank = library().bartorch_prox_domain(handle, len(out), out)
        if rank < 0:
            raise BartError(f"{self!r} would not say what it works on (code {rank})")

        shape = [int(out[i]) for i in range(rank)][::-1]
        while len(shape) > len(image_shape) and 1 == shape[0]:
            shape.pop(0)
        return tuple(shape)

    def prox(
        self,
        x: torch.Tensor,
        gamma: float = 1.0,
        *,
        image_shape: tuple[int, ...] | None = None,
        item: int = 0,
    ) -> torch.Tensor:
        r"""Apply the proximal operator :math:`\operatorname{prox}_{\gamma g}` as BART's solvers do.

        Parameters
        ----------
        x : tensor
            Of :meth:`prox_shape`, the image's shape for most terms.
        gamma : float, default=1.0
            Step :math:`\gamma`; the term's weight is already in :math:`g`.
        image_shape : tuple of int, default=None
            Image shape the term is configured for, when it differs from
            ``x.shape``, as for total variation; ``x.shape`` when omitted.
        item : int, default=0
            Batch item ``x`` is; a term that draws random shifts keeps a generator
            per item.

        Returns
        -------
        torch.Tensor
            Complex64, of ``x``'s shape.

        Raises
        ------
        ValueError
            ``x`` requires a gradient: BART's proximal operators have no backward
            pass.  Use :class:`~bartorch.priors.ImplicitPrior` for a differentiable
            proximal step, or :meth:`detach` to hold this one fixed.  Also raised
            for an ``x`` not of :meth:`prox_shape`.
        ~bartorch.BartError
            BART could not apply the operator.

        Notes
        -----
        This is the proximal operator alone; a solver applying the term computes
        it on :math:`G x` (:meth:`apply_transform`).  BART's IST and FISTA apply
        it to the image and ignore :math:`G`, so they take only terms whose
        transform is the identity.

        Examples
        --------
        >>> priors.Wavelet((-1, -2), 0.01).prox(image, gamma=0.95)
        """
        from bartorch.linop._base import _tracking

        if _tracking(x):
            raise ValueError(
                f"{self!r} is one of BART's proximal operators and has no backward pass: "
                "`operator_p_fun_t` is (data, mu, dst, src), with nowhere to carry one, "
                "so an iteration containing this term cannot be differentiated.  Use a "
                "denoiser in place of the term -- `optim.admm(y, A, denoiser)` differentiates "
                "end to end -- or `term.detach()` to hold this one fixed in the graph"
            )

        image_shape = tuple(x.shape) if image_shape is None else tuple(image_shape)
        shape = self.prox_shape(image_shape)
        if tuple(x.shape) != shape:
            raise ValueError(
                f"over an image of {image_shape}, {self!r} works on {shape}, not {tuple(x.shape)}"
            )

        handle = self.build(image_shape, item)
        src = as_operand(x, shape, "x")
        out = torch.empty_like(src)

        with _lock, _on_device(src.device):
            code = library().bartorch_prox_apply(
                handle, float(gamma), out.data_ptr(), src.data_ptr()
            )
        if code != 0:
            said = library().bartorch_solve_error(code).decode(errors="replace")
            raise BartError(f"{self!r} could not be applied: {said}")
        return out

    def apply_transform(
        self, x: torch.Tensor, image_shape: tuple[int, ...] | None = None, mode: str = "forward"
    ) -> torch.Tensor:
        """Apply :math:`G`, its adjoint or :math:`G^H G` to ``x``.

        Works at any rank, including total variation's, whose codomain exceeds
        BART's 16 dimensions and so has no :meth:`transform` operator.

        Parameters
        ----------
        x : tensor
            The image for ``"forward"`` and ``"normal"``, of :meth:`prox_shape` for
            ``"adjoint"``.
        image_shape : tuple of int, default=None
            Image shape the term is configured for; ``x.shape`` when omitted, which
            holds for ``"forward"`` and ``"normal"``.
        mode : {"forward", "adjoint", "normal"}, default="forward"
            :math:`G`, :math:`G^H` or :math:`G^H G`.

        Returns
        -------
        torch.Tensor
            Complex64.

        Raises
        ------
        ValueError
            An unknown ``mode``, or ``x`` not of the shape ``mode`` takes.

        Notes
        -----
        Recorded for autograd when ``x`` requires a gradient: the backward pass of
        ``"forward"`` is ``"adjoint"`` and vice versa, and ``"normal"`` is its own.
        """
        from bartorch.linop._base import _tracking

        if _tracking(x):
            from bartorch.priors._autograd import apply_transform

            return apply_transform(
                self, x, tuple(image_shape if image_shape is not None else x.shape), mode
            )
        return self._transform_apply(x, image_shape, mode)

    def _transform_apply(
        self, x: torch.Tensor, image_shape: tuple[int, ...] | None = None, mode: str = "forward"
    ) -> torch.Tensor:
        """:meth:`apply_transform`, without recording for autograd."""
        modes = {"forward": 0, "adjoint": 1, "normal": 2}
        if mode not in modes:
            raise ValueError(f"mode is forward, adjoint or normal, not {mode!r}")

        if image_shape is None:
            image_shape = tuple(x.shape)
        image_shape = tuple(image_shape)

        # forward: image -> the proximal operator's domain.  adjoint: back.
        # normal: image to image, which is the pair of them.
        transformed = self.prox_shape(image_shape)
        want, oshape = {
            "forward": (image_shape, transformed),
            "adjoint": (transformed, image_shape),
            "normal": (image_shape, image_shape),
        }[mode]
        if tuple(x.shape) != want:
            raise ValueError(f"the {mode} takes {want}, not {tuple(x.shape)}")

        handle = self.build(image_shape)
        src = as_operand(x, want, "x")
        out = torch.empty(oshape, dtype=torch.complex64, device=src.device)

        with _lock, _on_device(src.device):
            code = library().bartorch_prox_transform_apply(
                handle, modes[mode], out.data_ptr(), src.data_ptr()
            )
        if code != 0:
            raise BartError(f"{self!r} could not apply its transform ({mode})")
        return out

    def transform(self, image_shape: tuple[int, ...]):
        """The transform :math:`G` as a :class:`~bartorch.linop.LinearOperator`.

        The identity for a term whose transform is inside its proximal operator.
        The shapes do not tell the two arrangements apart: the Laplace term's
        transform is a convolution whose codomain has the image's shape.

        Parameters
        ----------
        image_shape : tuple of int
            Image shape.

        Returns
        -------
        LinearOperator
            :math:`G`, from the image to :meth:`prox_shape`.

        Raises
        ------
        ~bartorch.BartError
            The transform's rank exceeds BART's 16 dimensions, as for total
            variation; :meth:`apply_transform` applies it instead.
        """
        handle = self.build(tuple(image_shape))
        lib = library()
        with _lock:
            ptr = lib.bartorch_prox_transform(handle)
        if not ptr:
            raise BartError(f"{self!r} has no transform to give")

        try:
            ishape = _handle_shape(lib.bartorch_linop_domain, ptr, len(image_shape))
            oshape = _handle_shape(lib.bartorch_linop_codomain, ptr, len(image_shape))
        except BartError:
            lib.bartorch_linop_free(ptr)
            raise
        return _Transform(ptr, ishape, oshape)

    def rewind(self, image_shape: tuple[int, ...]) -> None:
        """Reset the term's own random generator, for an image of ``image_shape``.

        A wavelet or locally low-rank term shifts its transform by a draw from a
        generator seeded when BART builds the operator.  ``pics`` builds a fresh
        operator per run; every solver here rewinds its terms before it iterates,
        so a reused term draws as the command does.  A term without a generator
        is unaffected.

        Parameters
        ----------
        image_shape : tuple of int
            Image shape.
        """
        image_shape = tuple(image_shape)
        self.build(image_shape)
        handles = [
            handle
            for key, handle in self._handles.items()
            if key == image_shape or (2 == len(key) and key[0] == image_shape)
        ]
        for handle in handles:
            with _lock:
                code = library().bartorch_prox_rewind(handle)
            if code != 0:
                raise BartError(f"{self!r} could not be rewound")

    def transform_is_identity(self, image_shape: tuple[int, ...]) -> bool:
        """Whether :math:`G` is the identity, as BART's ``linop_is_identity`` decides.

        Decided on the operator, not from shapes: the Laplace term's transform is
        a convolution on the image's own shape.  ``iter2_chambolle_pock`` asks this
        of the first term to choose it as the primal proximal step.

        Parameters
        ----------
        image_shape : tuple of int
            Image shape.

        Returns
        -------
        bool
            True when :math:`G` is the identity.
        """
        handle = self.build(tuple(image_shape))
        with _lock:
            answer = library().bartorch_prox_transform_is_identity(handle)
        if answer < 0:
            raise BartError(f"{self!r} could not be asked about its transform")
        return 1 == answer

    def _options(self) -> tuple[int, str, int]:
        """Block size, wavelet family and shift mode for ``opt_reg_configure``.

        Only the wavelet and locally low-rank terms read them.  Shift mode 0
        is no shift, 1 a random cycle spin between solves, 2 fully overlapping
        blocks.
        """
        return 8, "dau2", 1

    def _settings(self) -> dict[str, object]:
        """The settings a command line gives once for all its terms, as this term needs them.

        Keys are those of :data:`_SHARED_DEFAULTS`.
        """
        return {}

    def _check(self, ndim: int | None) -> None:
        """Raise if this term cannot be built over an ``ndim``-axis image.

        BART states some of a term's preconditions as assertions, which end
        the process rather than return; and a term with auxiliary variables is
        built by `opt_reg_configure`, deep inside the solve, where there is
        nothing left to catch.  Whatever can be decided from the rank is
        decided here instead, while an exception still reaches the caller.
        Most terms have nothing to say and this does nothing.
        """

    def _flags(self, ndim: int) -> tuple[int, int]:
        """BART's bitmasks for :attr:`axes` and :attr:`joint_axes`, for an ``ndim``-axis image."""
        return (
            axes_flags(self.axes, ndim) if self.axes else 0,
            axes_flags(self.joint_axes, ndim) if self.joint_axes else 0,
        )

    def _argument(self, ndim: int) -> str:
        """This term as a ``-R`` argument, for an ``ndim``-axis image."""
        x, j = self._flags(ndim)
        if self.kind == "Q":
            return f"Q:{self.weight!r}"
        if self.kind == "S":
            return "S"
        if self.kind in ("I", "R1", "R2"):
            return f"{self.kind}:{j}:{self.weight!r}"
        if self.kind in ("H", "N"):
            return f"{self.kind}:{x}:{j}:{self.count}"
        return f"{self.kind}:{x}:{j}:{self.weight!r}"

    def detach(self) -> Regularizer:
        """Return this term with its proximal step detached from the autograd graph.

        The detached term thresholds exactly as before, and the gradient obtained
        is the iteration's with this term held fixed: the use is a solve in which a
        differentiable denoiser occupies one term and a BART term another, and
        only the denoiser is trained.

        Returns
        -------
        Regularizer
            A view of this term whose :meth:`prox` detaches its input.

        Examples
        --------
        >>> optim.admm(y, A, [denoiser, priors.Wavelet((-1, -2), 0.01).detach()])
        """
        return _Frozen(self)

    def __repr__(self) -> str:
        parts = [f"weight={self.weight}"] if self.weight else []
        if self.axes:
            parts.insert(0, f"axes={self.axes}")
        if self.joint_axes:
            parts.append(f"joint_axes={self.joint_axes}")
        if self.count:
            parts.append(f"count={self.count}")
        return f"{type(self).__name__}({', '.join(parts)})"


def _handle_shape(query, ptr: int, min_ndim: int) -> tuple[int, ...]:
    """The C-order shape BART records for an operator handle.

    The query fills as many entries as it is given and returns the rank it
    has, so a rank past BART's sixteen is detected rather than truncated
    without notice.
    """
    vector = _marshal.dim_vector()
    rank = query(ptr, DIMS, vector)
    if rank > DIMS:
        raise BartError(
            f"this term's transform works at rank {rank}, past BART's {DIMS}, which an "
            "operator here cannot hold; total variation is the one that does -- its "
            "gradient puts the components on an axis of their own beyond the image's -- "
            "and its proximal operator is still reachable through prox()"
        )
    return tuple(_marshal.shape_from_dims(vector, min_ndim))


class _Transform:
    """The operator a term carries, from a handle BART has already built."""

    def __new__(cls, ptr: int, ishape, oshape):
        from bartorch._operator import Built
        from bartorch.linop._base import LinearOperator

        class Held(LinearOperator):
            def __init__(self):
                self._held = (ptr, tuple(ishape), tuple(oshape))
                super().__init__()

            def _create(self):
                return Built(self._held[0], self._held[1], self._held[2])

            def __repr__(self) -> str:
                return f"<term transform {self._held[1]} -> {self._held[2]}>"

        return Held()


#: BART's value for each setting a command line gives once for all its terms.
_SHARED_DEFAULTS: dict[str, object] = {
    "randshift": True,
    "overlapping": False,
    "block": 8,
    "family": "dau2",
    "alpha": (1.0, 3.0**0.5),
    "gamma": (1.0, 1.0),
}


def _as_terms(regularizers) -> list[Regularizer]:
    """``regularizers`` -- None, one term or an iterable of them -- as a list of terms."""
    if isinstance(regularizers, str):
        raise TypeError(
            f"a regularizer is a term from bartorch.priors, not the string {regularizers!r}; "
            "for instance `priors.Wavelet(axes=(-1, -2), weight=0.005)`"
        )
    if regularizers is None:
        return []
    if isinstance(regularizers, Regularizer):
        return [regularizers]
    terms = list(regularizers) if isinstance(regularizers, Iterable) else [regularizers]
    for term in terms:
        if not isinstance(term, Regularizer) and not _term_shaped(term):
            hint = "; a denoiser goes in priors.ImplicitPrior" if callable(term) else ""
            raise TypeError(f"a regularizer is a term from bartorch.priors, not {term!r}{hint}")
    return terms


def _term_shaped(thing) -> bool:
    """Whether something answers the four questions an iteration asks a term.

    :class:`~bartorch.priors.ImplicitPrior` does.  BART cannot take one, so a
    solver holding one has no library route.
    """
    return all(
        callable(getattr(thing, name, None))
        for name in ("prox", "prox_shape", "apply_transform", "rewind")
    )


def _command_line(
    terms: list[Regularizer], ndim: int | None, command: str, kinds: Iterable[str] | None = None
) -> tuple[list[str], dict[str, object]]:
    """``-R`` arguments for ``terms`` over an ``ndim``-axis image, and their shared settings.

    With ``ndim`` None only negative axes are accepted.  The settings returned
    are those that differ from BART's defaults.

    Raises
    ------
    TypeError
        ``command`` does not take one of the terms (``kinds`` lists those it takes).
    ValueError
        Two terms need different values of one shared setting.
    """
    arguments: list[str] = []
    shared: dict[str, object] = {}
    for term in terms:
        if kinds is not None and term.kind not in kinds:
            raise TypeError(f"{command} does not take {type(term).__name__} terms")
        term._check(ndim)
        arguments.append(term._argument(ndim))
        for name, value in term._settings().items():
            if shared.setdefault(name, value) != value:
                raise ValueError(
                    f"{command} sets {name} once for every term, "
                    f"and these terms ask for {shared[name]!r} and {value!r}"
                )
    return arguments, {k: v for k, v in shared.items() if v != _SHARED_DEFAULTS[k]}


def _release(handle: int) -> None:
    with _lock:
        library().bartorch_prox_free(handle)


class _Frozen:
    """A term whose proximal step is detached, making it a constant in the graph.

    Everything but :meth:`prox` is the wrapped term's own, and :meth:`prox`
    detaches its input first.  Registered as a :class:`Regularizer` because as
    far as BART is concerned it is one -- detaching changes no numbers -- so a
    solve containing it still has a library route and still reproduces the
    library's output bit for bit.
    """

    def __init__(self, term):
        self.term = term

    def __getattr__(self, name):
        # `term` itself is never delegated: asking for it before __init__ has
        # set it -- which is what unpickling does -- would recur forever.
        if "term" == name:
            raise AttributeError(name)
        return getattr(self.term, name)

    def prox(self, x, gamma: float = 1.0, *, image_shape=None, item: int = 0):
        return self.term.prox(x.detach(), gamma, image_shape=image_shape, item=item)

    def __repr__(self) -> str:
        return f"{self.term!r}.detach()"


Regularizer.register(_Frozen)


class _Penalty(Regularizer):
    """One penalty of a set BART configured together, over the image and the unknowns behind it.

    Its proximal operator detaches first when every term the set came from was
    detached.
    """

    kind = "penalty"

    def __init__(self, handle: int, shape: tuple[int, ...], frozen: bool):
        self._handles = {tuple(shape): handle}
        self._frozen = bool(frozen)
        weakref.finalize(self, _release, handle)

    def build(self, shape: tuple[int, ...], item: int = 0) -> int:
        shape = tuple(shape)
        if shape not in self._handles:
            raise ValueError(f"this penalty walks {next(iter(self._handles))}, not {shape}")
        return self._handles[shape]

    def prox(self, x, gamma: float = 1.0, *, image_shape=None, item: int = 0):
        return super().prox(x.detach() if self._frozen else x, gamma, image_shape=image_shape)

    def __repr__(self) -> str:
        return f"penalty over {next(iter(self._handles))}"
