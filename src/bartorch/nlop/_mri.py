"""Nonlinear MRI encoding operators: the image and the coils as two unknowns.

:class:`NonlinearSense` is BART's own model, from ``noir/model2.c``, built by
the same code ``nlinv`` runs::

    kspace = A[ (mask * image) * ifftuc(weights * ksens) ]

:func:`CartesianSense` and :func:`NoncartesianSense` name its two cases the
way the linear operators do.  :class:`CoilSense` is the general recipe
instead: any linear encoding from coil images to data, with a product of two
unknowns in front of it.
"""

from __future__ import annotations

import torch

from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, _Handle, as_operand, dims
from bartorch.linop._base import LinearOperator
from bartorch.linop._mri import _spatial
from bartorch.nlop._base import NonlinearOperator, _built, _chain
from bartorch.nlop._basic import Multiply

__all__ = ["CartesianSense", "CoilSense", "NoncartesianSense", "NonlinearSense"]

#: BART's three spatial axes, as a flag word.
_FFT_FLAGS = 7

#: What ``bartorch_noir_dims`` calls each shape.
_SHAPES = {
    "kspace": 0,
    "coil_images": 1,
    "image": 2,
    "coils": 3,
    "coils_as_multiplied": 4,
    "pattern": 5,
    "trajectory": 6,
}


def _tensor(x, what: str):
    return None if x is None else as_operand(x, tuple(x.shape), what)


def _optional(x):
    """``(dims, pointer)`` for a tensor BART may be given or not."""
    return (None, None) if x is None else (dims(tuple(x.shape)), x.data_ptr())


class _Part(LinearOperator):
    """One of the linear operators the model is made of, held against its lifetime."""

    def __init__(self, fn, noir, ishape: Shape, oshape: Shape, device):
        self.fn, self.noir = fn, noir
        self._ishape, self._oshape = tuple(ishape), tuple(oshape)
        self._device = device
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(self.fn, self.noir.ptr, device=self._device)
        return Built(ptr, self._ishape, self._oshape, keep=(self.noir,), device=self._device)


class NonlinearSense(NonlinearOperator):
    r"""Joint image and coil-sensitivity forward model, BART's ``noir``.

    Two inputs, the image and the coil coefficients, and one output.  The coil
    input is not the sensitivity maps but a k-space representation
    :math:`\hat{s}` of them, mapped to the maps by the Sobolev weighting

    .. math::

        S = \mathcal{F}^{-1} \left[ \, c \,
            (1 + a |k|^2)^{-b/2} \, \hat{s} \, \right]

    (BART's ``noir_calc_weights``), so the coils are smooth by construction
    and a Gauss-Newton step needs no separate penalty on them.  :attr:`coils`
    maps fitted coefficients to sensitivities.

    On a grid the output is k-space.  Off the grid, as BART builds the model,
    the output is the gridded coil images, ``nufft^H`` of k-space;
    :meth:`prepare` brings a measurement to that form.

    Parameters
    ----------
    image_shape : tuple of int
        Coil-image shape ``(coils, *spatial)``, C order, as the linear
        encodings take it.  The image input is this shape with one coil.
    pattern : torch.Tensor, default=None
        Sampling mask on the grid, one at acquired positions; ``None`` on a
        grid is all ones.  Off the grid it is passed to BART as is and may not
        be combined with ``basis``.
    trajectory : torch.Tensor, default=None
        Trajectory ``(..., samples, 3)`` in grid units, as
        :func:`bartorch.tools.traj` produces; selects the non-Cartesian model.
    kspace_shape : tuple of int, default=None
        Sample shape.  Default: ``image_shape`` on a grid,
        ``(coils, *trajectory.shape[:-1], 1)`` off it.
    coil_shape, coefficient_shape : tuple of int, default=None
        Shapes of the sensitivities and of their coefficients; both default to
        ``image_shape``.
    weights : torch.Tensor, default=None
        k-space diagonal applied to the samples off the grid; its conjugate is
        applied in the adjoint.
    basis : torch.Tensor, default=None
        Temporal subspace basis ``(coefficients, frames)`` over the last
        encoding axis; off the grid only.
    mask : torch.Tensor, default=None
        Support the image is restricted to.
    sobolev : tuple of float, default=(220.0, 32.0)
        ``(a, b)`` of the coil weighting; BART's defaults.
    c : float, default=1.0
        Scale of the coil weighting.
    real : bool, default=False
        Constrain the image to be real.
    sos : bool, default=False
        BART's sum-of-squares variant of the coil weighting.
    oversampling_coils : float, default=None
        Grid oversampling of the coil coefficients; ``None`` takes ``nlinv``'s
        choice, 2 off the grid and 1 on it.
    oversampled_coils : bool, default=False
        Return the sensitivities on the oversampled grid rather than the
        image's.
    toeplitz : bool, default=True
        Off the grid, apply the normal operator as a convolution with a point
        spread function rather than as a NUFFT and its adjoint.
    optimized : bool, default=False
        Off the grid, BART's ``noir2_noncart_optimized_create``.

    Raises
    ------
    ValueError
        If ``basis`` is given on a grid, or with ``pattern`` off the grid.

    Examples
    --------
    >>> F = NonlinearSense((8, 128, 128), pattern=mask)
    >>> F.ishapes
    ((1, 128, 128), (8, 128, 128))
    >>> image, coefficients = nlop.IRGNM()(kspace, F)
    """

    def __init__(
        self,
        image_shape: Shape,
        pattern: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        *,
        kspace_shape: Shape | None = None,
        coil_shape: Shape | None = None,
        coefficient_shape: Shape | None = None,
        weights: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sobolev: tuple[float, float] = (220.0, 32.0),
        c: float = 1.0,
        real: bool = False,
        sos: bool = False,
        oversampling_coils: float | None = None,
        oversampled_coils: bool = False,
        toeplitz: bool = True,
        optimized: bool = False,
    ):
        self.image_shape = tuple(image_shape)
        # BART counts a coil on its fourth axis and a slice on its third, so a
        # two-dimensional problem has to have that third spatial axis written
        # out before the coils land where the model expects them.  This is the
        # same expansion the linear encodings do.
        coils, spatial = _spatial(self.image_shape)
        self.coil_image_shape = (coils, *spatial)

        self.noncart = trajectory is not None
        if pattern is None and not self.noncart:
            # BART's model always carries one; `nlinv` estimates it from the
            # measurement, and a fully sampled acquisition is all ones.
            pattern = torch.ones((1, *spatial), dtype=torch.complex64)
        self.pattern = _tensor(pattern, "pattern")
        self.trajectory = _tensor(trajectory, "trajectory")
        self.weights = _tensor(weights, "weights")
        self.basis = _tensor(basis, "basis")
        self.mask = _tensor(mask, "mask")

        if self.noncart and self.basis is not None and self.pattern is not None:
            raise ValueError("off the grid the pattern goes into the weights, not beside them")
        if not self.noncart and self.basis is not None:
            raise ValueError(
                "BART's Cartesian noir model refuses a basis; build the subspace encoding "
                "with CoilSense over a linear operator that carries it"
            )

        self.coil_shape = tuple(coil_shape) if coil_shape is not None else self.coil_image_shape
        self.coefficient_shape = (
            tuple(coefficient_shape) if coefficient_shape is not None else self.coil_shape
        )

        if kspace_shape is not None:
            self.kspace_shape = tuple(kspace_shape)
        elif self.noncart:
            # BART's noir model reads the samples in BART's order, with the
            # coils in front of the trajectory's axes and the coordinate axis a
            # singleton: `(coils, *trajectory[:-1], 1)`.
            self.kspace_shape = (
                self.coil_image_shape[0],
                *tuple(self.trajectory.shape)[:-1],
                1,
            )
        else:
            self.kspace_shape = self.coil_image_shape

        self.sobolev = (float(sobolev[0]), float(sobolev[1]))
        self.c = float(c)
        self.real = bool(real)
        self.sos = bool(sos)
        # `nlinv` fits the coils on a doubled grid off the grid and on the
        # image's own on it; None asks for whichever BART would use.
        self.oversampling_coils = (
            (2.0 if self.noncart else 1.0)
            if oversampling_coils is None
            else float(oversampling_coils)
        )
        self.oversampled_coils = bool(oversampled_coils)
        self.toeplitz = bool(toeplitz)
        self.optimized = bool(optimized)
        super().__init__()

    # --- construction ------------------------------------------------------

    def _image_only(self) -> Shape:
        """The coil-image shape with one coil, which is where the image lives."""
        return (1,) + self.coil_image_shape[1:]

    def _create(self) -> Built:
        _ensure_ready()
        lib = library()
        pat_dims, pattern = _optional(self.pattern)
        trj_dims, traj = _optional(self.trajectory)
        wgh_dims, weights = _optional(self.weights)
        bas_dims, basis = _optional(self.basis)
        msk_dims, mask = _optional(self.mask)
        a, b = self.sobolev
        device = None
        for t in (self.trajectory, self.pattern, self.weights, self.mask):
            if t is not None:
                device = t.device
                break
        with _lock, _on_device(device or torch.device("cpu")):
            ptr = lib.bartorch_noir_create(
                DIMS,
                dims(self.kspace_shape),
                dims(self.coil_image_shape),
                dims(self._image_only()),
                dims(self.coefficient_shape),
                dims(self.coil_shape),
                pat_dims,
                pattern,
                trj_dims,
                traj,
                wgh_dims,
                weights,
                bas_dims,
                basis,
                msk_dims,
                mask,
                int(self.noncart),
                int(self.optimized),
                int(self.toeplitz),
                _FFT_FLAGS,
                _FFT_FLAGS,
                int(self.real),
                int(self.sos),
                a,
                b,
                self.c,
                self.oversampling_coils,
                int(self.oversampled_coils),
            )
        keep = tuple(
            t
            for t in (self.pattern, self.trajectory, self.weights, self.basis, self.mask)
            if t is not None
        )
        self._noir = _Handle(ptr, lib.bartorch_noir_free, keep)
        self.device = device

        model = self._under_lock(lib.bartorch_noir_model, self._noir.ptr, device=device)
        # BART settles the shapes itself -- the coils may be oversampled, and
        # off the grid the model returns gridded coil images -- so they are
        # read back rather than assumed.
        data_shape = self._read(_SHAPES["coil_images"] if self.noncart else _SHAPES["kspace"])
        return _built(
            model,
            (self._image_only(), self.coefficient_shape),
            (data_shape,),
            keep=(self._noir,),
            device=device,
        )

    def _composition(self) -> NonlinearOperator:
        """The model written out: the two unknowns' linear parts, their product, the transform.

        ``noir2_join`` (``model2.c:164``) is these four operators, and off the
        grid its last stage is asymmetric -- see
        :class:`~bartorch.nlop._bundle.Asymmetric`.
        """
        from bartorch.linop._basic import Identity
        from bartorch.nlop._bundle import Asymmetric

        image, coils, transform = self.image, self.coils, self.transform
        product = Multiply(image.oshape, coils.oshape)
        made = _chain(coils.to_nonlinear(), product, output=0, input=1)
        made = _chain(image.to_nonlinear(), made, output=0, input=0)
        # The chain leaves the coil coefficients in front of the image.
        made = made._permute_inputs([1, 0])
        last = (
            Asymmetric(transform.gram(), Identity(product.oshape), source=transform)
            if self.noncart
            else transform.to_nonlinear()
        )
        return _chain(made, last, output=0, input=0)

    def _bundle(self):
        from bartorch.nlop._bundle import of_composition

        return of_composition(self, self._composition())

    def _read(self, which: int) -> Shape:
        """One of BART's dimension vectors, as a C-order shape.

        BART answers in all sixteen axes; what comes back has the rank of the
        image, or more when BART put something on an axis beyond it.
        """
        from bartorch import _marshal

        vector = _marshal.dim_vector()
        if library().bartorch_noir_dims(self._noir.ptr, which, DIMS, vector) < 0:
            raise BartError("BART would not report the model's dimensions")
        bart = [int(vector[i]) for i in range(DIMS)]
        used = [axis for axis, size in enumerate(bart) if 1 != size]
        rank = max(len(self.coil_image_shape), 1 + used[-1] if used else 1)
        return tuple(bart[:rank][::-1])

    # --- what the model is made of -----------------------------------------

    def _linop(self, fn, ishape: Shape, oshape: Shape):
        return _Part(fn, self._noir, ishape, oshape, self.device)

    @property
    def coils(self):
        """Linear operator from coil coefficients to sensitivity maps.

        The Sobolev weighting followed by the inverse Fourier transform.
        """
        return self._linop(
            library().bartorch_noir_coils, self.ishapes[1], self._read(_SHAPES["coils"])
        )

    @property
    def image(self):
        """Linear operator applied to the image input: support mask and real constraint."""
        shape = self.ishapes[0]
        return self._linop(library().bartorch_noir_image, shape, shape)

    @property
    def data(self):
        """Linear operator from the model's output to k-space.

        The NUFFT off the grid, whose adjoint is :meth:`prepare`; the identity
        on a grid.
        """
        return self._linop(library().bartorch_noir_data, self.oshapes[0], self.kspace_shape)

    @property
    def transform(self):
        """Linear operator from coil images to k-space, the encoding after the product."""
        return self._linop(
            library().bartorch_noir_transform,
            self._read(_SHAPES["coil_images"]),
            self.kspace_shape,
        )

    def prepare(self, kspace: torch.Tensor) -> torch.Tensor:
        """Bring a measurement to the model's output.

        Off the grid this is ``nufft^H(kspace)``, as ``noir2_recon`` applies it
        before the first Gauss-Newton step; on a grid ``kspace`` is returned
        unchanged.

        Parameters
        ----------
        kspace : torch.Tensor
            Measurement of :attr:`kspace_shape`.

        Returns
        -------
        torch.Tensor
            Tensor of :attr:`oshape`.
        """
        if not self.noncart:
            return as_operand(kspace, self.kspace_shape, "kspace")
        return self.data.adjoint(kspace)

    def __repr__(self) -> str:
        where = "noncartesian" if self.noncart else "cartesian"
        return f"NonlinearSense({self.image_shape}, {where})"


def CartesianSense(  # noqa: N802  (it is a constructor)
    image_shape: Shape,
    pattern: torch.Tensor | None = None,
    **kwargs,
) -> NonlinearSense:
    """Cartesian joint image and coil-sensitivity forward model.

    :class:`NonlinearSense` on a grid, the model ``nlinv`` inverts without a
    trajectory: the nonlinear counterpart of
    :func:`bartorch.linop.CartesianSense`, with the sensitivities a second
    unknown.

    Parameters
    ----------
    image_shape : tuple of int
        Coil-image shape ``(coils, *spatial)``, C order.
    pattern : torch.Tensor, default=None
        Sampling mask; ``None`` is fully sampled.
    **kwargs
        Keyword arguments of :class:`NonlinearSense`, except ``trajectory``.

    Returns
    -------
    NonlinearSense
        Inputs: image, coil coefficients.  Output: k-space.

    Raises
    ------
    ValueError
        If ``trajectory`` is given.
    """
    if kwargs.get("trajectory") is not None:
        raise ValueError("a trajectory makes the non-Cartesian model; use NoncartesianSense")
    return NonlinearSense(image_shape, pattern=pattern, **kwargs)


def NoncartesianSense(  # noqa: N802  (it is a constructor)
    trajectory: torch.Tensor,
    image_shape: Shape,
    **kwargs,
) -> NonlinearSense:
    """Non-Cartesian joint image and coil-sensitivity forward model.

    :class:`NonlinearSense` along a trajectory, the model ``nlinv`` inverts
    off the grid: the nonlinear counterpart of
    :class:`bartorch.linop.NoncartesianSense`.  The output is the gridded coil
    images; :meth:`NonlinearSense.prepare` brings a measurement to that form.

    Parameters
    ----------
    trajectory : torch.Tensor
        Trajectory ``(..., samples, 3)`` in grid units.
    image_shape : tuple of int
        Coil-image shape ``(coils, *spatial)``, C order.
    **kwargs
        Keyword arguments of :class:`NonlinearSense`.

    Returns
    -------
    NonlinearSense
        Inputs: image, coil coefficients.  Output: gridded coil images.
    """
    return NonlinearSense(image_shape, trajectory=trajectory, **kwargs)


def CoilSense(  # noqa: N802  (it is a constructor)
    encoding,
    image_shape: Shape | None = None,
    coil_shape: Shape | None = None,
    *,
    items: bool = False,
) -> NonlinearOperator:
    """Product of an image and unknown coil sensitivities before a linear encoding.

    The model ``y = E(x * s)`` with image ``x`` and sensitivities ``s`` both
    unknown, for any linear ``E`` from coil images to data: a wave encoding,
    a field-corrected or subspace encoding, or any other.  Unlike
    :class:`NonlinearSense` the sensitivities themselves are the unknown, with
    no Sobolev weighting; smoothness is imposed by chaining a smoothing
    operator onto the coil input, if at all.

    Parameters
    ----------
    encoding : LinearOperator
        Encoding from coil images, its domain, to data.
    image_shape : tuple of int, default=None
        Image shape; the encoding's domain with one coil.
    coil_shape : tuple of int, default=None
        Sensitivity shape; the encoding's domain.
    items : bool, default=False
        The encoding's leading axis holds independent items, each with its own
        image and coils, and the coil axis is the next one.  A Gauss-Newton
        step applies the model to all items at once and solves each item's
        inner problem on its own.

    Returns
    -------
    NonlinearOperator
        Inputs: image, sensitivities.  Output: data.

    Raises
    ------
    ValueError
        If ``items`` is set and the image, coil and encoding shapes do not
        share their leading size.

    Examples
    --------
    >>> E = linop.Diagonal(pattern, shape) @ linop.FFT(shape, axes=(-1, -2))
    >>> F = CoilSense(E)
    >>> F.ishapes
    ((1, 128, 128), (8, 128, 128))
    """
    cim = tuple(encoding.ishape)
    lead = cim[:1] if items else ()
    body = cim[1:] if items else cim
    image_shape = tuple(image_shape) if image_shape is not None else lead + (1,) + body[1:]
    coil_shape = tuple(coil_shape) if coil_shape is not None else cim
    if items and not (image_shape[0] == coil_shape[0] == cim[0]):
        raise ValueError(
            f"items lead every shape: the image {image_shape}, the coils {coil_shape} and "
            f"the encoding's domain {cim} begin with different counts"
        )
    made = _chain(Multiply(image_shape, coil_shape), encoding.to_nonlinear(), output=0, input=0)
    if items:
        made.items = cim[0]
    return made
