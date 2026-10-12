"""Signal models from BlochSim, as BART nonlinear operators.

:class:`SignalModel` bridges BlochSim's
:class:`~blochsim.recon.ModelOperator` -- a model's value, its
Jacobian-vector product and its adjoint product, none of which builds a
Jacobian -- onto the three things BART's ``nlop_s`` asks for.
:func:`InversionRecovery`, :func:`MultiEcho` and :func:`Bloch` are the
quantitative models built on BlochSim's simulators, in BlochSim's
parameterisation.  The suite holds their curves and their fits against BART's
own signal models.

Notes
-----
BlochSim's parameter maps are real and stacked on the **last** axis.  BART
works in ``complex float`` on a C-order shape with the channels in front, so
the bridge moves the axis and carries the maps in the real part of a complex
buffer.  The imaginary half is an exact null direction of the derivative --
nothing reads it and the adjoint returns zero there -- so an iterate that
starts real stays real to the bit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from bartorch._operator import Shape
from bartorch.nlop._callback import _Callback

__all__ = ["Bloch", "SignalModel", "InversionRecovery", "MultiEcho"]


def _operator(acquisition, unknown, bounds, scale, amplitude, subspace):
    from blochsim.recon import ModelOperator

    return ModelOperator(
        acquisition,
        *unknown,
        bounds=bounds,
        scale=scale,
        amplitude=amplitude,
        subspace=subspace,
    )


class SignalModel(_Callback):
    """BlochSim signal model as a nonlinear operator from parameter maps to contrast images.

    The value, derivative and adjoint are BlochSim's
    :meth:`~blochsim.recon.ModelOperator.A`, ``A_jvp`` and ``A_vjp``.  The
    domain is ``(channels, *shape)``: the model's variables (bounded
    parameters in their transformed form, the amplitude as real and imaginary
    parts) in the real part of a complex tensor, in the order of
    :attr:`names`.  The codomain is ``(contrasts, *shape)``.
    :meth:`initial` and :meth:`split` convert from and to named maps in
    physical units.

    Parameters
    ----------
    model : blochsim.recon.ModelOperator
        Signal model with its unknowns, bounds and scales set.
    shape : tuple of int, default=()
        Voxel shape, C order.
    contrasts : int, default=None
        Number of contrast images; evaluated from the model when ``None``.

    Attributes
    ----------
    channels : int
        Number of channels of the domain.
    contrasts : int
        Number of contrast images of the codomain.

    Examples
    --------
    >>> from blochsim.recon import ModelOperator
    >>> from blochsim.simulators import MultiEchoSimulator
    >>> model = ModelOperator(
    ...     MultiEchoSimulator(TE=echo_times), "T2", bounds={"T2": (10.0, 300.0)}
    ... )
    >>> M = SignalModel(model, (128, 128))
    >>> M.ishape, M.oshape
    ((3, 128, 128), (8, 128, 128))
    >>> images = M(M.initial(T2=80.0))

    Under an encoding:

    >>> F = encoding @ M
    >>> maps = nlop.IRGNM()(kspace, F, x0=M.initial(T2=80.0))
    >>> M.split(maps)["T2"]
    """

    def __init__(self, model, shape: Shape = (), contrasts: int | None = None):
        self.model = model
        self.voxels = tuple(shape)
        self.channels = int(model.channels)
        if contrasts is None:
            contrasts = int(model.A(model.initial(())).shape[-1])
        self.contrasts = int(contrasts)
        # A model without a complex amplitude returns real images, and its
        # adjoint is taken against the real part of a complex cotangent.
        self._real: bool | None = None

        state: dict[str, torch.Tensor] = {}

        def forward(x: torch.Tensor) -> torch.Tensor:
            maps = self._to_maps(x)
            state["x"] = maps
            images = model.A(maps)
            self._real = not images.is_complex()
            return self._to_bart(images)

        def derivative(dx: torch.Tensor) -> torch.Tensor:
            return self._to_bart(model.A_jvp(state["x"], self._to_maps(dx)))

        def adjoint(dy: torch.Tensor) -> torch.Tensor:
            return self._to_bart(model.A_vjp(state["x"], self._cotangent(dy, state["x"])))

        super().__init__(
            (self.contrasts, *self.voxels),
            (self.channels, *self.voxels),
            forward,
            derivative,
            adjoint,
        )

    # --- the two layouts ---------------------------------------------------

    def _to_maps(self, x: torch.Tensor) -> torch.Tensor:
        """BART's ``(channels, *voxels)`` complex to BlochSim's real ``(*voxels, channels)``."""
        return x.reshape(self.channels, *self.voxels).movedim(0, -1).real.contiguous()

    def _to_bart(self, x: torch.Tensor) -> torch.Tensor:
        """The way back, into the real part of a complex buffer."""
        return x.movedim(-1, 0).to(torch.complex64).contiguous()

    def _cotangent(self, dy: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """One image per contrast, in BlochSim's layout and the images' own field."""
        if self._real is None:
            self._real = not self.model.A(maps).is_complex()
        dy = dy.reshape(self.contrasts, *self.voxels).movedim(0, -1)
        return (dy.real if self._real else dy).contiguous()

    def _bundle(self):
        """BlochSim takes the point as an argument already, so the bundle is its own pair.

        ``A_jvp`` and ``A_vjp`` are ``(x, v)`` throughout -- no Jacobian is
        built and no point is stored, as a bundle requires; the
        members are torch operators so that a step differentiating by the point
        differentiates them again rather than reading a derivative this
        recorded.
        """
        from bartorch.nlop._bundle import Bundle
        from bartorch.nlop._callback import TorchOperator

        model = self.model

        def derivative(dx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self._to_bart(model.A_jvp(self._to_maps(x), self._to_maps(dx)))

        def adjoint(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            maps = self._to_maps(x)
            return self._to_bart(model.A_vjp(maps, self._cotangent(dy, maps)))

        return Bundle(
            self,
            TorchOperator(derivative, [self.ishape, self.ishape], self.oshape),
            TorchOperator(adjoint, [self.oshape, self.ishape], self.ishape),
            source="torch",
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Name of each channel of the domain, in order."""
        return tuple(self.model.names)

    def initial(self, **values: Any) -> torch.Tensor:
        """Pack starting maps into the operator's domain.

        Parameters
        ----------
        **values
            ``name=value`` in each property's physical units, as
            :meth:`~blochsim.recon.ModelOperator.initial` takes them; a scalar
            or a map of the voxel shape.  Properties not given take the
            model's default.

        Returns
        -------
        torch.Tensor
            Complex64 tensor of :attr:`ishape`.
        """
        maps = self.model.initial(self.voxels, **values)
        return maps.movedim(-1, 0).to(torch.complex64).contiguous()

    def split(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unpack a tensor of the domain into named maps in physical units.

        The inverse of :meth:`initial`; only the real part of ``x`` is read.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of :attr:`ishape`, such as a fit's result.

        Returns
        -------
        dict of str to torch.Tensor
            One map of the voxel shape per unknown, e.g. ``T1`` in ms and a
            complex ``amplitude``.
        """
        maps = x.reshape(self.channels, *self.voxels).movedim(0, -1).real.contiguous()
        return self.model.split(maps)

    def __repr__(self) -> str:
        return (
            f"SignalModel({type(self.model.acquisition).__name__}, {self.voxels}, "
            f"unknown={list(self.model.unknown)})"
        )


def _from(acquisition, unknown, shape, bounds, scale, amplitude, subspace, contrasts):
    return SignalModel(
        _operator(acquisition, unknown, bounds, scale, amplitude, subspace),
        shape,
        contrasts,
    )


def InversionRecovery(  # noqa: N802  (it is a constructor)
    TI: Sequence[float],  # noqa: N803  (BART and the literature spell it this way)
    shape: Shape = (),
    *,
    TR: float | None = None,  # noqa: N803
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    unknown: Sequence[str] = ("T1",),
    amplitude: bool = True,
    subspace: Any = None,
    **scale: float,
) -> SignalModel:
    """Inversion-recovery signal model for T1 mapping.

    The longitudinal magnetization at a series of inversion times, from
    BlochSim's ``InversionRecoverySimulator``.  The unknowns are ``T1`` and,
    with ``amplitude``, a complex amplitude.

    Parameters
    ----------
    TI : sequence of float
        Inversion times, in milliseconds.
    shape : tuple of int, default=()
        The voxel shape, C order.
    TR : float, default=None
        Repetition time in milliseconds; ``None`` is full relaxation before
        each inversion.
    bounds : dict, default=None
        ``{name: (low, high)}`` per unknown, in its units; ``T1`` defaults to
        ``(10, 5000)`` ms.  Bounds of properties not in ``unknown`` are
        dropped.
    unknown : sequence of str, default=('T1',)
        Properties solved for, e.g. also ``inv_efficiency`` or ``offset``.
    amplitude : bool, default=True
        Include a complex amplitude multiplying the signal.
    subspace : blochsim.Subspace, default=None
        Temporal basis the prediction is projected through.
    **scale
        ``name=value``: step size of an unbounded parameter.

    Returns
    -------
    SignalModel
        Domain ``(channels, *shape)``, codomain ``(len(TI), *shape)``.
    """
    from blochsim.simulators import InversionRecoverySimulator

    bounds = {"T1": (10.0, 5000.0), **(bounds or {})}
    bounds = {name: bound for name, bound in bounds.items() if name in tuple(unknown)}
    sequence = (
        InversionRecoverySimulator(TI=TI)
        if TR is None
        else (InversionRecoverySimulator(TI=TI, TR=TR))
    )
    return _from(
        sequence, tuple(unknown), shape, bounds, scale or None, amplitude, subspace, len(TI)
    )


def MultiEcho(  # noqa: N802  (it is a constructor)
    TE: Sequence[float],  # noqa: N803
    shape: Shape = (),
    *,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    unknown: Sequence[str] = ("T2",),
    amplitude: bool = True,
    subspace: Any = None,
    **scale: float,
) -> SignalModel:
    """Mono-exponential multi-echo signal model for T2 or T2* mapping.

    The transverse decay at a series of echo times, from BlochSim's
    ``MultiEchoSimulator``.  The fitted constant is T2 for a spin-echo train
    and T2* for a gradient-echo train; the model is the same.

    Parameters
    ----------
    TE : sequence of float
        Echo times, in milliseconds.
    shape : tuple of int, default=()
        The voxel shape, C order.
    bounds : dict, default=None
        ``{name: (low, high)}`` per unknown, in its units; ``T2`` defaults to
        ``(1, 1000)`` ms.  Bounds of properties not in ``unknown`` are
        dropped.
    unknown : sequence of str, default=('T2',)
        Properties solved for, e.g. also ``offset``.
    amplitude : bool, default=True
        Include a complex amplitude multiplying the signal.
    subspace : blochsim.Subspace, default=None
        Temporal basis the prediction is projected through.
    **scale
        ``name=value``: step size of an unbounded parameter.

    Returns
    -------
    SignalModel
        Domain ``(channels, *shape)``, codomain ``(len(TE), *shape)``.
    """
    from blochsim.simulators import MultiEchoSimulator

    bounds = {"T2": (1.0, 1000.0), **(bounds or {})}
    bounds = {name: bound for name, bound in bounds.items() if name in tuple(unknown)}
    return _from(
        MultiEchoSimulator(TE=TE),
        tuple(unknown),
        shape,
        bounds,
        scale or None,
        amplitude,
        subspace,
        len(TE),
    )


def Bloch(  # noqa: N802  (it is a constructor)
    acquisition,
    *unknown: str,
    shape: Shape = (),
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    amplitude: bool = True,
    subspace: Any = None,
    contrasts: int | None = None,
    **scale: float,
) -> SignalModel:
    """Signal model of any BlochSim simulator.

    For sequences without a closed-form signal equation, such as a fast
    spin-echo train, a fingerprinting schedule or a bSSFP sweep.

    Parameters
    ----------
    acquisition : blochsim.Simulator
        The sequence, with every property not solved for fixed on it; a
        property given as a map (a measured B1, a known T1) has one value per
        voxel.
    *unknown : str
        Properties solved for, in channel order.
    shape : tuple of int, default=()
        Voxel shape, C order.
    bounds : dict, default=None
        ``{name: (low, high)}``, either end ``None`` for unbounded.  A bounded
        parameter is solved for in a transformed variable, so no iterate
        leaves the bound.
    amplitude : bool, default=True
        Include a complex amplitude multiplying the signal.
    subspace : blochsim.Subspace, default=None
        Temporal basis the prediction is projected through.
    contrasts : int, default=None
        Number of contrast images; evaluated from the model when ``None``.
    **scale
        ``name=value``: step size of an unbounded parameter.

    Returns
    -------
    SignalModel
        Domain ``(channels, *shape)``, codomain ``(contrasts, *shape)``.

    Examples
    --------
    >>> from blochsim.simulators import FSESimulator
    >>> M = Bloch(
    ...     FSESimulator(flip=train, ESP=8.0, TR=3000.0),
    ...     "T1", "T2",
    ...     shape=(128, 128),
    ...     bounds={"T1": (100.0, 4000.0), "T2": (5.0, 500.0)},
    ... )
    """
    return _from(acquisition, unknown, shape, bounds, scale or None, amplitude, subspace, contrasts)
