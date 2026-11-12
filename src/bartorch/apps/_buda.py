"""BUDA: one image from EPI shots of opposite phase-encode polarity, solved jointly.

Not a BART command.  Each shot is encoded by :func:`~bartorch.linop.CartesianSense`
under :func:`~bartorch.linop.FieldCorrected` at its own sample times, so the
off-resonance that displaces a blip-up image one way and a blip-down image the
other is part of the encoding, and one image serves both shots' data.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

import bartorch
from bartorch import linop, optim
from bartorch.apps._pics import _MAXITER, _SOLVERS, _chosen, _terms
from bartorch.priors import Regularizer

__all__ = ["buda", "shot_phase"]


def _per_shot(values: torch.Tensor | None, shots: int, name: str) -> list:
    if values is None:
        return [None] * shots
    if values.shape[0] != shots:
        raise ValueError(f"{name} has {values.shape[0]} shots, the k-space {shots}")
    return list(values)


def _on_codomain(times: torch.Tensor, oshape: tuple[int, ...]) -> torch.Tensor:
    """Sample times ``([z,] y, x)``, or broadcastable to it, reshaped to broadcast to ``oshape``."""
    times = torch.as_tensor(times, dtype=torch.float32)
    return times.reshape((1,) * (len(oshape) - times.ndim) + tuple(times.shape))


def _encodings(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    field_map: torch.Tensor,
    readout_time: torch.Tensor,
    phase: torch.Tensor | None,
    segments: int,
    mask: torch.Tensor | None,
    device,
) -> list[linop.LinearOperator]:
    """One field-corrected SENSE encoding per shot, each on that shot's sampling pattern."""
    shots = kspace.shape[0]
    grid = tuple(kspace.shape[2:])
    times = _per_shot(torch.as_tensor(readout_time), shots, "readout_time")
    phases = _per_shot(phase, shots, "phase")
    encodings = []
    for shot in range(shots):
        pattern = (kspace[shot].abs().sum(dim=0) > 0).to(torch.complex64)
        maps = sensitivities
        if phases[shot] is not None:
            maps = maps * torch.exp(1j * phases[shot].to(torch.float32))
        sense = linop.CartesianSense(maps, grid, pattern=pattern, device=device)
        encodings.append(
            linop.FieldCorrected(
                sense,
                field_map.to(torch.float32),
                _on_codomain(times[shot], tuple(sense.oshape)),
                mask=mask,
                segments=segments,
            )
        )
    return encodings


def shot_phase(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    field_map: torch.Tensor,
    readout_time: torch.Tensor,
    *,
    fraction: float = 0.25,
    l2: float = 1e-3,
    maxiter: int = 20,
    segments: int = -1,
    mask: torch.Tensor | None = None,
    device=None,
) -> torch.Tensor:
    """Return each shot's low-resolution image phase, from its own field-corrected SENSE image.

    Each shot is solved alone, by conjugate gradients with Tikhonov weight
    ``l2`` relative to its data scaling, under the encoding :func:`buda`
    builds, and its image is low-pass filtered by a Hann window spanning
    ``fraction`` of k-space along each axis.  The phase that remains is what
    differs between shots acquired at different times, such as a drift of the
    field or of the receiver, which a joint image cannot carry.

    Parameters
    ----------
    kspace, sensitivities, field_map, readout_time
        As :func:`buda`.
    fraction : float, default=0.25
        Width of the Hann window, as a fraction of each axis of k-space.
    l2 : float, default=0.001
        Tikhonov weight of each shot's solve.
    maxiter : int, default=20
        Conjugate-gradient iterations of each shot's solve.
    segments : int, default=-1
        As :func:`buda`.
    mask : torch.Tensor, default=None
        As :func:`buda`.
    device : torch.device or str, default=None
        As :func:`buda`.

    Returns
    -------
    torch.Tensor
        ``(shots, [z,] y, x)`` phase in radians.
    """
    grid = tuple(kspace.shape[2:])
    phases = []
    for shot, encoding in enumerate(
        _encodings(kspace, sensitivities, field_map, readout_time, None, segments, mask, device)
    ):
        data = (kspace[shot] * (kspace[shot].abs().sum(dim=0) > 0)).reshape(encoding.oshape)
        scale = optim.data_scaling(data, A=encoding) or 1.0
        image = optim.CG(l2, maxiter=maxiter)(data * (1.0 / scale), encoding).reshape(grid)
        axes = tuple(range(-len(grid), 0))
        spectrum = bartorch.fft(image.to(torch.complex64), axes, unitary=True).reshape(grid)
        window = torch.ones(grid)
        for axis, n in zip(axes, grid, strict=True):
            width = max(int(round(fraction * n)), 1)
            hann = torch.zeros(n)
            start = n // 2 - width // 2
            hann[start : start + width] = torch.hann_window(width, periodic=False)
            shape = [1] * len(grid)
            shape[axis] = n
            window = window * hann.reshape(shape)
        low = bartorch.fft(spectrum * window, axes, inverse=True, unitary=True).reshape(grid)
        phases.append(torch.angle(low))
    return torch.stack(phases)


def buda(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    field_map: torch.Tensor,
    readout_time: torch.Tensor,
    *,
    phase: torch.Tensor | str | None = None,
    regularizers: Regularizer | Iterable[Regularizer] | None = None,
    l2: float | None = None,
    maxiter: int | None = None,
    segments: int = -1,
    mask: torch.Tensor | None = None,
    device=None,
    scaling: float | None = None,
) -> torch.Tensor:
    r"""Reconstruct one image from EPI shots of opposite phase-encode polarity (BUDA).

    Blip-up/blip-down acquisition (BUDA; Liao et al., Magn Reson Med 2021):
    the shots are encoded together,

    .. math::

        y_s = P_s F \, S \, e^{i \phi_s} e^{i 2 \pi f t_s} x ,

    with :math:`P_s` the sampling of shot :math:`s`, :math:`S` the coil
    sensitivities, :math:`\phi_s` the shot's image phase, :math:`f` the
    off-resonance and :math:`t_s` the time each k-space sample was acquired, by
    time segmentation (:func:`~bartorch.linop.FieldCorrected`).  The image
    solves the stacked data consistency over every shot, under ``regularizers``
    as :func:`~bartorch.apps.pics` chooses its iteration from them, after
    dividing the data by the scaling estimated from the adjoint image.  The
    off-resonance displaces the shots in opposite directions along the phase
    encode, so the joint image is undistorted, and the shots' samples sum
    where a lattice is shared, or fill each other's gaps where it is not.

    Parameters
    ----------
    kspace : torch.Tensor
        ``(shots, coils, [z,] y, x)`` k-space on the reconstruction grid, centred,
        zero where a shot has no sample.
    sensitivities : torch.Tensor
        ``(coils, [z,] y, x)`` coil sensitivities, undistorted, such as those of
        a gradient-echo calibration.
    field_map : torch.Tensor
        ``([z,] y, x)`` off-resonance in Hz.
    readout_time : torch.Tensor
        ``(shots, [z,] y, x)``, or broadcastable to it, the time of each
        k-space sample from the excitation, in s; an EPI line's samples share
        its echo time to within a readout.
    phase : torch.Tensor or "estimate" or None, default=None
        ``(shots, [z,] y, x)`` image phase of each shot in radians;
        ``"estimate"`` takes :func:`shot_phase`; ``None`` is a phase shared by
        every shot.
    regularizers : Regularizer or iterable of Regularizer, default=None
        Terms from :mod:`bartorch.priors`, relative to the scaled data.
    l2 : float, default=None
        Tikhonov weight, with conjugate gradients.
    maxiter : int, default=None
        Iterations; ``None`` is BART's 30.
    segments : int, default=-1
        Time segments of :func:`~bartorch.linop.FieldCorrected`; -1 lets it
        choose.
    mask : torch.Tensor, default=None
        Support of the field map, which its fit weights.
    device : torch.device or str, default=None
        Where the encodings compute; the data and the image stay where they are.
    scaling : float, default=None
        Data scaling; ``None`` estimates it from the adjoint image.

    Returns
    -------
    torch.Tensor
        ``([z,] y, x)`` complex image in the units of ``kspace``.

    Raises
    ------
    ValueError
        If ``readout_time`` or ``phase`` does not hold one entry per shot.
    """
    if isinstance(phase, str):
        if phase != "estimate":
            raise ValueError(f"phase is a tensor, 'estimate' or None, not {phase!r}")
        phase = shot_phase(
            kspace,
            sensitivities,
            field_map,
            readout_time,
            segments=segments,
            mask=mask,
            device=device,
        )
    encodings = _encodings(
        kspace, sensitivities, field_map, readout_time, phase, segments, mask, device
    )
    A = linop.stack(encodings)
    patterns = (kspace.abs().sum(dim=1, keepdim=True) > 0).to(kspace.dtype)
    data = (kspace * patterns).reshape(A.oshape)
    scale = scaling or optim.data_scaling(data, A=A) or 1.0

    terms = _terms(regularizers)
    solver = _chosen(terms)
    if solver == "cg":
        arguments = [] if l2 is None else [l2]
    else:
        arguments = [terms]
    iterate = _SOLVERS[solver](*arguments, maxiter=_MAXITER if maxiter is None else maxiter)
    return (iterate(data * (1.0 / scale), A) * scale).reshape(tuple(kspace.shape[2:]))
