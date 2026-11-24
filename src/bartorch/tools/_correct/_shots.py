"""Shot-to-shot phase of segmented acquisitions: the MUSE encoding and its phase fit."""

from __future__ import annotations

import torch

__all__ = [
    "estimate_shot_phases",
    "shot_encoding",
    "shot_phase_basis",
    "shot_residual",
]


def shot_encoding(encoding, masks: torch.Tensor, phases: torch.Tensor):
    r"""Encoding of a segmented acquisition whose shots each multiply the image by their own phase.

    :math:`\sum_s \mathrm{diag}(M_s)\, E\, \mathrm{diag}(c_s)`, built as
    :class:`~bartorch.linop.FieldCorrected` with its coefficients given: shot
    :math:`s` reads the lines :math:`M_s` of the image times its phase
    :math:`c_s`, as MUSE encodes them (Chen et al., NeuroImage 2013,
    doi:10.1016/j.neuroimage.2013.01.076).

    Parameters
    ----------
    encoding : bartorch.linop.CartesianSense
        Encoding of the image grid ``(..., y, x)``; ``(y, x)`` for one partition.
    masks : torch.Tensor
        ``(shots, ..., y)`` booleans, the lines each shot placed: the image
        grid without its readout, a plane given as one partition or none.
    phases : torch.Tensor
        ``(shots, ..., y, x)`` unit-magnitude phases on the image grid.

    Returns
    -------
    bartorch.linop.LinearOperator
    """
    from bartorch import linop

    grid = tuple(encoding.ishape)
    count = int(masks.shape[0])
    rows = masks.reshape(count, 1, *grid[:-1], 1).expand(count, 1, *grid)
    weights = rows.to(torch.complex64).contiguous()
    phases = phases.reshape(count, *grid).to(torch.complex64).contiguous()
    return linop.FieldCorrected(encoding, coefficients=(weights, phases))


def shot_phase_basis(shape: tuple[int, int, int], order: int) -> torch.Tensor:
    """Legendre products in ``y`` and ``x`` of total degree at most ``order``, per partition.

    The polynomials run over ``[-1, 1]`` across the field of view, so a shot's
    phase is fitted to each partition (each band of a multiband unit) on its
    own.

    Parameters
    ----------
    shape : tuple of int
        ``(z, y, x)`` image grid.
    order : int
        Largest total degree.

    Returns
    -------
    torch.Tensor
        ``(z * terms, z, y, x)`` float32 functions, ``terms`` the
        ``(order + 1) (order + 2) / 2`` products.

    Raises
    ------
    ValueError
        If ``order`` is negative.
    """
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    planes, rows, columns = shape
    y = torch.linspace(-1.0, 1.0, rows)
    x = torch.linspace(-1.0, 1.0, columns)

    def legendre(degree, t):
        low, high = torch.ones_like(t), t
        if degree == 0:
            return low
        for n in range(1, degree):
            low, high = high, ((2 * n + 1) * t * high - n * low) / (n + 1)
        return high

    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    basis = torch.zeros((planes * len(terms), *shape))
    for plane in range(planes):
        for index, (i, j) in enumerate(terms):
            basis[plane * len(terms) + index, plane] = (
                legendre(i, y)[:, None] * legendre(j, x)[None]
            )
    return basis


def _axes(maps: torch.Tensor) -> tuple[int, ...]:
    return (-1, -2, -3) if maps.shape[1] > 1 else (-1, -2)


def _forward(maps, mask, images):
    from bartorch._fourier import fft

    coils = maps * images.unsqueeze(-4)
    return fft(coils, _axes(maps), unitary=True).reshape(coils.shape) * mask[..., None]


def _adjoint(maps, mask, kspace):
    from bartorch._fourier import fft

    lines = kspace * mask[..., None]
    images = fft(lines, _axes(maps), inverse=True, unitary=True).reshape(lines.shape)
    return (maps.conj() * images).sum(dim=-4)


def shot_residual(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    masks: torch.Tensor,
    image: torch.Tensor,
    phases: torch.Tensor,
) -> float:
    """Data residual of ``image`` under the shot ``phases``, relative to the placed k-space.

    Parameters
    ----------
    kspace, maps : torch.Tensor
        ``(coils, z, y, x)`` centred k-space and coil maps.
    masks : torch.Tensor
        ``(shots, z, y)`` booleans, the lines each shot placed.
    image : torch.Tensor
        ``(z, y, x)`` complex image.
    phases : torch.Tensor
        ``(shots, z, y, x)`` phases, in rad.

    Returns
    -------
    float
        ``||sum_s M_s F S exp(i phi_s) x - M y|| / ||M y||``.
    """
    model = torch.zeros_like(kspace)
    for mask, phase in zip(masks, phases, strict=True):
        model = model + _forward(maps, mask.to(kspace.dtype), torch.exp(1j * phase) * image)
    placed = kspace * masks.any(dim=0)[..., None]
    return float((model - placed).norm() / placed.norm().clamp_min(1e-30))


def estimate_shot_phases(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    masks: torch.Tensor,
    image: torch.Tensor,
    phases: torch.Tensor,
    basis: torch.Tensor,
    *,
    steps: int = 2,
    damping: float = 1e-3,
    terms_at_once: int = 28,
) -> torch.Tensor:
    r"""Phase each shot of a segmented acquisition adds to ``image``, relative to the first shot's.

    Shot :math:`s` reads ``kspace`` on the lines ``masks[s]`` of ``image``
    times :math:`e^{i\phi_s}`, through the SENSE encoding of ``maps``. Each
    :math:`\phi_s` is moved, from ``phases``, by its closed-form constant and
    then by ``steps`` Levenberg-damped Gauss-Newton steps over the smooth
    ``basis`` (:func:`shot_phase_basis`): the residual's Jacobian along term
    :math:`k` is the encoding of :math:`i B_k e^{i\phi_s} x`. The phases are
    then taken relative to the first shot's.

    Parameters
    ----------
    kspace, maps : torch.Tensor
        ``(coils, z, y, x)`` centred k-space and coil maps.
    masks : torch.Tensor
        ``(shots, z, y)`` booleans, the lines each shot placed.
    image : torch.Tensor
        ``(z, y, x)`` complex image the shots are fitted against.
    phases : torch.Tensor
        ``(shots, z, y, x)`` starting phases, in rad.
    basis : torch.Tensor
        ``(terms, z, y, x)`` real functions the phases move along.
    steps : int, default=2
        Gauss-Newton steps per shot.
    damping : float, default=1e-3
        Levenberg damping, a fraction of the mean curvature.
    terms_at_once : int, default=28
        Basis functions carried through the encoding at once while the
        curvature is built; each holds the coils times the volume.

    Returns
    -------
    torch.Tensor
        ``(shots, z, y, x)`` float32 phases, the first zero.
    """
    phases = phases.clone()
    terms = basis.shape[0]
    flat = basis.reshape(terms, -1)
    for shot in range(masks.shape[0]):
        mask = masks[shot].to(kspace.dtype)
        data = kspace * mask[..., None]
        shifted = torch.exp(1j * phases[shot]) * image
        model = _forward(maps, mask, shifted)
        phases[shot] += float(torch.angle((model.conj() * data).sum()))
        for _ in range(steps):
            shifted = torch.exp(1j * phases[shot]) * image
            residual = _forward(maps, mask, shifted) - data
            slope = (shifted.conj() * _adjoint(maps, mask, residual)).imag
            gradient = flat @ slope.reshape(-1)
            curvature = torch.cat(
                [
                    flat
                    @ (
                        shifted.conj()
                        * _adjoint(maps, mask, _forward(maps, mask, 1j * chunk * shifted))
                    )
                    .imag.reshape(chunk.shape[0], -1)
                    .T
                    for chunk in basis.split(terms_at_once)
                ],
                dim=1,
            )
            curvature = 0.5 * (curvature + curvature.T)
            curvature = curvature + damping * torch.trace(curvature) / terms * torch.eye(
                terms, device=curvature.device
            )
            step = torch.linalg.solve(curvature.double(), -gradient.double())
            phases[shot] += (step.float() @ flat).reshape(phases.shape[1:])
    return (phases - phases[:1]).float()
