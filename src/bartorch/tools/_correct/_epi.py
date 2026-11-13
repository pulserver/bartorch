"""EPI readout corrections: ramp-sampling resampling and odd/even phase."""

from __future__ import annotations

import math

import torch

__all__ = [
    "correct_lines",
    "epi_ramp_operator",
    "estimate_epi_echo_phase",
    "estimate_epi_phase",
]


def _hybrid(rows: torch.Tensor, *, inverse: bool = True) -> torch.Tensor:
    """Centred unitary transform along the readout, the last axis."""
    from bartorch._fourier import fft

    rows = torch.as_tensor(rows)
    return fft(rows, -1, inverse=inverse, unitary=True).reshape(rows.shape)


def _readout_coordinate(size: int, device) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, size, dtype=torch.float64, device=device)


def _polyval(coordinate: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    total = torch.zeros_like(coordinate)
    for power in range(coefficients.shape[0] - 1, -1, -1):
        total = total * coordinate + coefficients[power]
    return total


def epi_ramp_operator(
    sample_positions: torch.Tensor,
    target_positions: torch.Tensor,
    support: int,
    *,
    regularization: float = 1e-6,
) -> torch.Tensor:
    r"""Band-limited resampling of a readout from its sampled to its target positions.

    The readout is modelled as the transform of an object of ``support``
    pixels, :math:`s(k) = \sum_x e^{-2\pi i k x} o_x` over
    :math:`x = -\lfloor S/2 \rfloor, \dots, S - 1 - \lfloor S/2 \rfloor`.  The
    operator is :math:`E_t (E_s^H E_s + \lambda n I)^{-1} E_s^H`, the
    regularized least-squares fit of the object to the ``n`` samples followed
    by the transform at the targets.  It is exact for a band-limited readout
    whose samples determine the object, which ramp sampling with readout
    oversampling provides; where the sample spacing exceeds ``1 / support``
    the readout is aliased and ``regularization`` only limits the noise gain.

    Parameters
    ----------
    sample_positions : torch.Tensor
        k-space position of each sample, in cycles per pixel of the
        reconstructed matrix, within :math:`[-1/2, 1/2]`: the readout's
        trajectory under the ADC, ramps included.
    target_positions : torch.Tensor
        Positions of the uniform grid, in the same units.
    support : int
        Pixels the object spans along the readout: the reconstructed matrix
        size.
    regularization : float, default=1e-06
        Tikhonov weight :math:`\lambda`, relative to the sample count.

    Returns
    -------
    torch.Tensor
        Complex64 operator ``(targets, samples)`` on the device of
        ``sample_positions``; a ``(coils, samples)`` readout is resampled as
        ``readout @ operator.T``.

    Raises
    ------
    ValueError
        If either position set has fewer than two entries, or ``support`` is
        not positive.
    """
    sample_positions = torch.as_tensor(sample_positions)
    device = sample_positions.device
    taken_at = sample_positions.reshape(-1).to(torch.float64)
    wanted_at = torch.as_tensor(target_positions, device=device).reshape(-1).to(torch.float64)
    support = int(support)
    if taken_at.shape[0] < 2 or wanted_at.shape[0] < 2:
        raise ValueError("both position sets must describe a readout")
    if support < 1:
        raise ValueError(f"support must be positive, got {support}")

    grid = torch.arange(support, dtype=torch.float64, device=device) - support // 2
    taken = torch.exp(-2j * math.pi * torch.outer(taken_at, grid))
    wanted = torch.exp(-2j * math.pi * torch.outer(wanted_at, grid))
    identity = torch.eye(support, dtype=torch.complex128, device=device)
    normal = taken.conj().T @ taken + regularization * taken_at.shape[0] * identity
    return (wanted @ torch.linalg.solve(normal, taken.conj().T)).to(torch.complex64)


def estimate_epi_phase(
    navigator_lines: list[torch.Tensor],
    *,
    polynomial_order: int = 1,
) -> torch.Tensor:
    r"""Odd/even phase of an EPI readout, fitted to a three-line navigator.

    The navigator is three blip-nulled readouts of alternating polarity.  In
    hybrid space (the centred inverse transform along the readout), the phase
    of :math:`\sum_c m_c b_c^*`, with :math:`m` the mean of the two forward
    lines and :math:`b` the reversed line, is unwrapped and fitted with a
    polynomial by least squares weighted by the modulus of that sum, ignoring
    samples below a tenth of its peak.

    Parameters
    ----------
    navigator_lines : list of torch.Tensor
        Three ``(coils, samples)`` lines of polarity ``+ - +``, the reversed
        line already flipped into forward readout order.
    polynomial_order : int, default=1
        Polynomial order; one is a constant and a linear term, the phase of a
        constant offset and a gradient or ADC delay.

    Returns
    -------
    torch.Tensor
        Float64 coefficients, lowest order first, on the navigator's device,
        in a coordinate running from -1 to 1 across the readout.  They are the
        correction :func:`correct_lines` applies to a reversed line: the
        negative of the phase that line carries.  The constant is defined
        modulo :math:`2\pi`.

    Raises
    ------
    ValueError
        If fewer than three lines are given, or the order is negative.
    """
    if len(navigator_lines) < 3:
        raise ValueError(
            f"the navigator is three lines of alternating polarity, got {len(navigator_lines)}"
        )
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")

    forward = 0.5 * (_hybrid(navigator_lines[0]) + _hybrid(navigator_lines[2]))
    return _fit_odd_even(forward, _hybrid(navigator_lines[1]), polynomial_order)


def _fit_odd_even(forward: torch.Tensor, backward: torch.Tensor, order: int) -> torch.Tensor:
    """Weighted polynomial fit of the phase of ``forward * conj(backward)``, in hybrid space."""
    from bartorch._util import unwrap

    cross = (forward * backward.conj()).sum(0)
    weights = cross.abs().to(torch.float64)
    phase = unwrap(torch.angle(cross).to(torch.complex64), -1).real.to(torch.float64)
    coordinate = _readout_coordinate(phase.shape[0], phase.device)
    keep = weights > 0.1 * weights.max()
    design = torch.stack([coordinate[keep] ** p for p in range(order + 1)], dim=-1)
    scaled = design * weights[keep, None]
    return torch.linalg.lstsq(scaled, (phase[keep] * weights[keep])[:, None]).solution[:, 0]


def estimate_epi_echo_phase(
    reference_lines: list[torch.Tensor],
    reversed_lines: list[bool],
    *,
    polynomial_order: int = 1,
) -> torch.Tensor:
    r"""Odd/even phase of every echo of an EPI train, fitted to a blip-free reference train.

    The reference is the imaging train played with every phase-encode
    gradient at zero, so echo :math:`k` sees the read gradient, eddy currents
    and timing of echo :math:`k` of every imaging train.  For each reversed
    echo, the forward echoes either side of it stand in for the line it would
    have been read forward, their mean cancelling the off-resonance phase that
    grows linearly along the train, and the phase of the reversed echo
    against them is fitted as :func:`estimate_epi_phase` fits a navigator's.
    A reversed echo with a forward echo on one side only (the first or last
    of the train) takes the fit of the nearest reversed echo that has both,
    or, in a train too short to have one, the one-sided fit.  Forward echoes
    define the grid and take no correction.

    Parameters
    ----------
    reference_lines : list of torch.Tensor
        The reference train's ``(coils, samples)`` lines in echo order,
        reversed lines already flipped into forward readout order.
    reversed_lines : list of bool
        Whether each echo was read with the negative read gradient.
    polynomial_order : int, default=1
        Polynomial order; one is a constant and a linear term.

    Returns
    -------
    torch.Tensor
        Float64 ``(echoes, polynomial_order + 1)`` coefficients, lowest order
        first, on the reference's device, in the coordinate of
        :func:`estimate_epi_phase`: row :math:`k` is what
        :func:`correct_lines` applies to a reversed line of echo :math:`k`,
        zeros for a forward echo.

    Raises
    ------
    ValueError
        If the two lists differ in length, the train has no forward echo next
        to a reversed one, or the order is negative.
    """
    if len(reference_lines) != len(reversed_lines):
        raise ValueError(
            f"{len(reference_lines)} reference lines but {len(reversed_lines)} polarities"
        )
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")
    count = len(reference_lines)
    flags = [bool(flag) for flag in reversed_lines]
    hybrid = [_hybrid(line) for line in reference_lines]
    device = hybrid[0].device if hybrid else None
    coefficients = torch.zeros(count, polynomial_order + 1, dtype=torch.float64, device=device)

    interior, edges = [], []
    for echo in range(count):
        if not flags[echo]:
            continue
        sides = [j for j in (echo - 1, echo + 1) if 0 <= j < count and not flags[j]]
        if not sides:
            continue
        forward = sum(hybrid[j] for j in sides) / len(sides)
        fit = _fit_odd_even(forward, hybrid[echo], polynomial_order)
        coefficients[echo] = fit
        (interior if len(sides) == 2 else edges).append(echo)
    if not interior and not edges:
        raise ValueError("the reference train has no forward echo next to a reversed one")
    if interior:
        for echo in edges:
            nearest = min(interior, key=lambda other: abs(other - echo))
            coefficients[echo] = coefficients[nearest]
    return coefficients


def correct_lines(
    lines: list[tuple[torch.Tensor, bool]], phase: torch.Tensor | None = None
) -> list[torch.Tensor]:
    """Flip reversed EPI lines into forward readout order and remove their phase.

    A reversed line is flipped along the readout and, in hybrid space,
    multiplied by :math:`e^{i p(u)}`, the polynomial ``phase`` over
    :math:`u \\in [-1, 1]`.  Forward lines define the grid and pass unchanged.

    Parameters
    ----------
    lines : list of tuple
        ``(data, reversed)`` per line, ``data`` of shape ``(coils, samples)``.
    phase : torch.Tensor, default=None
        Coefficients from :func:`estimate_epi_phase`.  ``None`` flips reversed
        lines without phase correction.

    Returns
    -------
    list of torch.Tensor
        Complex64 lines in forward readout order, each on its input's device.
    """
    corrected = []
    for data, backwards in lines:
        row = torch.as_tensor(data)
        if backwards:
            row = torch.flip(row, [-1])
            if phase is not None:
                hybrid = _hybrid(row)
                coefficients = torch.as_tensor(phase, device=hybrid.device).to(torch.float64)
                ramp = _polyval(_readout_coordinate(hybrid.shape[-1], hybrid.device), coefficients)
                row = _hybrid(hybrid * torch.polar(torch.ones_like(ramp), ramp), inverse=False)
        corrected.append(row.to(torch.complex64))
    return corrected
