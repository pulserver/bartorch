"""Partition of the acquired samples, for training without a fully sampled reference."""

from __future__ import annotations

import torch

__all__ = ["split"]

_DENSITIES = ("gaussian", "uniform")


def split(
    pattern: torch.Tensor,
    fraction: float = 0.4,
    *,
    density: str = "gaussian",
    width: float = 0.5,
    keep=None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition the acquired samples into a set to reconstruct from and a set held out.

    Self-supervised training by data undersampling (SSDU) reconstructs from
    one part of the acquired samples and evaluates the loss on the other, in
    k-space, so that no fully sampled reference is needed.  Drawing a new
    partition at each step gives the multi-mask variant.

    The held-out samples are drawn without replacement among the acquired
    entries of ``pattern`` outside the ``keep`` region.  With a Gaussian
    density the probability of an entry falls with its distance from the
    centre of the pattern, as a centred k-space is sampled; ``"uniform"`` is
    for a pattern over readouts -- spokes, interleaves, shots -- whose index
    is not a position in k-space.

    Parameters
    ----------
    pattern : torch.Tensor
        The acquired samples, nonzero where acquired: a Cartesian sampling
        pattern over its phase-encode axes, or a mask over the readouts of a
        non-Cartesian trajectory, broadcast onto the k-space over its leading
        axes and wherever it has an extent of one.
    fraction : float, default=0.4
        Share of the acquired samples outside ``keep`` that is held out.
    density : {"gaussian", "uniform"}, default="gaussian"
        How the held-out samples are distributed over ``pattern``.
    width : float, default=0.5
        Standard deviation of the Gaussian density, relative to the extent of
        each axis.
    keep : sequence of int, default=None
        Extent, along each axis of ``pattern``, of a central region whose
        samples all stay in the set reconstructed from, such as the
        calibration region.
    generator : torch.Generator, default=None
        Source of the random draw.

    Returns
    -------
    reconstruct, held : torch.Tensor
        Disjoint float masks of ``pattern``'s shape whose sum is the acquired
        mask.

    References
    ----------
    Yaman B, Hosseini SAH, Moeller S, Ellermann J, Ugurbil K, Akcakaya M.
    Self-supervised learning of physics-guided reconstruction neural networks
    without fully sampled reference data. Magn Reson Med 2020;84:3172-3191.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"the held-out share is strictly between 0 and 1, not {fraction}")
    if density not in _DENSITIES:
        raise ValueError(f"density is one of {_DENSITIES}, not {density!r}")
    acquired = pattern.detach().abs() > 0 if pattern.is_complex() else pattern.detach() != 0
    acquired = acquired.cpu()
    candidates = acquired.clone()
    if keep is not None:
        keep = tuple(int(k) for k in keep)
        if len(keep) != pattern.ndim:
            raise ValueError(
                f"keep gives an extent per axis of the pattern, {pattern.ndim} of them"
            )
        centre = tuple(
            slice(max(n // 2 - k // 2, 0), n // 2 - k // 2 + k) for n, k in zip(pattern.shape, keep)
        )
        candidates[centre] = False

    weights = candidates.to(torch.float64)
    if "gaussian" == density:
        for axis, n in enumerate(pattern.shape):
            r = (torch.arange(n, dtype=torch.float64) - n // 2) / (width * n)
            shape = [1] * pattern.ndim
            shape[axis] = n
            weights = weights * torch.exp(-0.5 * r.square()).reshape(shape)

    count = round(fraction * int(candidates.sum()))
    held = torch.zeros(pattern.numel(), dtype=torch.bool)
    if count:
        chosen = torch.multinomial(weights.reshape(-1), count, generator=generator)
        held[chosen] = True
    held = held.reshape(pattern.shape)
    reconstruct = acquired & ~held
    return reconstruct.float().to(pattern.device), held.float().to(pattern.device)
