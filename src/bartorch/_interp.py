"""Interpolation on a grid."""

from __future__ import annotations

import torch

from bartorch import _grid
from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = ["interpolate"]


@curated("interpolate")
def interpolate(
    input: torch.Tensor,
    coord: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    order: int = 1,
) -> torch.Tensor:
    """Sample ``input`` at the voxel positions ``coord`` along ``axes``.

    ``out[..., q, ...] = sum_j input[..., j, ...] w(coord[q, i] - j_i)`` over
    the ``axes``, where ``w`` is the kernel of ``order``.  Positions are array
    indices: ``coord = 2.0`` is the sample ``input[2]``.  The grid is
    zero outside ``[0, n - 1]``, so a position within one kernel radius of
    the edge is weighted against zeros.  BART's ``interpolate``.

    Parameters
    ----------
    input : torch.Tensor
        Array to sample.
    coord : torch.Tensor
        Real positions ``(..., len(axes))``: component ``i`` is the position
        along ``axes[i]``.  The leading shape aligns right against
        ``input.shape``: the output size along each of ``axes``, and one
        (shared positions) or the input's size elsewhere.  Along an axis
        before the last three, the output size must equal the input's.
    axes : int or tuple of int
        Axes interpolated over.
    order : {0, 1, 3}, default=1
        Nearest neighbour (rounding half up), linear, or Keys cubic
        with ``a = -1/2``.

    Returns
    -------
    torch.Tensor
        ``input.shape`` with each of ``axes`` replaced by ``coord``'s size
        there.

    Raises
    ------
    ValueError
        If ``coord`` does not match ``input`` as described.

    Examples
    --------
    >>> line = interpolate(image, torch.tensor([[1.5, 2.0]]), axes=(-2, -1))  # (1, 1)
    """
    ndim = input.ndim
    axes = _grid.normalize_axes(axes, ndim)
    _grid.check_motion_dim(input, axes)
    field, grid = _grid.motion_field(coord, ndim, axes, "coord")
    out_shape = list(input.shape)
    for ax in range(ndim):
        size, have = grid[ax], input.shape[ax]
        if ax in axes:
            if ndim - 1 - ax > 2 and size != have:
                raise ValueError(
                    f"coord has size {size} along axis {ax}, outside the last three: "
                    f"it must equal the input's {have}"
                )
            out_shape[ax] = size
        elif size not in (1, have):
            raise ValueError(
                f"coord has size {size} along axis {ax}, which is not interpolated: "
                f"it must be one or the input's {have}"
            )
    x = tuple(out_shape[ndim - 1 - b] if b < ndim else 1 for b in range(3))
    out = dispatch(
        "interpolate",
        [input, field],
        None,
        _pos=[axes_flags(axes, ndim)],
        x=x,
        **_grid.order_flags(order),
    )
    return out.reshape(out_shape)
