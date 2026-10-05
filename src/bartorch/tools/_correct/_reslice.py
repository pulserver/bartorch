"""Resampling of an image onto another prescription's grid from header geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

__all__ = ["reslice"]

_INTERPOLATION = {"linear": "sitkLinear", "nearest": "sitkNearestNeighbor"}


def _simpleitk():
    try:
        import SimpleITK
    except ImportError as error:
        raise ImportError("reslice requires SimpleITK: pip install 'bartorch[correct]'") from error
    return SimpleITK


def _field(geometry: Any, name: str) -> np.ndarray:
    try:
        value = getattr(geometry, name)
    except AttributeError:
        value = geometry[name]
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _grid(geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Origin (mm), spacing (mm), direction (3x3, axes as columns) and size (x, y, z)."""
    planes = (
        geometry if isinstance(geometry, Sequence) and not isinstance(geometry, Mapping) else None
    )
    first = planes[0] if planes is not None else geometry
    read, phase = _field(first, "read_dir"), _field(first, "phase_dir")
    fov, matrix = _field(first, "field_of_view"), _field(first, "matrix_size")
    nx, ny = int(matrix[0]), int(matrix[1])
    dx, dy = fov[0] / matrix[0], fov[1] / matrix[1]
    position = _field(first, "position")
    if planes is None:
        nz = int(matrix[2])
        dz = fov[2] / matrix[2]
        slice_dir = _field(first, "slice_dir")
        origin = position - slice_dir * (nz // 2) * dz
    else:
        nz = len(planes)
        positions = np.stack([_field(plane, "position") for plane in planes])
        if nz == 1:
            dz = fov[2] / matrix[2]
            slice_dir = _field(first, "slice_dir")
        else:
            steps = np.diff(positions, axis=0)
            lengths = np.linalg.norm(steps, axis=1)
            dz = lengths[0]
            if dz == 0 or np.any(np.abs(lengths - dz) > 1e-3 * dz):
                raise ValueError("planes are not equally spaced along the slice direction")
            slice_dir = steps[0] / dz
            if np.any(np.linalg.norm(np.cross(steps / lengths[:, None], slice_dir), axis=1) > 1e-4):
                raise ValueError("plane positions are not collinear along the slice direction")
        origin = position
    origin = origin - read * (nx // 2) * dx - phase * (ny // 2) * dy
    direction = np.stack([read, phase, slice_dir], axis=1)
    return origin, np.array([dx, dy, dz]), direction, (nx, ny, nz)


def _sitk_image(sitk, origin, spacing, direction, size, array=None):
    if array is None:
        image = sitk.Image(list(size), sitk.sitkFloat32)
    else:
        image = sitk.GetImageFromArray(array, isVector=True)
    image.SetOrigin(origin.tolist())
    image.SetSpacing(spacing.tolist())
    image.SetDirection(direction.reshape(-1).tolist())
    return image


def reslice(
    image: torch.Tensor | np.ndarray,
    source: Any,
    target: Any,
    *,
    interpolation: str = "linear",
    fill: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Resample an image onto the grid of another prescription from header geometry.

    No registration is performed: the transform between the grids is the one
    the two geometries define in patient coordinates.  The leading axes of
    ``image`` and the real and imaginary parts are components of one vector
    image and are resampled in a single pass.

    A geometry has the fields of an MRD ``ImageHeader`` (``position``,
    ``read_dir``, ``phase_dir``, ``slice_dir``, ``field_of_view``,
    ``matrix_size``), read as attributes or, failing that, as mapping keys.
    Positions and field of view are in millimetres; directions are unit
    vectors in patient coordinates.  Along an axis of ``n`` voxels the spacing
    is ``field_of_view / matrix_size`` and ``position`` is the centre of voxel
    ``n // 2``.

    Parameters
    ----------
    image : torch.Tensor or numpy.ndarray
        Real or complex image of shape ``(..., z, y, x)``, with ``x`` the
        readout, ``y`` the phase-encode and ``z`` the slice axis.
    source, target
        Geometry of ``image`` and of the grid wanted.  Either one geometry for
        the whole volume, with ``matrix_size[2]`` voxels of spacing
        ``field_of_view[2] / matrix_size[2]`` along ``slice_dir``, or a list
        or tuple of per-plane geometries, one per ``z`` index.  Plane ``k``
        is centred on its own ``position``; the slice spacing is the distance
        between consecutive positions and the slice direction runs from the
        first position to the second.  A single plane takes ``slice_dir`` and
        ``field_of_view[2]``.  The in-plane geometry is the first plane's.
    interpolation : {"linear", "nearest"}, default="linear"
        Interpolation of the source samples.
    fill : float, default=0.0
        Value of target voxels whose centre lies outside the source volume.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Image of shape ``(..., tz, ty, tx)`` on the target grid, in the
        container type of ``image``; a tensor stays on its device.  Complex
        input gives complex64 output; real input keeps its floating dtype, and
        other dtypes give float32.

    Raises
    ------
    ValueError
        If the shape of ``image`` does not match ``source``, the planes of a
        sequence are not equally spaced to 1e-3 relative or not collinear to
        1e-4, or ``interpolation`` is unknown.
    ImportError
        If SimpleITK is not installed.
    """
    if interpolation not in _INTERPOLATION:
        raise ValueError(
            f"interpolation must be one of {sorted(_INTERPOLATION)}: {interpolation!r}"
        )
    sitk = _simpleitk()
    s_origin, s_spacing, s_direction, (nx, ny, nz) = _grid(source)
    t_origin, t_spacing, t_direction, t_size = _grid(target)
    if tuple(image.shape[-3:]) != (nz, ny, nx):
        raise ValueError(
            f"image spatial shape {tuple(image.shape[-3:])} does not match the source "
            f"geometry (z, y, x) = {(nz, ny, nx)}"
        )

    is_tensor = isinstance(image, torch.Tensor)
    array = image.detach().cpu().numpy() if is_tensor else np.asarray(image)
    lead = array.shape[:-3]
    complex_input = np.iscomplexobj(array)
    components = array.reshape(-1, nz, ny, nx)
    if complex_input:
        components = np.concatenate([components.real, components.imag])
    vector = np.ascontiguousarray(np.moveaxis(components, 0, -1), dtype=np.float32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(_sitk_image(sitk, t_origin, t_spacing, t_direction, t_size))
    resampler.SetInterpolator(getattr(sitk, _INTERPOLATION[interpolation]))
    resampler.SetDefaultPixelValue(float(fill))
    resampled = resampler.Execute(_sitk_image(sitk, s_origin, s_spacing, s_direction, None, vector))

    # A one-component vector image comes back without its component axis.
    out = np.moveaxis(sitk.GetArrayFromImage(resampled).reshape(*t_size[::-1], -1), -1, 0)
    if complex_input:
        half = out.shape[0] // 2
        out = out[:half] + 1j * out[half:]
    out = out.reshape(*lead, t_size[2], t_size[1], t_size[0])
    if not is_tensor:
        return (
            out
            if complex_input
            else out.astype(array.dtype if array.dtype.kind == "f" else np.float32, copy=False)
        )
    result = torch.from_numpy(np.ascontiguousarray(out))
    if not complex_input and image.is_floating_point():
        result = result.to(image.dtype)
    return result.to(image.device)
