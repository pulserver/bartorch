"""Voxel-to-world affines, between MRD, DICOM and NIfTI conventions.

An affine here maps voxel indices ``(x, y, z)`` -- the last, second-to-last and
third-to-last axes of an image tensor -- to RAS coordinates in millimetres,
which is NIfTI's convention.  MRD and DICOM state positions in the patient
coordinate system, LPS, which differs in the sign of the first two axes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def mrd_affine(space: dict[str, Any], readouts: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Affine of the image an MRD encoding space reconstructs to.

    ``x`` runs along the readouts' ``read_dir`` and ``y`` along ``phase_dir``,
    each spanning the reconstructed field of view over the reconstructed
    matrix, and ``position`` is the centre of the field of view.  ``z`` is the
    partition axis of a volume, spanning the field of view along ``slice_dir``,
    or, with more than one slice, the slice counter, one step being the mean
    distance between consecutive slices; a single slice has the slab thickness.

    Returns ``None`` when the readouts carry no orientation or the header no
    field of view.
    """
    read = readouts["read_dir"][0].numpy()
    phase = readouts["phase_dir"][0].numpy()
    normal = readouts["slice_dir"][0].numpy()
    fov = space["fov_mm"]
    if fov is None or not all(fov) or not (np.linalg.norm(read) and np.linalg.norm(phase)):
        return None
    if not np.linalg.norm(normal):
        normal = np.cross(read, phase)
    nx, ny, nz = space["matrix"]
    columns = np.zeros((3, 3))
    columns[:, 0] = read * fov[0] / nx
    columns[:, 1] = phase * fov[1] / ny

    slices = readouts["slice"].numpy()
    positions = readouts["position"].numpy()
    order = np.unique(slices)
    first = positions[np.flatnonzero(slices == order[0])[0]]
    if order.size > 1:
        last = positions[np.flatnonzero(slices == order[-1])[0]]
        columns[:, 2] = (last - first) / float(order[-1] - order[0])
        centre = first - order[0] * columns[:, 2]
        middle = 0.0
    else:
        columns[:, 2] = normal * (fov[2] / nz if nz > 1 else fov[2])
        centre = first
        middle = (nz - 1) / 2
    origin = (
        centre
        - (nx - 1) / 2 * columns[:, 0]
        - (ny - 1) / 2 * columns[:, 1]
        - middle * columns[:, 2]
    )
    return lps_affine(columns, origin)


def lps_affine(columns: np.ndarray, origin: np.ndarray) -> torch.Tensor:
    """RAS affine from the steps along ``x``, ``y``, ``z`` and the first voxel, all in LPS."""
    affine = np.eye(4)
    affine[:3, :3] = _LPS_TO_RAS @ columns
    affine[:3, 3] = _LPS_TO_RAS @ origin
    return torch.from_numpy(affine)


def to_lps(affine: Any) -> tuple[np.ndarray, np.ndarray]:
    """Steps along ``x``, ``y``, ``z`` as columns, and the first voxel, in LPS."""
    affine = np.asarray(torch.as_tensor(affine, dtype=torch.float64))
    if affine.shape != (4, 4):
        raise ValueError(f"an affine is (4, 4), got {affine.shape}")
    return _LPS_TO_RAS @ affine[:3, :3], _LPS_TO_RAS @ affine[:3, 3]
