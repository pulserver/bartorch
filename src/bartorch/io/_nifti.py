"""NIfTI images, with BIDS sidecar timings, read into and written from image tensors."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._dicom import _TIMINGS, Images
from ._optional import require

#: BIDS sidecar keys, in seconds except the flip angle, for each timing.
_SIDECAR = {
    "inversion_time": "InversionTime",
    "echo_time": "EchoTime",
    "repetition_time": "RepetitionTime",
    "flip_angle": "FlipAngle",
}


def read_nifti(path: str | Sequence[str]) -> Images:
    """Read NIfTI images, and the timings of their BIDS sidecars, as contrasts.

    Parameters
    ----------
    path : str or sequence of str
        A ``.nii`` or ``.nii.gz`` file, or several of the same shape and
        geometry, one per contrast as BIDS stores the echoes of a multi-echo
        acquisition.  A JSON sidecar beside a file, of the same name, gives its
        ``EchoTime``, ``InversionTime`` and ``RepetitionTime`` in seconds and
        its ``FlipAngle`` in degrees.

    Returns
    -------
    Images
        A named tuple of ``image``, ``(contrasts, z, y, x)`` where the volumes
        of every file follow each other; ``affine``, the file's ``(4, 4)``
        from voxel indices ``(x, y, z)`` to RAS millimetres; ``timings`` in
        milliseconds and degrees, one per contrast, NaN where no sidecar states
        one; and ``header``, the first file's NIfTI header.

    Raises
    ------
    ValueError
        If the files differ in shape or geometry.
    """
    nibabel = require("nibabel")
    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
    volumes, timings, affine, header = [], {name: [] for name in _TIMINGS}, None, None
    for file in paths:
        loaded = nibabel.load(file)
        data = np.asanyarray(loaded.dataobj)
        data = data.astype(np.complex64 if np.iscomplexobj(data) else np.float32)
        slope, intercept = loaded.header.get_slope_inter()
        if slope is not None and not np.iscomplexobj(data):
            data = data * np.float32(slope) + np.float32(intercept or 0.0)
        data = data.reshape(*data.shape[:3], -1) if data.ndim > 3 else data[..., None]
        if affine is None:
            affine, header = loaded.affine, loaded.header
        elif (
            not np.allclose(loaded.affine, affine, atol=1e-4)
            or data.shape[:3] != volumes[0].shape[1:][::-1]
        ):
            raise ValueError(f"{file} differs from {paths[0]} in shape or geometry")
        volumes.append(np.ascontiguousarray(np.transpose(data, (3, 2, 1, 0))))
        sidecar = _sidecar(file)
        for name, key in _SIDECAR.items():
            value = sidecar.get(key)
            values = np.broadcast_to(
                np.nan if value is None else np.asarray(value, float), (data.shape[-1],)
            )
            timings[name].extend(values * (1.0 if name == "flip_angle" else 1e3))
    image = torch.from_numpy(np.concatenate(volumes))
    return Images(
        image,
        torch.from_numpy(np.asarray(affine, np.float64)),
        {name: torch.tensor(values, dtype=torch.float64) for name, values in timings.items()},
        header,
    )


def write_nifti(
    path: str, image: Any, affine: Any, *, timings: dict[str, Any] | None = None
) -> Path:
    """Write an image as a NIfTI file, and its timings as a BIDS sidecar.

    Parameters
    ----------
    path : str
        ``.nii`` or ``.nii.gz``.
    image : torch.Tensor or numpy.ndarray
        ``(*leading, z, y, x)`` or ``(y, x)``, real or complex; leading axes
        become the fourth NIfTI dimension, one after another.
    affine : torch.Tensor or numpy.ndarray
        ``(4, 4)``, from voxel indices ``(x, y, z)`` to RAS millimetres; it is
        stored as both the sform and the qform.
    timings : dict, default=None
        ``echo_time``, ``inversion_time``, ``repetition_time`` in milliseconds
        and ``flip_angle`` in degrees, each a number or one per volume, written
        beside the image as a JSON sidecar in BIDS units.

    Returns
    -------
    pathlib.Path
        The image file.
    """
    nibabel = require("nibabel")
    data = np.asarray(torch.as_tensor(image).detach().cpu().numpy())
    if data.ndim == 2:
        data = data[None]
    data = data.reshape(-1, *data.shape[-3:])
    data = np.transpose(data, (3, 2, 1, 0))
    if data.shape[-1] == 1:
        data = data[..., 0]
    matrix = np.asarray(torch.as_tensor(affine, dtype=torch.float64))
    nifti = nibabel.Nifti1Image(np.ascontiguousarray(data), matrix)
    nifti.set_qform(matrix, code=1)
    nifti.set_sform(matrix, code=1)
    path = Path(path)
    nibabel.save(nifti, path)
    if timings:
        sidecar = {}
        for name, value in timings.items():
            if name not in _SIDECAR:
                raise ValueError(f"unknown timing {name!r}; expected one of {list(_SIDECAR)}")
            values = np.atleast_1d(np.asarray(torch.as_tensor(value, dtype=torch.float64)))
            values = values * (1.0 if name == "flip_angle" else 1e-3)
            values = [float(v) for v in values if not math.isnan(v)]
            if values:
                sidecar[_SIDECAR[name]] = values[0] if len(set(values)) == 1 else values
        _stem(path).with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return path


def _stem(path: Path) -> Path:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path.with_suffix("")


def _sidecar(path: Path) -> dict[str, Any]:
    file = _stem(path).with_suffix(".json")
    return json.loads(file.read_text()) if file.exists() else {}
