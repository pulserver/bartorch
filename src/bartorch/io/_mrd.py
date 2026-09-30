"""ISMRMRD raw data files read into k-space tensors."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch

from ._geometry import mrd_affine
from ._optional import require

#: Loop counters that become k-space axes when the header states more than one
#: position along them, outermost first.  ``segment`` indexes part of one
#: readout train, not an image, and is not among them.
LOOP_COUNTERS = ("repetition", "phase", "slice", "contrast", "set", "average")

# ISMRMRD acquisition flags, as the 1-based bit positions the standard numbers them.
_NOISE = 19
_CALIBRATION = (20, 21)  # IS_PARALLEL_CALIBRATION, ..._AND_IMAGING
_SKIPPED = (
    17,  # IS_NAVIGATION_DATA
    18,  # IS_PHASECORR_DATA
    23,  # IS_SURFACECOILCORRECTIONSCAN_DATA
    24,  # IS_SURFACECOILCORRECTIONSCAN_DATA_AND_IMAGING
    25,  # IS_RTFEEDBACK_DATA
    26,  # IS_HPFEEDBACK_DATA
    28,  # IS_DUMMYSCAN_DATA
)

_TRAJECTORY_UNITS = (None, "grid", "1/m", "fraction")


class MrdRaw(NamedTuple):
    """The imaging readouts of one encoding space, placed by their counters."""

    kspace: torch.Tensor
    mask: torch.Tensor
    reference: torch.Tensor
    trajectory: torch.Tensor | None
    noise: torch.Tensor | None
    axes: tuple[str, ...]
    affine: torch.Tensor | None
    header: Any
    readouts: dict[str, torch.Tensor]


def read_mrd(
    path: str,
    *,
    encoding: int = 0,
    group: str = "dataset",
    trajectory_units: str | None = None,
) -> MrdRaw:
    """Read the readouts of one encoding space of an ISMRMRD HDF5 file.

    Each imaging and calibration readout is placed where its
    ``kspace_encode_step_1``, ``kspace_encode_step_2`` and loop counters say it
    belongs; a readout shorter than the others is right-aligned, where a partial
    echo's samples belong.  Noise readouts are returned apart; navigator,
    phase-correction, dummy-scan, feedback and surface-coil readouts are not
    read.

    Parameters
    ----------
    path : str
        The ``.h5`` file.
    encoding : int, default=0
        Encoding space to read, as the readouts' ``encoding_space_ref`` names it.
    group : str, default="dataset"
        HDF5 group holding the header and the readouts.
    trajectory_units : {None, "grid", "1/m", "fraction"}, default=None
        Units the file stores the trajectory in, to convert it to grid units:
        ``"1/m"`` is scaled by the reconstructed field of view along each axis,
        ``"fraction"`` (of the sampling bandwidth, within ``[-0.5, 0.5)``) by the
        reconstructed matrix.  ``None`` returns the trajectory as stored:
        ISMRMRD does not fix its units.

    Returns
    -------
    MrdRaw
        A named tuple of

        ``kspace``
            ``complex64``, ``(*loops, coils, [partitions,] phase_encodes,
            readout)``: the loop counters -- ``repetition``, ``phase``,
            ``slice``, ``contrast``, ``set``, ``average``, in that order -- the
            header gives more than one position, then the channels; the
            partition axis is present when the header states more than one.
            Zero where nothing was acquired.  For a non-Cartesian space the
            phase encodes are the shots and the readout the samples of each,
            which is how :class:`~bartorch.linop.NoncartesianSense` takes them.
        ``mask``, ``reference``
            ``bool``, the shape of ``kspace`` without its coil axis: every
            placed sample, and the samples of parallel-imaging calibration
            readouts.
        ``trajectory``
            ``float32``, the shape of ``mask`` with ``(kx, ky, kz)`` last, or
            ``None`` when no readout carries one.  Components a readout does
            not carry are zero.
        ``noise``
            ``complex64``, ``(coils, samples)``, the noise readouts side by
            side, or ``None``.
        ``axes``
            The name of each axis of ``kspace``: the counter names, ``"coil"``,
            ``"partition"``, ``"phase_encode"``, ``"readout"``.
        ``affine``
            ``float64`` ``(4, 4)`` from voxel indices ``(x, y, z)`` of the image
            the header's reconstructed matrix describes to RAS coordinates in
            millimetres, as :func:`write_nifti` and :func:`write_dicom` take
            it; ``None`` when the readouts carry no orientation.
        ``header``
            The parsed ``ismrmrd.xsd.ismrmrdHeader``.
        ``readouts``
            One row per placed readout, in file order: the counters and
            ``flags``, ``scan_counter``, ``acquisition_time_stamp``,
            ``physiology_time_stamp``, ``sample_time_us``, ``center_sample``,
            and ``position``, ``read_dir``, ``phase_dir``, ``slice_dir`` and
            ``patient_table_position`` in the scanner's patient coordinates,
            millimetres for positions.

    Raises
    ------
    ValueError
        If the file has no header, the header has no such encoding space, no
        readout belongs to it, or a counter runs past the extent the header
        states.
    """
    if trajectory_units not in _TRAJECTORY_UNITS:
        raise ValueError(
            f"trajectory_units must be one of {_TRAJECTORY_UNITS}, got {trajectory_units!r}"
        )
    h5py = require("h5py")
    xsd = require("ismrmrd.xsd")

    with h5py.File(path, "r") as file:
        if group not in file or "xml" not in file[group]:
            raise ValueError(f"{path} has no ISMRMRD header in group {group!r}")
        document = file[group]["xml"][0]
        records = file[group]["data"][:] if "data" in file[group] else None
    if isinstance(document, bytes):
        document = document.decode()
    header = xsd.CreateFromDocument(document)
    space = _space(header, encoding)
    if records is None or len(records) == 0:
        raise ValueError(f"{path} holds no readouts")

    head = records["head"]
    flags = head["flags"].astype(np.uint64)
    noise_rows = np.flatnonzero(_is_set(flags, _NOISE))
    placed = ~_is_set(flags, _NOISE)
    for bit in _SKIPPED:
        placed &= ~_is_set(flags, bit)
    placed &= head["encoding_space_ref"] == encoding
    rows = np.flatnonzero(placed)
    if rows.size == 0:
        raise ValueError(f"{path} holds no imaging readouts in encoding space {encoding}")

    channels = int(head["active_channels"][rows].max())
    readout = max(space["readout"], int(head["number_of_samples"][rows].max()))
    extents = space["extents"]
    loops = tuple(name for name, extent in extents if name in LOOP_COUNTERS)
    placement = [(name, extent) for name, extent in extents if name not in LOOP_COUNTERS]
    placement = [
        (name, extent) for name, extent in placement if extent > 1 or name == "phase_encode"
    ]
    axes = (*loops, "coil", *(name for name, _ in placement), "readout")
    loop_shape = tuple(extent for name, extent in extents if name in LOOP_COUNTERS)
    inner_shape = tuple(extent for _, extent in placement)
    grid = (*loop_shape, *inner_shape)

    kspace = np.zeros((*loop_shape, channels, *inner_shape, readout), np.complex64)
    mask = np.zeros((*grid, readout), bool)
    reference = np.zeros_like(mask)
    trajectory = None

    idx = head["idx"]
    ordered = [(name, extent) for name, extent in extents if name in LOOP_COUNTERS] + placement
    names = [name for name, _ in ordered]
    counters = {name: idx[_COUNTER_FIELD.get(name, name)][rows].astype(np.int64) for name in names}
    for name, extent in ordered:
        values = counters[name]
        if values.size and (values.min() < 0 or values.max() >= extent):
            field = _COUNTER_FIELD.get(name, name)
            raise ValueError(
                f"a readout has {field}={int(values.max())}, past the {extent} {name} "
                f"positions encoding space {encoding} states"
            )

    calibration = np.zeros(rows.size, bool)
    for bit in _CALIBRATION:
        calibration |= _is_set(flags[rows], bit)

    for n, row in enumerate(rows):
        samples = int(head["number_of_samples"][row])
        coils = int(head["active_channels"][row])
        data = records["data"][row].view(np.complex64).reshape(coils, samples)
        where = tuple(int(counters[name][n]) for name in names)
        outer, inner = where[: len(loops)], where[len(loops) :]
        span = slice(readout - samples, readout)
        kspace[(*outer, slice(0, coils), *inner, span)] = data
        mask[(*where, span)] = True
        if calibration[n]:
            reference[(*where, span)] = True
        dimensions = int(head["trajectory_dimensions"][row])
        if dimensions:
            if trajectory is None:
                trajectory = np.zeros((*grid, readout, 3), np.float32)
            points = records["traj"][row].reshape(samples, dimensions)[:, :3]
            trajectory[(*where, span, slice(0, points.shape[1]))] = points

    if trajectory is not None and trajectory_units in ("1/m", "fraction"):
        scale = space["fov_m"] if trajectory_units == "1/m" else space["matrix"]
        if scale is None:
            raise ValueError(
                f"encoding space {encoding} states no field of view to scale the trajectory by"
            )
        trajectory *= np.asarray(scale, np.float32)

    noise = None
    if noise_rows.size:
        noise = np.concatenate(
            [
                records["data"][row]
                .view(np.complex64)
                .reshape(int(head["active_channels"][row]), int(head["number_of_samples"][row]))
                for row in noise_rows
            ],
            axis=1,
        )

    readouts = _readouts(head[rows])
    return MrdRaw(
        kspace=torch.from_numpy(kspace),
        mask=torch.from_numpy(mask),
        reference=torch.from_numpy(reference),
        trajectory=None if trajectory is None else torch.from_numpy(trajectory),
        noise=None if noise is None else torch.from_numpy(noise),
        axes=axes,
        affine=mrd_affine(space, readouts),
        header=header,
        readouts=readouts,
    )


#: The MRD counter each placement axis is read from.
_COUNTER_FIELD = {"partition": "kspace_encode_step_2", "phase_encode": "kspace_encode_step_1"}


def _is_set(flags: np.ndarray, bit: int) -> np.ndarray:
    return (flags & np.uint64(1 << (bit - 1))) != 0


def _space(header: Any, index: int) -> dict[str, Any]:
    """Layout of encoding space ``index``: extents, readout length, matrix and field of view.

    A Cartesian space has as many phase encodes as the larger of its encoded
    matrix and its ``kspace_encoding_step_1`` limit, since an undersampled grid
    still needs every line; a non-Cartesian one has as many as the limit, which
    counts shots, or the encoded matrix when no limit is stated.  The partition
    count is always the larger of the two.  The reconstructed matrix and field
    of view fall back to the encoded space when the header has no ``reconSpace``.
    """
    encodings = list(getattr(header, "encoding", None) or ())
    if not 0 <= index < len(encodings):
        raise ValueError(
            f"the header describes {len(encodings)} encoding spaces, not one numbered {index}"
        )
    encoding = encodings[index]
    encoded = encoding.encodedSpace
    recon = getattr(encoding, "reconSpace", None) or encoded
    limits = getattr(encoding, "encodingLimits", None)

    extents = [(name, _limit(limits, name)) for name in LOOP_COUNTERS]
    extents = [(name, extent) for name, extent in extents if extent > 1]
    views = _limit(limits, "kspace_encoding_step_1")
    partitions = _limit(limits, "kspace_encoding_step_2")
    trajectory = getattr(encoding, "trajectory", None)
    name = getattr(trajectory, "name", None) or str(trajectory or "cartesian")
    cartesian = name.rsplit(".", 1)[-1].upper() == "CARTESIAN"
    phase_encodes = (
        max(views, int(encoded.matrixSize.y)) if cartesian else views or int(encoded.matrixSize.y)
    )
    extents += [
        ("partition", max(partitions, int(encoded.matrixSize.z))),
        ("phase_encode", max(phase_encodes, 1)),
    ]

    matrix = tuple(int(getattr(recon.matrixSize, axis)) for axis in "xyz")
    fov = getattr(recon, "fieldOfView_mm", None)
    fov_mm = None if fov is None else tuple(float(getattr(fov, axis) or 0.0) for axis in "xyz")
    return {
        "extents": tuple(extents),
        "readout": int(encoded.matrixSize.x),
        "matrix": matrix,
        "fov_mm": fov_mm,
        "fov_m": None if fov_mm is None or not all(fov_mm) else tuple(1e-3 * v for v in fov_mm),
    }


def _limit(limits: Any, name: str) -> int:
    """Extent of one encoding limit, or 0 when the header does not state it."""
    entry = getattr(limits, name, None) if limits is not None else None
    maximum = getattr(entry, "maximum", None) if entry is not None else None
    return 0 if maximum is None else int(maximum) + 1


def _readouts(head: np.ndarray) -> dict[str, torch.Tensor]:
    idx = head["idx"]
    table = {name: idx[name] for name in idx.dtype.names if name != "user"}
    for name in (
        "flags",
        "scan_counter",
        "acquisition_time_stamp",
        "physiology_time_stamp",
        "sample_time_us",
        "center_sample",
        "position",
        "read_dir",
        "phase_dir",
        "slice_dir",
        "patient_table_position",
    ):
        table[name] = head[name]
    out = {}
    for name, values in table.items():
        values = np.asarray(values)
        if values.dtype.kind == "u":
            values = values.astype(np.int64)
        elif values.dtype.kind == "f":
            values = values.astype(np.float64)
        out[name] = torch.from_numpy(np.ascontiguousarray(values))
    return out
