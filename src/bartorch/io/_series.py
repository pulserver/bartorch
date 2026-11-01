"""Image series as an analysis reads them: magnitude and phase paired, phase in radians.

A scanner, or a reconstruction of one's own, stores the parts of a complex
image -- magnitude, phase, real, imaginary -- as images of their own, in one
series or several, and the phase as integers whose span stands for a turn.
:func:`read_series` puts them back together as one complex tensor, with the
geometry and the acquisition parameters an analysis of it needs.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from ._dicom import _number, _read_files, _select, _stack
from ._nifti import _sidecar, _stem

#: The parts of a complex image, as DICOM's ComplexImageComponent names them.
_PARTS = ("MAGNITUDE", "PHASE", "REAL", "IMAGINARY")

#: Spellings of each part in ImageType values, BIDS ``part-`` entities and GE's
#: private image-type tag (0043,102F).
_SPELLINGS = {
    "M": "MAGNITUDE",
    "MAG": "MAGNITUDE",
    "MAGNITUDE": "MAGNITUDE",
    "P": "PHASE",
    "PH": "PHASE",
    "PHASE": "PHASE",
    "R": "REAL",
    "REAL": "REAL",
    "I": "IMAGINARY",
    "IMAG": "IMAGINARY",
    "IMAGINARY": "IMAGINARY",
}
_GE_PARTS = {0: "MAGNITUDE", 1: "PHASE", 2: "REAL", 3: "IMAGINARY"}

#: Spans phase is commonly stored in, each standing for one turn: radians,
#: degrees, milliradians, and the 12-bit integers of most scanners, unsigned
#: and as Siemens rescales them.
_PHASE_RANGES = (
    (-math.pi, math.pi),
    (-180.0, 180.0),
    (-1000 * math.pi, 1000 * math.pi),
    (0.0, 4096.0),
    (-4096.0, 4096.0),
)

#: Fraction of an assumed span the phase of an image has to fill.  Wrapped phase
#: fills its span: the noise outside the object alone is spread over all of it.
_MIN_LEVELS = 256
_COVERAGE = 0.8


@dataclass(frozen=True)
class Acquisition:
    """What an analysis of an image series needs to know about how it was acquired.

    Attributes
    ----------
    echo_time, inversion_time, repetition_time : torch.Tensor
        ``float64``, one per contrast, in milliseconds; NaN where the files do
        not state one.
    flip_angle : torch.Tensor
        ``float64``, one per contrast, in degrees; NaN where not stated.
    voxel_size : tuple of float
        Voxel spacing in millimetres along the image's last three axes,
        ``(z, y, x)``.
    b0_direction : tuple of float
        Unit vector of the main field along the image's last three axes,
        ``(z, y, x)``, taking the field along the patient's head-foot axis.
    field_strength : float
        In tesla; NaN where not stated.
    vendor : str or None
        The manufacturer the files name.
    body_part : str or None
        The body part examined the files name, upper case.
    phase_range : tuple of float or None
        The span of stored phase values taken to be one turn, mapped to
        ``[-pi, pi)``; ``None`` when the series has no phase.
    """

    echo_time: torch.Tensor
    inversion_time: torch.Tensor
    repetition_time: torch.Tensor
    flip_angle: torch.Tensor
    voxel_size: tuple[float, float, float]
    b0_direction: tuple[float, float, float]
    field_strength: float
    vendor: str | None
    body_part: str | None
    phase_range: tuple[float, float] | None


class Series(NamedTuple):
    """An image series with its geometry and acquisition."""

    image: torch.Tensor
    affine: torch.Tensor
    acquisition: Acquisition
    header: Any


def read_series(
    path: str | Sequence[str],
    *,
    series: Any = None,
    phase_range: tuple[float, float] | None = None,
) -> Series:
    """Read a DICOM or NIfTI image series, pairing magnitude with phase or real with imaginary.

    Each image is assigned to a part -- magnitude, phase, real or imaginary --
    by, in order, DICOM's ``ComplexImageComponent``, GE's private image-type
    tag, Philips's private one, and the values of ``ImageType``; a NIfTI file by
    its sidecar's ``ImageType``, then a BIDS ``part-`` entity, then ``phase``,
    ``real`` or ``imag`` in its name.  An image none of these place is
    magnitude.  The parts have to cover the same contrasts and slices, and agree
    on every timing both state; a timing one part does not state is taken from
    another.

    Phase is scaled to radians from the span of stored values that stands for
    one turn.  Given ``phase_range``, that span is used.  Otherwise it is the
    narrowest of radians, degrees, milliradians, ``[0, 4096)`` and
    ``[-4096, 4096)`` that holds every value, else the span of the stored
    integer type after its rescale, else the values' own span.  Wrapped phase fills its
    span, so a warning is issued when the values fill less than 80% of the span
    used, or fall outside a given one: a reconstruction of one's own may scale
    phase in a way no rule can guess.  A warning is also issued when the phase
    takes fewer than 256 distinct values: phase rounded on export (to whole
    radians, say) carries no field.

    Parameters
    ----------
    path : str or sequence of str
        A DICOM directory, searched recursively, or DICOM files; or ``.nii`` and
        ``.nii.gz`` files, one per part and contrast or with contrasts along the
        fourth axis, each with an optional JSON sidecar of the same name.
    series : int, str or sequence, default=None
        DICOM only: ``SeriesNumber`` or ``SeriesInstanceUID`` of the series to
        read, or several, read as one.  ``None`` reads every series the files
        hold, provided no two hold the same part, as a scanner that stores
        magnitude and phase as series of their own does.
    phase_range : tuple of float, default=None
        Stored phase values ``(low, high)`` that stand for ``-pi`` and ``pi``.

    Returns
    -------
    Series
        A named tuple of

        ``image``
            ``complex64`` when the series holds phase or real and imaginary
            parts, ``float32`` when it holds magnitude only;
            ``(contrasts, z, y, x)``.  Phase without magnitude reads as unit
            magnitude.
        ``affine``
            ``float64`` ``(4, 4)``, from voxel indices ``(x, y, z)`` to RAS
            millimetres.
        ``acquisition``
            The :class:`Acquisition` the files describe.
        ``header``
            The first DICOM dataset or NIfTI header read, a template for the
            writers.

    Raises
    ------
    ValueError
        If two series hold the same part and ``series`` is not given, the parts
        cover different contrasts, slices or geometry, a part is missing its
        counterpart (real without imaginary), or ``phase_range`` is empty.
    ImportError
        If ``pydicom`` or ``nibabel`` is not installed.
    """
    if phase_range is not None:
        low, high = (float(v) for v in phase_range)
        if not high > low:
            raise ValueError(f"phase_range has to rise, got {phase_range!r}")
        phase_range = (low, high)
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if paths and all(str(p).endswith((".nii", ".nii.gz")) for p in paths):
        parts, affine, timings, fields, header = _nifti_parts(paths)
    else:
        parts, affine, timings, fields, header = _dicom_parts(path, series)

    image, used = _combine(parts, phase_range, fields.pop("_ranges"))
    linear = affine[:3, :3].numpy()
    spacing = np.linalg.norm(linear, axis=0)
    field = linear[2] / spacing
    field = field / max(np.linalg.norm(field), 1e-12)
    acquisition = Acquisition(
        echo_time=timings["echo_time"],
        inversion_time=timings["inversion_time"],
        repetition_time=timings["repetition_time"],
        flip_angle=timings["flip_angle"],
        voxel_size=tuple(float(v) for v in spacing[::-1]),
        b0_direction=tuple(float(v) for v in field[::-1]),
        field_strength=_number(fields.get("MagneticFieldStrength")),
        vendor=_text(fields.get("Manufacturer")),
        body_part=(_text(fields.get("BodyPartExamined")) or "").upper() or None,
        phase_range=used,
    )
    return Series(torch.from_numpy(image), affine, acquisition, header)


def _dicom_parts(path: Any, series: Any) -> tuple:
    pydicom = import_module("pydicom")
    datasets = _read_files(pydicom, path)
    if series is not None:
        datasets = _select(datasets, series)
    by_part: dict[str, list[Any]] = {}
    owners: dict[str, set[str]] = {}
    for ds in datasets:
        part = _dicom_part(ds)
        by_part.setdefault(part, []).append(ds)
        owners.setdefault(part, set()).add(str(ds.SeriesInstanceUID))
    shared = [part for part, uids in owners.items() if len(uids) > 1]
    if series is None and shared:
        _select(datasets, None)  # raises, listing the series
    parts, affine, timings, header, keys = {}, None, None, None, None
    for part in _PARTS:
        if part not in by_part:
            continue
        stack, part_affine, part_timings, first = _stack(by_part[part])
        if affine is None:
            affine, timings, header = part_affine, part_timings, first
            keys = stack.shape
        else:
            _same(part, stack.shape, keys, part_affine, affine, part_timings, timings)
        parts[part] = stack
    first = by_part[next(p for p in _PARTS if p in by_part)][0]
    fields = {
        key: first.get(key) for key in ("MagneticFieldStrength", "Manufacturer", "BodyPartExamined")
    }
    fields["_ranges"] = _stored_ranges(by_part.get("PHASE", [None])[0])
    return parts, affine, timings, fields, header


def _nifti_parts(paths: list[Any]) -> tuple:
    from ._nifti import read_nifti

    nibabel = import_module("nibabel")
    by_part: dict[str, list[Path]] = {}
    for file in map(Path, paths):
        by_part.setdefault(_nifti_part(file), []).append(file)
    parts, affine, timings, header, keys = {}, None, None, None, None
    for part in _PARTS:
        if part not in by_part:
            continue
        files = sorted(by_part[part], key=lambda f: (_number(_sidecar(f).get("EchoTime")), f.name))
        read = read_nifti(files)
        stack = read.image.numpy()
        if affine is None:
            affine, timings, header = read.affine, read.timings, read.header
            keys = stack.shape
        else:
            _same(part, stack.shape, keys, read.affine, affine, read.timings, timings)
        parts[part] = stack
    first = by_part[next(p for p in _PARTS if p in by_part)][0]
    fields = dict(_sidecar(first))
    ranges = []
    if "PHASE" in by_part:
        loaded = nibabel.load(by_part["PHASE"][0])
        dtype = loaded.get_data_dtype()
        if np.issubdtype(dtype, np.integer):
            slope, inter = loaded.header.get_slope_inter()
            slope = 1.0 if slope is None or not np.isfinite(slope) else float(slope)
            inter = 0.0 if inter is None or not np.isfinite(inter) else float(inter)
            info = np.iinfo(dtype)
            ranges.append(_scaled((float(info.min), float(info.max) + 1.0), slope, inter))
    fields["_ranges"] = ranges
    return parts, affine, timings, fields, header


def _same(part, shape, keys, part_affine, affine, part_timings, timings) -> None:
    if shape != keys:
        raise ValueError(f"the {part.lower()} images are {shape}, the others {keys}")
    if not np.allclose(part_affine.numpy(), affine.numpy(), atol=1e-3):
        raise ValueError(f"the {part.lower()} images lie elsewhere than the others")
    for name, values in timings.items():
        theirs = part_timings[name]
        both = ~(torch.isnan(theirs) | torch.isnan(values))
        if not torch.allclose(theirs[both], values[both], atol=1e-3):
            raise ValueError(
                f"the {part.lower()} images have {name} {theirs.tolist()}, "
                f"the others {values.tolist()}"
            )
        # A part whose files state no timing takes the other part's.
        values.copy_(torch.where(torch.isnan(values), theirs, values))


def _dicom_part(ds: Any) -> str:
    component = _text(ds.get("ComplexImageComponent"))
    if component and component.upper() in _PARTS:
        return component.upper()
    ge = ds.get((0x0043, 0x102F)) if hasattr(ds, "get") else None
    if ge is not None and "GE" in (_text(ds.get("Manufacturer")) or "").upper():
        try:
            return _GE_PARTS[int(_number(ge.value))]
        except (KeyError, TypeError, ValueError):
            pass
    philips = ds.get((0x2005, 0x1011)) if hasattr(ds, "get") else None
    if philips is not None and "PHILIPS" in (_text(ds.get("Manufacturer")) or "").upper():
        found = _SPELLINGS.get((_text(philips.value) or "").strip().upper())
        if found:
            return found
    return _image_type_part(ds.get("ImageType")) or "MAGNITUDE"


def _image_type_part(values: Any) -> str | None:
    """The part an ``ImageType`` names, past its first two values (ORIGINAL\\PRIMARY)."""
    if values is None:
        return None
    values = [values] if isinstance(values, str) else list(values)
    for value in values[2:]:
        token = str(value).strip().upper()
        if token in _SPELLINGS:
            return _SPELLINGS[token]
        head = token.split("_")[0]
        if "_" in token and head in ("M", "P", "R", "I"):
            return _SPELLINGS[head]
    return None


def _nifti_part(file: Path) -> str:
    found = _image_type_part(_sidecar(file).get("ImageType"))
    if found:
        return found
    name = _stem(file).name.lower()
    for entity in name.split("_"):
        if entity.startswith("part-") and entity[5:].upper() in _SPELLINGS:
            return _SPELLINGS[entity[5:].upper()]
    tokens = name.replace("-", "_").split("_")
    if "phase" in name or "ph" in tokens:
        return "PHASE"
    if "imag" in name:
        return "IMAGINARY"
    if "real" in tokens:
        return "REAL"
    return "MAGNITUDE"


def _stored_ranges(ds: Any) -> list[tuple[float, float]]:
    """The span the stored integers of a DICOM image can take, after its rescale."""
    if ds is None:
        return []
    bits = _number(ds.get("BitsStored"))
    if math.isnan(bits):
        return []
    signed = _number(ds.get("PixelRepresentation")) == 1
    span = (-(2.0 ** (bits - 1)), 2.0 ** (bits - 1)) if signed else (0.0, 2.0**bits)
    slope = _number(ds.get("RescaleSlope"))
    inter = _number(ds.get("RescaleIntercept"))
    return [_scaled(span, 1.0 if math.isnan(slope) else slope, 0.0 if math.isnan(inter) else inter)]


def _scaled(span: tuple[float, float], slope: float, inter: float) -> tuple[float, float]:
    ends = sorted((span[0] * slope + inter, span[1] * slope + inter))
    return (ends[0], ends[1])


def _combine(
    parts: dict[str, np.ndarray],
    phase_range: tuple[float, float] | None,
    ranges: list[tuple[float, float]],
) -> tuple[np.ndarray, tuple[float, float] | None]:
    if ("REAL" in parts) != ("IMAGINARY" in parts):
        raise ValueError("real and imaginary parts come together; one of them is missing")
    if "REAL" in parts:
        image = parts["REAL"] + 1j * parts["IMAGINARY"]
        return image.astype(np.complex64), None
    if "PHASE" not in parts:
        return parts["MAGNITUDE"].astype(np.float32, copy=False), None
    stored = parts["PHASE"]
    span = _phase_span(stored, phase_range, ranges)
    _check_levels(stored, span)
    low, high = span
    phase = (stored - np.float32(low)) * np.float32(2 * math.pi / (high - low)) - np.float32(
        math.pi
    )
    magnitude = parts.get("MAGNITUDE")
    if magnitude is None:
        warnings.warn(
            "the series holds phase without magnitude; magnitude reads as 1", stacklevel=3
        )
        magnitude = np.ones_like(stored)
    image = magnitude * np.exp(1j * phase)
    return image.astype(np.complex64), span


def _check_levels(stored: np.ndarray, span: tuple[float, float]) -> None:
    """Warn when the phase takes few distinct values: rounded on export (to whole radians,
    say), it no longer carries a field."""
    finite = stored[np.isfinite(stored)].ravel()
    if finite.size > 1_000_000:
        finite = finite[:: finite.size // 1_000_000]
    levels = np.unique(finite).size
    if levels < min(_MIN_LEVELS, finite.size // 4):
        spread = float(finite.max() - finite.min()) / (span[1] - span[0])
        step = 2 * math.pi * spread / max(levels - 1, 1)
        warnings.warn(
            f"the phase holds only {levels} distinct values, about {step:.2g} rad apart: "
            "it looks rounded on export and cannot give a field map",
            stacklevel=4,
        )


def _phase_span(
    stored: np.ndarray, given: tuple[float, float] | None, extra: list[tuple[float, float]]
) -> tuple[float, float]:
    finite = stored[np.isfinite(stored)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    if given is not None:
        span = given
        tolerance = 0.01 * (span[1] - span[0])
        if low < span[0] - tolerance or high > span[1] + tolerance:
            warnings.warn(
                f"phase values reach [{low:g}, {high:g}], outside phase_range "
                f"[{span[0]:g}, {span[1]:g}]",
                stacklevel=4,
            )
            return span
    else:
        # The common spans first: a stored type's span is also what a value range
        # quantized onto the whole type looks like, which says nothing of a turn.
        candidates = [
            *sorted(_PHASE_RANGES, key=lambda s: s[1] - s[0]),
            *sorted(extra, key=lambda s: s[1] - s[0]),
        ]
        span = next(
            (
                s
                for s in candidates
                if low >= s[0] - 0.01 * (s[1] - s[0]) and high <= s[1] + 0.01 * (s[1] - s[0])
            ),
            None,
        )
        if span is None:
            warnings.warn(
                f"phase values [{low:g}, {high:g}] fit no known scaling; their own span is "
                "taken as one turn; pass phase_range= if the reconstruction states it",
                stacklevel=4,
            )
            return (low, high) if high > low else (low, low + 1.0)
    filled = (high - low) / (span[1] - span[0])
    if filled < _COVERAGE:
        warnings.warn(
            f"phase values fill [{low:g}, {high:g}], {100 * filled:.0f}% "
            f"of the span [{span[0]:g}, {span[1]:g}] taken as one turn; pass phase_range= if "
            "the reconstruction scales phase otherwise",
            stacklevel=4,
        )
    return span


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value") and not isinstance(value, str):
        value = value.value
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        value = value[0] if len(value) else None
    text = None if value is None else str(value).strip()
    return text or None
