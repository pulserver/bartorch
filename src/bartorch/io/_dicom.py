"""DICOM MR image series, read into and written from image tensors.

The series fields taken from an MRD header are adapted from the converter of
python-ismrmrd-server (Copyright (c) 2024 Kelvin Chow; MIT, see
``LICENSES/python-ismrmrd-server-MIT.txt``), as pulserver carries it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from ._geometry import lps_affine, to_lps
from ._optional import require

#: Timing fields a series is sorted into contrasts by, as :attr:`Images.timings`
#: names them, with the DICOM keyword each is read from.  DICOM states times in
#: milliseconds and the flip angle in degrees.
_TIMINGS = {
    "inversion_time": "InversionTime",
    "echo_time": "EchoTime",
    "repetition_time": "RepetitionTime",
    "flip_angle": "FlipAngle",
}
#: Further fields that tell two images of one slice apart.
_REPEATS = ("TriggerTime", "TemporalPositionIdentifier", "DiffusionBValue")

#: Third ``ImageType`` value to the value of GE's private image-type tag (0043,102F).
_GE_IMAGE_TYPES = {"M": 0, "P": 1, "R": 2, "I": 3}

#: Fields a template dataset lends every image of a new series.
_INHERITED = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientSize",
    "PatientWeight",
    "PatientPosition",
    "StudyInstanceUID",
    "StudyDate",
    "StudyTime",
    "StudyID",
    "StudyDescription",
    "AccessionNumber",
    "ReferringPhysicianName",
    "BodyPartExamined",
    "FrameOfReferenceUID",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "ImagingFrequency",
    "InstitutionName",
    "StationName",
    "DeviceSerialNumber",
)

#: DICOM value representations whose values are strings.
_STRING_VRS = {"AE", "AS", "CS", "DA", "DT", "LO", "LT", "PN", "SH", "ST", "TM", "UI", "UT"}


class Images(NamedTuple):
    """Images sorted into contrasts, with their geometry and acquisition timings."""

    image: torch.Tensor
    affine: torch.Tensor
    timings: dict[str, torch.Tensor]
    header: Any


def read_dicom(path: str | Sequence[str], *, series: Any = None) -> Images:
    """Read a DICOM MR image series into a tensor sorted by contrast and slice.

    Images are grouped into contrasts by inversion time, echo time, repetition
    time and flip angle, then trigger time, temporal position and b-value, and
    contrasts are ordered by those fields in that order; within a contrast,
    slices are ordered along the slice normal.  Every contrast has to cover the
    same slices.

    Parameters
    ----------
    path : str or sequence of str
        A directory, searched recursively, or the files themselves.  Files that
        are not DICOM are skipped.
    series : int, str or sequence, default=None
        ``SeriesNumber`` or ``SeriesInstanceUID`` of the series to read, or
        several of either, read as one: a protocol that stores each echo or
        inversion time as a series of its own.  ``None`` reads the only series
        the files hold.

    Returns
    -------
    Images
        A named tuple of

        ``image``
            ``float32``, ``(contrasts, slices, rows, columns)``, with the
            rescale slope and intercept applied.
        ``affine``
            ``float64`` ``(4, 4)``, from voxel indices ``(column, row, slice)``
            to RAS coordinates in millimetres.
        ``timings``
            ``inversion_time``, ``echo_time`` and ``repetition_time`` in
            milliseconds and ``flip_angle`` in degrees, one per contrast; NaN
            where the files do not state one.
        ``header``
            The first dataset read, a template for :func:`write_dicom`.

    Raises
    ------
    ValueError
        If no series matches, several do and ``series`` is not given, the
        files are multi-frame, two images share a contrast and a slice, or the
        contrasts cover different slices.
    """
    pydicom = require("pydicom")
    datasets = _read_files(pydicom, path)
    datasets = _select(datasets, series)
    if any(int(getattr(ds, "NumberOfFrames", 1) or 1) > 1 for ds in datasets):
        raise ValueError(
            "multi-frame (enhanced) DICOM is not read; convert it with dcm2niix and use read_nifti"
        )

    first = datasets[0]
    orientation = np.asarray(first.ImageOrientationPatient, float)
    along_row, along_column = orientation[:3], orientation[3:]
    normal = np.cross(along_row, along_column)

    def location(ds: Any) -> float:
        return round(float(np.dot(np.asarray(ds.ImagePositionPatient, float), normal)), 3)

    def contrast(ds: Any) -> tuple:
        keys = [*_TIMINGS.values(), *_REPEATS]
        return tuple(_number(getattr(ds, key, None)) for key in keys)

    contrasts = sorted(
        {contrast(ds) for ds in datasets},
        key=lambda key: tuple(-math.inf if math.isnan(v) else v for v in key),
    )
    locations = sorted({location(ds) for ds in datasets})
    grid: dict[tuple, Any] = {}
    for ds in datasets:
        key = (contrast(ds), location(ds))
        if key in grid:
            raise ValueError(
                f"two images share a contrast "
                f"{dict(zip((*_TIMINGS, *_REPEATS), key[0]))} and the slice at {key[1]} mm; "
                "select one series, or one image type, with series="
            )
        grid[key] = ds
    missing = [(c, s) for c in contrasts for s in locations if (c, s) not in grid]
    if missing:
        raise ValueError(
            f"{len(missing)} contrast and slice combinations have no image; "
            "the contrasts cover different slices"
        )

    rows, columns = int(first.Rows), int(first.Columns)
    image = np.empty((len(contrasts), len(locations), rows, columns), np.float32)
    for c, key in enumerate(contrasts):
        for s, where in enumerate(locations):
            ds = grid[(key, where)]
            pixels = ds.pixel_array.astype(np.float32)
            image[c, s] = pixels * float(getattr(ds, "RescaleSlope", 1) or 1) + float(
                getattr(ds, "RescaleIntercept", 0) or 0
            )

    first_slice = grid[(contrasts[0], locations[0])]
    last_slice = grid[(contrasts[0], locations[-1])]
    affine = _series_affine(first_slice, last_slice, len(locations))
    timings = {
        name: torch.tensor([key[n] for key in contrasts], dtype=torch.float64)
        for n, name in enumerate(_TIMINGS)
    }
    return Images(torch.from_numpy(image), affine, timings, first)


def _series_affine(first: Any, last: Any, slices: int) -> torch.Tensor:
    """Affine of a stack of ``slices`` images from its first and its last.

    One slice takes ``SpacingBetweenSlices``, or ``SliceThickness``, along the
    normal as its step.
    """
    orientation = np.asarray(first.ImageOrientationPatient, float)
    along_row, along_column = orientation[:3], orientation[3:]
    row_spacing, column_spacing = (float(v) for v in first.PixelSpacing)
    steps = np.zeros((3, 3))
    steps[:, 0] = along_row * column_spacing
    steps[:, 1] = along_column * row_spacing
    start = np.asarray(first.ImagePositionPatient, float)
    if slices > 1:
        end = np.asarray(last.ImagePositionPatient, float)
        steps[:, 2] = (end - start) / (slices - 1)
    else:
        thickness = _number(getattr(first, "SpacingBetweenSlices", None))
        if math.isnan(thickness):
            thickness = _number(getattr(first, "SliceThickness", None))
        normal = np.cross(along_row, along_column)
        steps[:, 2] = normal * (1.0 if math.isnan(thickness) else thickness)
    return lps_affine(steps, start)


def to_dicom(image: Any, affine: Any, *, header: Any = None, **fields: Any) -> list[Any]:
    """Convert a real image to the datasets of one DICOM MR image series.

    Parameters
    ----------
    image : torch.Tensor or numpy.ndarray
        Real, ``(*leading, z, y, x)`` or ``(y, x)``.  Each ``(y, x)`` plane is
        an image; leading axes are numbered through after the slices.
        Floating-point values are quantised to ``uint16`` over the whole series
        and ``RescaleSlope`` and ``RescaleIntercept`` recover them; integer
        values are stored as they are.
    affine : torch.Tensor or numpy.ndarray
        ``(4, 4)``, from voxel indices ``(x, y, z)`` to RAS coordinates in
        millimetres.  ``x`` runs along the rows, ``y`` down the columns.
    header : ismrmrd.xsd.ismrmrdHeader or pydicom.Dataset, default=None
        Where the patient, study, frame-of-reference and equipment fields come
        from: the MRD header of the scan, or a dataset of it such as
        :attr:`Images.header`.  The series always gets a new
        ``SeriesInstanceUID``.
    **fields
        DICOM keywords and values set on every image, such as
        ``SeriesDescription``, ``SeriesNumber``, ``EchoTime`` or ``ImageType``;
        they are applied last.

    Returns
    -------
    list of pydicom.Dataset
        One per image, ``InstanceNumber`` counting from 1.

    Raises
    ------
    TypeError
        If the image is complex: write its magnitude, phase, real or imaginary
        part as a series of its own.
    """
    pydicom = require("pydicom")
    pixels = np.asarray(torch.as_tensor(image).detach().cpu().numpy())
    if np.iscomplexobj(pixels):
        raise TypeError(
            "DICOM stores real images; write the magnitude or the phase of a complex one"
        )
    if pixels.ndim == 2:
        pixels = pixels[None]
    if pixels.ndim < 3:
        raise ValueError(f"an image is (y, x) or (*leading, z, y, x), got shape {pixels.shape}")
    slices, rows, columns = pixels.shape[-3:]
    planes = pixels.reshape(-1, rows, columns)
    stored, rescale = _quantize(planes)
    window = (
        (float(np.percentile(planes, 5)), float(np.percentile(planes, 95)))
        if planes.size
        else (0.0, 0.0)
    )

    steps, origin = to_lps(affine)
    column_spacing = float(np.linalg.norm(steps[:, 0]))
    row_spacing = float(np.linalg.norm(steps[:, 1]))
    spacing = float(np.linalg.norm(steps[:, 2]))
    orientation = [*(steps[:, 0] / column_spacing), *(steps[:, 1] / row_spacing)]
    normal = np.cross(steps[:, 0], steps[:, 1])
    normal /= np.linalg.norm(normal)

    template = _template(pydicom, header)
    series_uid = pydicom.uid.generate_uid()
    datasets = []
    for number, plane in enumerate(stored):
        ds = pydicom.Dataset()
        for elem in template:
            ds.add(elem)
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.SOPClassUID = pydicom.uid.MRImageStorage
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "MR"
        for keyword, default in (
            ("ImageType", ["ORIGINAL", "PRIMARY", "M"]),
            ("SeriesNumber", 1),
            ("SeriesDescription", ""),
        ):
            if keyword not in ds:
                setattr(ds, keyword, default)
        ds.InstanceNumber = number + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = rows, columns
        ds.BitsAllocated = ds.BitsStored = 8 * plane.dtype.itemsize
        ds.HighBit = ds.BitsStored - 1
        ds.PixelRepresentation = int(plane.dtype.kind == "i")
        ds.PixelSpacing = [round(row_spacing, 6), round(column_spacing, 6)]
        ds.SliceThickness = round(spacing, 6)
        ds.SpacingBetweenSlices = round(spacing, 6)
        position = origin + (number % slices) * steps[:, 2]
        ds.ImagePositionPatient = [round(float(v), 6) for v in position]
        ds.ImageOrientationPatient = [round(float(v), 6) for v in orientation]
        ds.SliceLocation = round(float(np.dot(position, normal)), 6)
        if rescale is not None:
            ds.RescaleIntercept, ds.RescaleSlope = (f"{v:.10g}" for v in rescale)
            ds.RescaleType = "US"
        ds.WindowCenter = f"{0.5 * (window[0] + window[1]):.6g}"
        ds.WindowWidth = f"{max(window[1] - window[0], 1e-12):.6g}"
        for keyword, value in fields.items():
            setattr(ds, keyword, value)
        if "GE" in str(ds.get("Manufacturer", "")).upper():
            kind = str(ds.ImageType[2]) if len(ds.ImageType) > 2 else "M"
            ds.add(pydicom.DataElement((0x0043, 0x102F), "SS", _GE_IMAGE_TYPES.get(kind, 0)))
        ds.PixelData = np.ascontiguousarray(plane).tobytes()
        datasets.append(_string_vrs(pydicom, ds))
    return datasets


def write_dicom(
    path: str, image: Any, affine: Any, *, header: Any = None, **fields: Any
) -> list[Path]:
    """Write a real image as one DICOM MR image series, a file per image.

    Parameters and conversion are those of :func:`to_dicom`.  Files are named
    ``EX<StudyID>_<SeriesNumber:02>_<SeriesDescription>_<InstanceNumber:03>.dcm``
    in the directory ``path``, which is created.

    Returns
    -------
    list of pathlib.Path
        The files written, in instance order.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for ds in to_dicom(image, affine, header=header, **fields):
        study = ds.get("StudyID", None) or "0"
        series = ds.get("SeriesNumber", None) or 0
        description = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(ds.get("SeriesDescription", ""))
        )
        instance = int(ds.InstanceNumber)
        name = f"EX{study}_{float(series):02.0f}_{description or 'image'}_{instance:03d}.dcm"
        ds.save_as(directory / name, enforce_file_format=True)
        written.append(directory / name)
    return written


def _read_files(pydicom: Any, path: str | Sequence[str]) -> list[Any]:
    if isinstance(path, (str, Path)):
        root = Path(path)
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    else:
        files = [Path(p) for p in path]
    datasets = []
    for file in files:
        try:
            ds = pydicom.dcmread(file)
        except pydicom.errors.InvalidDicomError:
            continue
        if "PixelData" in ds and "ImagePositionPatient" in ds:
            datasets.append(ds)
    if not datasets:
        raise ValueError(f"{path} holds no DICOM images")
    return datasets


def _select(datasets: list[Any], series: Any) -> list[Any]:
    found = {}
    for ds in datasets:
        found.setdefault(
            str(ds.SeriesInstanceUID), (ds.get("SeriesNumber"), ds.get("SeriesDescription", ""))
        )
    if series is None:
        if len(found) > 1:
            listing = ", ".join(
                f"{number} ({description})" for number, description in found.values()
            )
            raise ValueError(f"the files hold {len(found)} series: {listing}; choose with series=")
        return datasets
    wanted = [series] if isinstance(series, (int, str)) else list(series)
    chosen = [
        ds
        for ds in datasets
        if str(ds.SeriesInstanceUID) in map(str, wanted)
        or (
            ds.get("SeriesNumber") is not None
            and int(ds.SeriesNumber) in [w for w in wanted if isinstance(w, int)]
        )
    ]
    if not chosen:
        raise ValueError(f"no series matches {series!r}")
    return chosen


def _number(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        value = value[0] if len(value) else None
        return _number(value)
    return float(value)


def _template(pydicom: Any, header: Any) -> Any:
    """Fields a new series inherits: from a DICOM dataset, or from an MRD header."""
    ds = pydicom.Dataset()
    if header is None:
        pass
    elif isinstance(header, pydicom.Dataset):
        for keyword in _INHERITED:
            if keyword in header:
                setattr(ds, keyword, header.data_element(keyword).value)
    else:
        _from_mrd(ds, header)
    if "StudyInstanceUID" not in ds:
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
    if not pydicom.uid.UID(str(ds.get("FrameOfReferenceUID", ""))).is_valid:
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    return ds


def _from_mrd(ds: Any, header: Any) -> None:
    """Patient, study, series and equipment fields of an MRD header.

    A section that does not convert is logged and skipped.  On a GE system the
    series number is the header's ``measurementID``.
    """
    sections = {
        "subjectInformation": {
            "patientName": "PatientName",
            "patientWeight_kg": "PatientWeight",
            "patientHeight_m": "PatientSize",
            "patientID": "PatientID",
            "patientBirthdate": ("PatientBirthDate", _date),
            "patientGender": "PatientSex",
        },
        "studyInformation": {
            "studyDate": ("StudyDate", _date),
            "studyTime": ("StudyTime", _time),
            "studyID": "StudyID",
            "accessionNumber": ("AccessionNumber", str),
            "referringPhysicianName": "ReferringPhysicianName",
            "studyDescription": "StudyDescription",
            "studyInstanceUID": "StudyInstanceUID",
            "bodyPartExamined": "BodyPartExamined",
        },
        "measurementInformation": {
            "seriesDate": ("SeriesDate", _date),
            "seriesTime": ("SeriesTime", _time),
            "patientPosition": ("PatientPosition", lambda v: getattr(v, "name", str(v))),
            "protocolName": "ProtocolName",
            "sequenceName": "SequenceName",
            "seriesDescription": "SeriesDescription",
            "frameOfReferenceUID": "FrameOfReferenceUID",
        },
        "acquisitionSystemInformation": {
            "systemVendor": "Manufacturer",
            "systemModel": "ManufacturerModelName",
            "systemFieldStrength_T": "MagneticFieldStrength",
            "institutionName": "InstitutionName",
            "stationName": "StationName",
            "deviceID": "DeviceSerialNumber",
            "deviceSerialNumber": "DeviceSerialNumber",
        },
        "experimentalConditions": {
            "H1resonanceFrequency_Hz": ("ImagingFrequency", lambda v: v / 1e6),
        },
    }
    for section, entries in sections.items():
        block = getattr(header, section, None)
        if block is None:
            continue
        try:
            for attribute, target in entries.items():
                value = getattr(block, attribute, None)
                if value is None:
                    continue
                keyword, convert = target if isinstance(target, tuple) else (target, lambda v: v)
                setattr(ds, keyword, convert(value))
        except Exception:  # noqa: BLE001 - a malformed section must not stop the series
            logging.getLogger(__name__).warning(
                "the MRD header's %s section did not convert", section
            )
    measurement = getattr(getattr(header, "measurementInformation", None), "measurementID", None)
    if "GE" in str(ds.get("Manufacturer", "")).upper() and measurement is not None:
        if not str(measurement).strip().isdigit():
            raise ValueError(
                "the header's measurementID is the DICOM series number on a GE "
                f"system, and {measurement!r} is not a non-negative integer"
            )
        ds.SeriesNumber = int(measurement)


def _date(value: Any) -> str:
    """An ISMRMRD ``XmlDate`` as DICOM DA (``YYYYMMDD``); any other value as ``str``."""
    if type(value).__name__ == "XmlDate":
        return value.to_date().isoformat().replace("-", "")
    return str(value)


def _time(value: Any) -> str:
    """An ISMRMRD ``XmlTime`` as DICOM TM (``HHMMSS``); any other value as ``str``."""
    if type(value).__name__ == "XmlTime":
        return value.to_time().isoformat().replace(":", "").split(".")[0]
    return str(value)


def _string_vrs(pydicom: Any, ds: Any) -> Any:
    """Convert numeric and person-name values of string-VR elements to ``str``, in sequences too."""
    for elem in ds:
        if elem.VR == "SQ":
            for item in elem.value:
                _string_vrs(pydicom, item)
        elif elem.VR in _STRING_VRS:
            value = elem.value
            if isinstance(value, (int, float, pydicom.valuerep.PersonName)):
                elem.value = str(value)
            elif isinstance(value, (list, tuple)):
                elem.value = [str(v) if isinstance(v, (int, float)) else v for v in value]
    return ds


def _quantize(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Pixels as DICOM stores them, with the ``(intercept, slope)`` that recovers the values.

    An integer image passes through; a floating-point one is mapped onto the
    full ``uint16`` range, so ``stored * slope + intercept`` is the value.
    """
    if not np.issubdtype(image.dtype, np.floating):
        if image.dtype.itemsize > 4:
            image = image.astype(np.int32)
        return image, None
    finite = image[np.isfinite(image)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    span = high - low
    if span <= 0.0:
        return np.zeros(image.shape, np.uint16), (low, 1.0)
    full = float(np.iinfo(np.uint16).max)
    stored = np.rint((np.nan_to_num(image, nan=low, posinf=high, neginf=low) - low) * (full / span))
    return stored.astype(np.uint16), (low, span / full)
