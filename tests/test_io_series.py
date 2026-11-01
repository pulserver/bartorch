"""Image series read as an analysis needs them: parts paired, phase in radians."""

from __future__ import annotations

import json
import math
import warnings

import nibabel
import numpy as np
import pydicom
import pytest
import torch

from bartorch import io

SHAPE = (3, 6, 5)  # slices, rows, columns


def _phase(echo: int) -> np.ndarray:
    """Wrapped phase filling [-pi, pi): a ramp over several turns."""
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE), indexing="ij")
    ramp = (echo + 1) * (0.9 * x + 1.7 * y + 2.3 * z) + 0.1
    return np.angle(np.exp(1j * ramp))


def _magnitude(echo: int) -> np.ndarray:
    return np.full(SHAPE, 100.0 * math.exp(-(echo + 1) * 0.2)) + np.arange(SHAPE[-1])


def _affine():
    affine = np.eye(4)
    affine[:3, :3] = np.diag([0.8, 0.9, 2.0])
    affine[:3, 3] = [-10.0, 5.0, 3.0]
    return torch.from_numpy(affine)


def _datasets(image, *, echo_time, series_uid, **fields):
    datasets = io.to_dicom(image, _affine(), EchoTime=echo_time, **fields)
    for ds in datasets:
        ds.SeriesInstanceUID = series_uid
    return datasets


def _save(directory, datasets):
    directory.mkdir(parents=True, exist_ok=True)
    for n, ds in enumerate(datasets):
        ds.save_as(directory / f"im{n:04d}.dcm", enforce_file_format=True)


def _siemens_phase(phase):
    """Stored the way Siemens stores phase: 12-bit, [0, 4096) a turn."""
    return np.clip(np.rint((phase + np.pi) / (2 * np.pi) * 4096), 0, 4095).astype(np.uint16)


def _assert_phase(image, echoes):
    for echo in range(echoes):
        error = np.angle(image[echo].numpy() * np.exp(-1j * _phase(echo)))
        assert np.abs(error).max() < 2 * np.pi / 4096 + 1e-4


@pytest.fixture
def siemens(tmp_path):
    """Two echoes, magnitude and phase as two series of their own."""
    magnitude_uid, phase_uid = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    datasets = []
    for echo in range(2):
        te = 4.0 + 5.0 * echo
        common = dict(Manufacturer="SIEMENS", MagneticFieldStrength=3.0, BodyPartExamined="HEAD")
        datasets += _datasets(
            _magnitude(echo).astype(np.uint16),
            echo_time=te,
            series_uid=magnitude_uid,
            SeriesNumber=5,
            ImageType=["ORIGINAL", "PRIMARY", "M", "ND"],
            **common,
        )
        datasets += _datasets(
            _siemens_phase(_phase(echo)),
            echo_time=te,
            series_uid=phase_uid,
            SeriesNumber=6,
            ImageType=["ORIGINAL", "PRIMARY", "P", "ND"],
            **common,
        )
    _save(tmp_path / "siemens", datasets)
    return tmp_path / "siemens"


def test_magnitude_and_phase_series_pair_into_one_complex_image(siemens):
    read = io.read_series(siemens)
    assert read.image.dtype == torch.complex64
    assert read.image.shape == (2, *SHAPE)
    np.testing.assert_allclose(read.image[1].abs().numpy(), _magnitude(1), rtol=1e-2)
    _assert_phase(read.image, 2)
    assert read.acquisition.phase_range == (0.0, 4096.0)
    np.testing.assert_allclose(read.acquisition.echo_time.numpy(), [4.0, 9.0])


def test_the_acquisition_states_field_geometry_and_anatomy(siemens):
    acquisition = io.read_series(siemens).acquisition
    assert acquisition.field_strength == 3.0
    assert acquisition.vendor == "SIEMENS"
    assert acquisition.body_part == "HEAD"
    np.testing.assert_allclose(acquisition.voxel_size, (2.0, 0.9, 0.8), atol=1e-6)
    np.testing.assert_allclose(acquisition.b0_direction, (1.0, 0.0, 0.0), atol=1e-6)


def test_the_field_direction_follows_an_oblique_slab(tmp_path):
    angle = 0.4
    affine = np.eye(4)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )
    affine[:3, :3] = rotation @ np.diag([1.0, 1.0, 2.0])
    io.write_dicom(tmp_path / "oblique", torch.ones(SHAPE), torch.from_numpy(affine))
    direction = io.read_series(tmp_path / "oblique").acquisition.b0_direction
    # The voxel axes' components along RAS z: column z is rotation[:, 2], column y rotation[:, 1].
    np.testing.assert_allclose(direction, (math.cos(angle), math.sin(angle), 0.0), atol=1e-5)


def test_one_series_of_one_part_reads_as_magnitude(siemens):
    read = io.read_series(siemens, series=5)
    assert read.image.dtype == torch.float32
    assert read.acquisition.phase_range is None


def test_phase_alone_reads_with_unit_magnitude(siemens):
    with pytest.warns(UserWarning, match="without magnitude"):
        read = io.read_series(siemens, series=6)
    np.testing.assert_allclose(read.image.abs().numpy(), 1.0, rtol=1e-6)
    _assert_phase(read.image, 2)


def test_phase_rescaled_to_signed_integers_is_recognised(tmp_path):
    """Siemens also stores phase as 0..4095 with slope 2 and intercept -4096."""
    datasets = _datasets(
        _magnitude(0).astype(np.uint16), echo_time=5.0, series_uid=pydicom.uid.generate_uid()
    )
    phase = _datasets(
        _siemens_phase(_phase(0)),
        echo_time=5.0,
        series_uid=pydicom.uid.generate_uid(),
        ImageType=["ORIGINAL", "PRIMARY", "P"],
        SeriesNumber=2,
    )
    for ds in phase:
        ds.BitsStored, ds.HighBit = 12, 11
        ds.RescaleSlope, ds.RescaleIntercept = "2", "-4096"
    _save(tmp_path / "rescaled", datasets + phase)
    read = io.read_series(tmp_path / "rescaled")
    assert read.acquisition.phase_range == (-4096.0, 4096.0)
    _assert_phase(read.image, 1)


def test_ge_parts_in_one_series_are_told_apart_by_the_private_tag(tmp_path):
    uid = pydicom.uid.generate_uid()
    complex_image = _magnitude(0) * np.exp(1j * _phase(0))
    datasets = []
    for kind, values in ((2, complex_image.real), (3, complex_image.imag)):
        for ds in _datasets(
            values.astype(np.float32),
            echo_time=5.0,
            series_uid=uid,
            Manufacturer="GE MEDICAL SYSTEMS",
            ImageType=["ORIGINAL", "PRIMARY", "OTHER"],
        ):
            ds[0x0043, 0x102F].value = kind
            datasets.append(ds)
    _save(tmp_path / "ge", datasets)
    read = io.read_series(tmp_path / "ge")
    assert read.image.dtype == torch.complex64
    np.testing.assert_allclose(read.image[0].numpy(), complex_image, atol=0.05)
    assert read.acquisition.phase_range is None


def test_philips_phase_in_radians_from_image_type(tmp_path):
    uid = pydicom.uid.generate_uid()
    datasets = _datasets(
        _magnitude(0).astype(np.float32),
        echo_time=5.0,
        series_uid=uid,
        Manufacturer="Philips Medical Systems",
        ImageType=["ORIGINAL", "PRIMARY", "M_FFE", "M", "FFE"],
    )
    datasets += _datasets(
        _phase(0).astype(np.float32),
        echo_time=5.0,
        series_uid=uid,
        Manufacturer="Philips Medical Systems",
        ImageType=["ORIGINAL", "PRIMARY", "P_FFE", "P", "FFE"],
    )
    _save(tmp_path / "philips", datasets)
    read = io.read_series(tmp_path / "philips")
    assert read.acquisition.phase_range == (-math.pi, math.pi)
    _assert_phase(read.image, 1)


def _custom(tmp_path, scale=1000):
    """A reconstruction of one's own: phase stored as [0, scale) a turn."""
    stored = np.rint((_phase(0) + np.pi) / (2 * np.pi) * scale).astype(np.uint16) % scale
    datasets = _datasets(
        _magnitude(0).astype(np.uint16), echo_time=5.0, series_uid=pydicom.uid.generate_uid()
    )
    datasets += _datasets(
        stored,
        echo_time=5.0,
        series_uid=pydicom.uid.generate_uid(),
        ImageType=["DERIVED", "PRIMARY", "P"],
        SeriesNumber=2,
    )
    _save(tmp_path / "custom", datasets)
    return tmp_path / "custom"


def test_a_custom_phase_scaling_is_read_as_stated(tmp_path):
    read = io.read_series(_custom(tmp_path), phase_range=(0, 1000))
    error = np.angle(read.image[0].numpy() * np.exp(-1j * _phase(0)))
    assert np.abs(error).max() < 2 * np.pi / 1000 + 1e-4
    assert read.acquisition.phase_range == (0.0, 1000.0)


def test_a_custom_phase_scaling_not_stated_is_warned_about(tmp_path):
    with pytest.warns(UserWarning, match="phase_range="):
        io.read_series(_custom(tmp_path))


def test_a_stated_range_the_values_overrun_is_warned_about(tmp_path):
    with pytest.warns(UserWarning, match="outside phase_range"):
        io.read_series(_custom(tmp_path), phase_range=(0, 500))
    with pytest.raises(ValueError, match="rise"):
        io.read_series(_custom(tmp_path), phase_range=(1, 1))


def test_parts_that_disagree_are_refused(tmp_path):
    magnitude = _datasets(
        np.ones(SHAPE, np.uint16), echo_time=5.0, series_uid=pydicom.uid.generate_uid()
    )
    phase = _datasets(
        np.ones(SHAPE, np.uint16),
        echo_time=6.0,
        series_uid=pydicom.uid.generate_uid(),
        ImageType=["ORIGINAL", "PRIMARY", "P"],
        SeriesNumber=2,
    )
    _save(tmp_path / "te", magnitude + phase)
    with pytest.raises(ValueError, match="echo_time"):
        io.read_series(tmp_path / "te")
    real = _datasets(
        np.ones(SHAPE, np.uint16),
        echo_time=5.0,
        series_uid=pydicom.uid.generate_uid(),
        ImageType=["ORIGINAL", "PRIMARY", "R"],
    )
    _save(tmp_path / "real", real)
    with pytest.raises(ValueError, match="imaginary"):
        io.read_series(tmp_path / "real")


def test_two_series_of_one_part_need_choosing(tmp_path):
    for number in (1, 2):
        _save(
            tmp_path / f"s{number}",
            _datasets(
                np.ones(SHAPE, np.uint16),
                echo_time=5.0,
                series_uid=pydicom.uid.generate_uid(),
                SeriesNumber=number,
            ),
        )
    with pytest.raises(ValueError, match="choose with series="):
        io.read_series(tmp_path)
    assert io.read_series(tmp_path, series=2).image.shape == (1, *SHAPE)


def _enhanced(classic):
    """One enhanced MR dataset holding the frames of classic datasets."""
    first = classic[0]
    ds = pydicom.Dataset()
    for keyword in ("SeriesInstanceUID", "Manufacturer", "MagneticFieldStrength", "Modality"):
        if keyword in first:
            setattr(ds, keyword, first.data_element(keyword).value)
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4.1"  # Enhanced MR Image Storage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.ImageType = ["ORIGINAL", "PRIMARY", "MIXED"]
    ds.Rows, ds.Columns = first.Rows, first.Columns
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.NumberOfFrames = len(classic)

    def item(**values):
        entry = pydicom.Dataset()
        for keyword, value in values.items():
            setattr(entry, keyword, value)
        return entry

    shared = item(
        PlaneOrientationSequence=[item(ImageOrientationPatient=first.ImageOrientationPatient)],
        PixelMeasuresSequence=[
            item(PixelSpacing=first.PixelSpacing, SliceThickness=first.SliceThickness)
        ],
    )
    ds.SharedFunctionalGroupsSequence = [shared]
    frames = []
    for frame in classic:
        component = {"M": "MAGNITUDE", "P": "PHASE"}[frame.ImageType[2]]
        frames.append(
            item(
                PlanePositionSequence=[item(ImagePositionPatient=frame.ImagePositionPatient)],
                MREchoSequence=[item(EffectiveEchoTime=frame.EchoTime)],
                MRImageFrameTypeSequence=[
                    item(
                        FrameType=["ORIGINAL", "PRIMARY", component[0]],
                        ComplexImageComponent=component,
                    )
                ],
                PixelValueTransformationSequence=[
                    item(
                        RescaleSlope=frame.get("RescaleSlope", "1"),
                        RescaleIntercept=frame.get("RescaleIntercept", "0"),
                        RescaleType="US",
                    )
                ],
            )
        )
    ds.PerFrameFunctionalGroupsSequence = frames
    ds.PixelData = np.stack([f.pixel_array.astype(np.uint16) for f in classic]).tobytes()
    return ds


def test_an_enhanced_dataset_reads_as_its_classic_frames(siemens, tmp_path):
    classic = [pydicom.dcmread(f) for f in sorted(siemens.iterdir())]
    tmp = tmp_path / "enhanced"
    tmp.mkdir()
    for ds in classic:
        ds.SeriesInstanceUID = classic[0].SeriesInstanceUID
    _enhanced(classic).save_as(tmp / "enhanced.dcm", enforce_file_format=True)
    read = io.read_series(tmp)
    expected = io.read_series(siemens)
    np.testing.assert_allclose(read.image.numpy(), expected.image.numpy(), atol=1e-4)
    np.testing.assert_allclose(read.affine.numpy(), expected.affine.numpy(), atol=1e-5)
    np.testing.assert_allclose(read.acquisition.echo_time.numpy(), [4.0, 9.0])
    magnitude = [ds for ds in classic if ds.ImageType[2] == "M"]
    _enhanced(magnitude).save_as(tmp / "enhanced.dcm", enforce_file_format=True)
    assert io.read_dicom(tmp).image.shape == (2, *SHAPE)


def _nifti(path, data, **sidecar):
    nibabel.save(nibabel.Nifti1Image(np.transpose(data, (2, 1, 0)), _affine().numpy()), path)
    name = path.name.removesuffix(".gz").removesuffix(".nii")
    (path.parent / f"{name}.json").write_text(json.dumps(sidecar))
    return path


def test_bids_magnitude_and_phase_files_pair_in_echo_order(tmp_path):
    files = []
    for echo in (1, 0):  # given out of order
        te = 0.004 + 0.005 * echo
        common = dict(EchoTime=te, MagneticFieldStrength=7, Manufacturer="Siemens")
        files.append(
            _nifti(
                tmp_path / f"sub-01_echo-{echo + 1}_part-mag_MEGRE.nii.gz",
                _magnitude(echo).astype(np.float32),
                **common,
            )
        )
        phase = _siemens_phase(_phase(echo)).astype(np.int16) * 2 - 4096
        files.append(
            _nifti(tmp_path / f"sub-01_echo-{echo + 1}_part-phase_MEGRE.nii.gz", phase, **common)
        )
    read = io.read_series(files)
    assert read.image.shape == (2, *SHAPE)
    np.testing.assert_allclose(read.acquisition.echo_time.numpy(), [4.0, 9.0])
    assert read.acquisition.phase_range == (-4096.0, 4096.0)
    assert read.acquisition.field_strength == 7.0
    _assert_phase(read.image, 2)


def test_nifti_parts_are_found_from_the_sidecar_image_type(tmp_path):
    magnitude = _nifti(tmp_path / "a.nii", _magnitude(0).astype(np.float32), EchoTime=0.005)
    phase = _nifti(
        tmp_path / "b.nii",
        _phase(0).astype(np.float32),
        EchoTime=0.005,
        ImageType=["ORIGINAL", "PRIMARY", "P", "ND"],
    )
    read = io.read_series([magnitude, phase])
    assert read.acquisition.phase_range == (-math.pi, math.pi)
    _assert_phase(read.image, 1)


def test_phase_over_a_whole_signed_type_takes_the_type_as_a_turn(tmp_path):
    stored = np.rint(_phase(0) / np.pi * 32768).clip(-32768, 32767).astype(np.int16)
    datasets = _datasets(
        _magnitude(0).astype(np.uint16), echo_time=5.0, series_uid=pydicom.uid.generate_uid()
    )
    datasets += _datasets(
        stored,
        echo_time=5.0,
        series_uid=pydicom.uid.generate_uid(),
        ImageType=["ORIGINAL", "PRIMARY", "P"],
        SeriesNumber=2,
    )
    _save(tmp_path / "int16", datasets)
    read = io.read_series(tmp_path / "int16")
    assert read.acquisition.phase_range == (-32768.0, 32768.0)
    _assert_phase(read.image, 1)


def test_a_dcm2niix_export_pairs_a_4d_magnitude_with_per_echo_phase(tmp_path):
    """As dcm2niix exports it: one 4D magnitude whose sidecar lists ``EchoTimes``,
    and a phase file per echo in [-4096, 4096); a part with no sidecar takes the
    other part's timings."""
    magnitude = np.stack([_magnitude(e) for e in range(2)], axis=0).astype(np.float32)
    path = tmp_path / "gre_4.nii"
    nibabel.save(
        nibabel.Nifti1Image(np.transpose(magnitude, (3, 2, 1, 0)), _affine().numpy()), path
    )
    (tmp_path / "gre_4.json").write_text(
        json.dumps({"EchoTime": 0.004, "EchoTimes": [0.004, 0.009]})
    )
    files = [path]
    for echo in range(2):
        phase = _siemens_phase(_phase(echo)).astype(np.int16) * 2 - 4096
        name = tmp_path / f"gre_5_ph_{echo + 1}.nii"
        nibabel.save(nibabel.Nifti1Image(np.transpose(phase, (2, 1, 0)), _affine().numpy()), name)
        files.append(name)
    read = io.read_series(files)
    assert read.image.shape == (2, *SHAPE)
    np.testing.assert_allclose(read.acquisition.echo_time.numpy(), [4.0, 9.0])
    assert read.acquisition.phase_range == (-4096.0, 4096.0)
    _assert_phase(read.image, 2)


def test_phase_rounded_on_export_is_warned_about(tmp_path):
    """Phase rounded to whole radians (as some NIfTI exports store it) has a handful of
    values and no field left in it."""
    magnitude = _nifti(
        tmp_path / "a_part-mag.nii", _magnitude(0).astype(np.float32), EchoTime=0.005
    )
    rounded = np.rint(_phase(0)).astype(np.int16)
    phase = _nifti(tmp_path / "a_part-phase.nii", rounded, EchoTime=0.005)
    with pytest.warns(UserWarning, match="distinct values"):
        io.read_series([magnitude, phase])


def test_phase_with_many_levels_reads_without_warning(tmp_path):
    magnitude = _nifti(
        tmp_path / "a_part-mag.nii", _magnitude(0).astype(np.float32), EchoTime=0.005
    )
    phase = _nifti(tmp_path / "a_part-phase.nii", _phase(0).astype(np.float32), EchoTime=0.005)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        io.read_series([magnitude, phase])
