"""ISMRMRD, DICOM and NIfTI files against what the standards define them to hold."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

ismrmrd = pytest.importorskip("ismrmrd")
pydicom = pytest.importorskip("pydicom")
nibabel = pytest.importorskip("nibabel")

from bartorch import io  # noqa: E402

# ISMRMRD flag bit positions.
NOISE, CALIBRATION, NAVIGATION = 19, 20, 17


def _header(
    *, x=16, y=12, z=1, slices=1, contrasts=1, fov=(256.0, 192.0, 5.0), trajectory="cartesian"
):
    xsd = ismrmrd.xsd
    limits = xsd.encodingLimitsType(
        kspace_encoding_step_1=xsd.limitType(minimum=0, maximum=y - 1, center=y // 2),
        slice=xsd.limitType(minimum=0, maximum=slices - 1, center=0),
        contrast=xsd.limitType(minimum=0, maximum=contrasts - 1, center=0),
    )
    if z > 1:
        limits.kspace_encoding_step_2 = xsd.limitType(minimum=0, maximum=z - 1, center=z // 2)
    space = xsd.encodingSpaceType(
        matrixSize=xsd.matrixSizeType(x=x, y=y, z=z),
        fieldOfView_mm=xsd.fieldOfViewMm(x=fov[0], y=fov[1], z=fov[2]),
    )
    encoding = xsd.encodingType(
        encodedSpace=space,
        reconSpace=space,
        encodingLimits=limits,
        trajectory=xsd.trajectoryType(trajectory),
    )
    return xsd.ismrmrdHeader(
        experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=127_700_000),
        encoding=[encoding],
        acquisitionSystemInformation=xsd.acquisitionSystemInformationType(
            receiverChannels=2, systemVendor="Acme"
        ),
        subjectInformation=xsd.subjectInformationType(patientName="Phantom^Water", patientID="P01"),
    )


def _acquisition(
    data, *, flags=(), trajectory=None, position=(0.0, 0.0, 0.0), dirs=None, **counters
):
    acquisition = ismrmrd.Acquisition.from_array(
        np.asarray(data, np.complex64), trajectory=trajectory
    )
    for bit in flags:
        acquisition.set_flag(bit)
    for name, value in counters.items():
        setattr(acquisition.idx, name, value)
    acquisition.position[:] = position
    read, phase, normal = dirs or ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    acquisition.read_dir[:], acquisition.phase_dir[:], acquisition.slice_dir[:] = (
        read,
        phase,
        normal,
    )
    return acquisition


def _write(path, header, acquisitions):
    dataset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    dataset.write_xml_header(ismrmrd.xsd.ToXML(header))
    for acquisition in acquisitions:
        dataset.append_acquisition(acquisition)
    dataset.close()
    return str(path)


@pytest.fixture
def cartesian(tmp_path):
    """Two slices of a 12-line grid, every other line acquired, lines 5 to 7 calibration."""
    rng = np.random.default_rng(0)
    truth = (rng.standard_normal((2, 2, 12, 16)) + 1j * rng.standard_normal((2, 2, 12, 16))).astype(
        np.complex64
    )
    acquisitions = [_acquisition(rng.standard_normal((2, 32)), flags=(NOISE,))]
    for s in range(2):
        for line in range(12):
            if line % 2 and line not in (5, 6, 7):
                continue
            flags = (CALIBRATION,) if line in (5, 6, 7) else ()
            acquisitions.append(
                _acquisition(
                    truth[s, :, line],
                    flags=flags,
                    position=(0.0, 0.0, 10.0 * s - 5.0),
                    kspace_encode_step_1=line,
                    slice=s,
                )
            )
    acquisitions.append(_acquisition(np.ones((2, 16)), flags=(NAVIGATION,), kspace_encode_step_1=0))
    path = _write(tmp_path / "cartesian.h5", _header(slices=2), acquisitions)
    return path, truth


def test_readouts_land_where_their_counters_place_them(cartesian):
    path, truth = cartesian
    raw = io.read_mrd(path)
    assert raw.axes == ("slice", "coil", "phase_encode", "readout")
    lines = [line for line in range(12) if not line % 2 or line in (5, 6, 7)]
    expected = np.zeros_like(truth)
    expected[:, :, lines] = truth[:, :, lines]
    np.testing.assert_array_equal(raw.kspace.numpy(), expected)
    assert raw.mask[:, :, 0].sum() == 2 * len(lines)
    assert raw.reference[0, :, 0].nonzero().flatten().tolist() == [5, 6, 7]


def test_noise_readouts_are_kept_apart_and_navigators_are_not_read(cartesian):
    raw = io.read_mrd(cartesian[0])
    assert raw.noise.shape == (2, 32)
    assert raw.readouts["kspace_encode_step_1"].shape == (2 * 8,)


def test_a_short_readout_is_right_aligned(tmp_path):
    acquisitions = [_acquisition(np.ones((2, 10)), kspace_encode_step_1=line) for line in range(12)]
    raw = io.read_mrd(_write(tmp_path / "echo.h5", _header(), acquisitions))
    assert raw.kspace.shape[-1] == 16
    assert raw.mask[0].tolist() == [False] * 6 + [True] * 10


def test_a_counter_past_the_header_extent_is_refused(tmp_path):
    acquisitions = [_acquisition(np.ones((2, 16)), kspace_encode_step_1=12)]
    with pytest.raises(ValueError, match="kspace_encode_step_1=12"):
        io.read_mrd(_write(tmp_path / "past.h5", _header(), acquisitions))


def test_a_trajectory_in_cycles_per_metre_becomes_grid_units(tmp_path):
    fov = (256.0, 192.0, 5.0)
    k = np.stack(
        [np.linspace(-8, 8, 16, endpoint=False) / 0.256, np.full(16, 3 / 0.192)], axis=1
    ).astype(np.float32)
    acquisitions = [
        _acquisition(np.ones((2, 16)), trajectory=k, kspace_encode_step_1=line)
        for line in range(12)
    ]
    path = _write(tmp_path / "radial.h5", _header(fov=fov, trajectory="radial"), acquisitions)
    stored = io.read_mrd(path).trajectory
    grid = io.read_mrd(path, trajectory_units="1/m").trajectory
    np.testing.assert_allclose(stored[0, :, :2].numpy(), k)
    np.testing.assert_allclose(
        grid[0, :, 0].numpy(), np.linspace(-8, 8, 16, endpoint=False), atol=1e-5
    )
    np.testing.assert_allclose(grid[0, :, 1].numpy(), 3.0, atol=1e-5)
    assert (grid[..., 2] == 0).all()


def test_the_affine_puts_the_readout_position_at_the_centre_of_the_field_of_view(tmp_path):
    # A sagittal slice: read along anterior (-y in LPS), phase along head (+z), 1 mm pixels.
    dirs = ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    centre = (12.0, -30.0, 40.0)
    acquisitions = [
        _acquisition(np.ones((2, 16)), position=centre, dirs=dirs, kspace_encode_step_1=line)
        for line in range(12)
    ]
    raw = io.read_mrd(_write(tmp_path / "sag.h5", _header(fov=(16.0, 12.0, 4.0)), acquisitions))
    affine = raw.affine.numpy()
    ras = np.array([-centre[0], -centre[1], centre[2]])
    middle = affine @ np.array([7.5, 5.5, 0.0, 1.0])
    np.testing.assert_allclose(middle[:3], ras)
    np.testing.assert_allclose(affine[:3, 0], [0.0, 1.0, 0.0])  # -y in LPS is +A in RAS
    np.testing.assert_allclose(affine[:3, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(affine[:3, 2], [-4.0, 0.0, 0.0])  # +x in LPS is -R


def test_slices_step_the_affine_by_their_spacing(cartesian):
    affine = io.read_mrd(cartesian[0]).affine.numpy()
    np.testing.assert_allclose(affine[:3, 2], [0.0, 0.0, 10.0])
    np.testing.assert_allclose((affine @ [7.5, 5.5, 1.0, 1.0])[:3], [0.0, 0.0, 5.0])


def _volume():
    z, y, x = torch.meshgrid(torch.arange(3.0), torch.arange(5.0), torch.arange(4.0), indexing="ij")
    return 100 * z + 10 * y + x


def _oblique_affine():
    angle = 0.3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([0.8, 1.2, 3.0])
    affine[:3, 3] = [-20.0, 15.0, 7.0]
    return torch.from_numpy(affine)


def test_dicom_geometry_follows_the_standard_definitions(tmp_path):
    affine = _oblique_affine()
    first = io.to_dicom(_volume(), affine)[0]
    lps = np.diag([-1.0, -1.0, 1.0])
    steps = lps @ affine[:3, :3].numpy()
    np.testing.assert_allclose(first.ImagePositionPatient, lps @ affine[:3, 3].numpy(), atol=1e-5)
    np.testing.assert_allclose(first.ImageOrientationPatient[:3], steps[:, 0] / 0.8, atol=1e-5)
    np.testing.assert_allclose(first.ImageOrientationPatient[3:], steps[:, 1] / 1.2, atol=1e-5)
    assert [float(v) for v in first.PixelSpacing] == [1.2, 0.8]  # row spacing, then column spacing
    assert (first.Rows, first.Columns) == (5, 4)


def test_a_dicom_series_reads_back_what_was_written(tmp_path):
    affine, image = _oblique_affine(), _volume()
    io.write_dicom(tmp_path / "series", image, affine, SeriesDescription="map", SeriesNumber=7)
    back = io.read_dicom(tmp_path / "series")
    assert back.image.shape == (1, 3, 5, 4)
    np.testing.assert_allclose(
        back.image[0].numpy(), image.numpy(), atol=image.max().item() / 65535
    )
    np.testing.assert_allclose(back.affine.numpy(), affine.numpy(), atol=1e-5)
    assert back.header.SeriesDescription == "map"


def test_echoes_written_as_separate_series_read_back_as_sorted_contrasts(tmp_path):
    affine = _oblique_affine()
    for number, echo in enumerate((30.0, 10.0, 20.0)):
        io.write_dicom(
            tmp_path / "echoes", _volume() + echo, affine, SeriesNumber=number + 1, EchoTime=echo
        )
    with pytest.raises(ValueError, match="3 series"):
        io.read_dicom(tmp_path / "echoes")
    back = io.read_dicom(tmp_path / "echoes", series=[1, 2, 3])
    assert back.timings["echo_time"].tolist() == [10.0, 20.0, 30.0]
    np.testing.assert_allclose(back.image[:, 0, 0, 0].numpy(), [10.0, 20.0, 30.0], atol=1e-2)
    assert back.timings["inversion_time"].isnan().all()


def test_a_complex_image_is_not_written_as_dicom():
    with pytest.raises(TypeError, match="magnitude"):
        io.to_dicom(torch.ones(4, 4, dtype=torch.complex64), torch.eye(4))


def test_patient_and_equipment_come_from_the_mrd_header(tmp_path):
    first = io.to_dicom(_volume(), _oblique_affine(), header=_header())[0]
    assert str(first.PatientName) == "Phantom^Water"
    assert first.Manufacturer == "Acme"
    assert float(first.ImagingFrequency) == pytest.approx(127.7)


def test_a_dicom_template_lends_its_study_to_a_new_series(tmp_path):
    source = io.to_dicom(_volume(), _oblique_affine(), header=_header())[0]
    derived = io.to_dicom(_volume(), _oblique_affine(), header=source)[0]
    assert derived.StudyInstanceUID == source.StudyInstanceUID
    assert derived.SeriesInstanceUID != source.SeriesInstanceUID
    assert derived.PatientID == "P01"


def test_nifti_stores_the_affine_nibabel_reads(tmp_path):
    affine, image = _oblique_affine(), _volume()
    path = io.write_nifti(tmp_path / "map.nii.gz", image, affine)
    loaded = nibabel.load(path)
    np.testing.assert_allclose(loaded.affine, affine.numpy(), atol=1e-5)
    # nibabel indexes (x, y, z); the tensor is (z, y, x).
    assert loaded.get_fdata()[3, 4, 2] == image[2, 4, 3]


def test_nifti_echoes_with_bids_sidecars_read_back_as_contrasts(tmp_path):
    affine = _oblique_affine()
    paths = []
    for echo in (10.0, 20.0):
        paths.append(
            io.write_nifti(
                tmp_path / f"echo-{echo:.0f}.nii",
                _volume() + echo,
                affine,
                timings={"echo_time": echo},
            )
        )
    assert json.loads((tmp_path / "echo-10.json").read_text()) == {"EchoTime": 0.01}
    back = io.read_nifti(paths)
    assert back.image.shape == (2, 3, 5, 4)
    np.testing.assert_allclose(back.timings["echo_time"].numpy(), [10.0, 20.0])
    np.testing.assert_allclose(back.image[1].numpy(), (_volume() + 20.0).numpy())


def test_dicom_and_nifti_agree_on_where_a_voxel_is(tmp_path):
    affine, image = _oblique_affine(), _volume()
    io.write_dicom(tmp_path / "series", image, affine)
    dicom = io.read_dicom(tmp_path / "series")
    nifti = io.read_nifti(io.write_nifti(tmp_path / "v.nii", image, affine))
    np.testing.assert_allclose(dicom.affine.numpy(), nifti.affine.numpy(), atol=1e-5)
