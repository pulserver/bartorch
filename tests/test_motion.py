"""Rigid registration, the filter over it, and the thread count it holds."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bartorch.tools import (
    NavigatorMotionTracker,
    RigidMotionEKF,
    RigidMotionEstimate,
    RigidRegistration,
)

sitk = pytest.importorskip("SimpleITK", reason="rigid registration needs the motion extra")


def plane(size: int = 64, seed: int = 0) -> np.ndarray:
    """A block on a noisy background, with enough edge to register."""
    generator = np.random.default_rng(seed)
    image = np.zeros((size, size))
    image[20:44, 24:40] = 1.0
    return image + 0.02 * generator.standard_normal((size, size))


def test_registration_recovers_a_translation() -> None:
    fixed = plane()
    moved = np.roll(fixed, (3, -2), axis=(0, 1))
    estimate = RigidRegistration()(fixed, moved, spacing=(1.0, 1.0))
    shift = np.asarray(estimate.parameters)[1:]
    assert np.linalg.norm(np.abs(shift) - np.array([2.0, 3.0])) < 1.0


def test_the_itk_thread_count_comes_back_after_registration() -> None:
    before = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    fixed = plane()
    RigidRegistration(threads=1)(fixed, np.roll(fixed, 2, axis=0), spacing=(1.0, 1.0))
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == before


def test_zero_registration_threads_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RigidRegistration(threads=0)


def test_an_estimate_cannot_be_edited() -> None:
    estimate = RigidMotionEstimate(parameters=np.zeros(6), center=np.zeros(3))
    with pytest.raises((AttributeError, TypeError)):
        estimate.parameters = np.ones(6)  # type: ignore[misc]


def test_the_filter_takes_the_registration_it_filters() -> None:
    filtered = RigidMotionEKF(RigidRegistration())
    assert filtered.velocity.shape == (6,)


def test_the_tracker_has_no_reference_until_it_tracks() -> None:
    assert NavigatorMotionTracker().reference is None


def phantom(size: int = 64) -> np.ndarray:
    """Smooth ellipses with no rotational symmetry, which pin an angle as well as a shift."""
    from scipy.ndimage import gaussian_filter

    y, x = np.mgrid[0:size, 0:size] / size
    image = np.zeros((size, size))
    image[(x - 0.45) ** 2 / 0.09 + (y - 0.5) ** 2 / 0.16 < 1] = 1.0
    image[(x - 0.35) ** 2 / 0.005 + (y - 0.35) ** 2 / 0.01 < 1] = 2.0
    image[(x - 0.6) ** 2 / 0.004 + (y - 0.65) ** 2 / 0.002 < 1] = 0.5
    image[20:26, 40:44] = 1.5
    return gaussian_filter(image, 1.0)


def rotated(image: np.ndarray, angle: float) -> np.ndarray:
    """``image``'s content turned by ``angle`` about its centre, from columns towards rows."""
    from scipy.ndimage import map_coordinates

    centre = (image.shape[0] - 1) / 2
    y, x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float) - centre
    source_x = np.cos(angle) * x + np.sin(angle) * y + centre
    source_y = -np.sin(angle) * x + np.cos(angle) * y + centre
    return map_coordinates(image, [source_y, source_x], order=3)


def converged() -> RigidRegistration:
    return RigidRegistration(iterations=200, sampling_percentage=1.0)


@pytest.mark.parametrize("metric", ["correlation", "mutual_information", "mean_squares"])
def test_each_metric_recovers_a_shift_along_the_array_axes_reversed(metric) -> None:
    fixed = phantom()
    moved = np.roll(fixed, (3, -2), axis=(0, 1))
    estimate = RigidRegistration(metric=metric, iterations=200, sampling_percentage=1.0)(
        fixed, moved
    )
    # SimpleITK's x is the last array axis.
    np.testing.assert_allclose(estimate.translation, [-2.0, 3.0], atol=0.05)
    assert abs(estimate.angles[0]) < 5e-3
    assert estimate.stop_condition


def test_a_rotation_from_the_column_axis_towards_the_row_axis_is_a_positive_angle() -> None:
    fixed = phantom()
    estimate = converged()(fixed, rotated(fixed, 0.1))
    np.testing.assert_allclose(estimate.angles, [0.1], atol=2e-3)
    np.testing.assert_allclose(estimate.translation, [0.0, 0.0], atol=0.05)


def test_a_complex_image_is_registered_by_its_modulus() -> None:
    fixed = phantom()
    phase = np.exp(1j * np.linspace(0, np.pi, fixed.size)).reshape(fixed.shape)
    moved = np.roll(fixed, (3, -2), axis=(0, 1))
    estimate = converged()(torch.as_tensor(fixed * phase), moved * phase)
    np.testing.assert_allclose(estimate.translation, [-2.0, 3.0], atol=0.05)


def test_resampling_through_a_whole_voxel_shift_undoes_it() -> None:
    fixed = phantom()
    moved = np.roll(fixed, (3, -2), axis=(0, 1))
    estimate = RigidMotionEstimate(parameters=[0.0, -2.0, 3.0], center=[31.5, 31.5])

    back = RigidRegistration().resample(torch.as_tensor(moved), estimate)

    assert back.dtype == torch.float32
    np.testing.assert_allclose(back.numpy()[4:-4, 4:-4], fixed[4:-4, 4:-4], atol=1e-6)


def test_resampling_onto_a_reference_takes_its_grid_and_fills_outside_the_image() -> None:
    moving = np.ones((16, 16))
    estimate = RigidMotionEstimate(parameters=[0.0, 0.0, 0.0], center=[0.0, 0.0])
    result = RigidRegistration().resample(
        moving, estimate, reference=np.zeros((20, 24)), default_value=-1.0
    )
    assert tuple(result.shape) == (20, 24)
    np.testing.assert_array_equal(result.numpy()[:16, :16], 1.0)
    np.testing.assert_array_equal(result.numpy()[17:, :], -1.0)


def test_the_matrix_of_a_planar_estimate_turns_about_its_centre() -> None:
    angle, centre, shift = 0.3, np.array([2.0, -1.0]), np.array([0.5, 4.0])
    estimate = RigidMotionEstimate(parameters=[angle, *shift], center=centre)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    point = np.array([7.0, 3.0])

    mapped = estimate.matrix @ np.append(point, 1.0)

    np.testing.assert_allclose(mapped[:2], rotation @ (point - centre) + centre + shift)
    np.testing.assert_allclose(mapped[2], 1.0)


def test_the_matrix_of_a_spatial_estimate_composes_z_then_x_then_y() -> None:
    ax, ay, az = 0.1, -0.2, 0.3
    estimate = RigidMotionEstimate(parameters=[ax, ay, az, 1.0, 2.0, 3.0], center=np.zeros(3))
    c, s = np.cos, np.sin
    rx = np.array([[1, 0, 0], [0, c(ax), -s(ax)], [0, s(ax), c(ax)]])
    ry = np.array([[c(ay), 0, s(ay)], [0, 1, 0], [-s(ay), 0, c(ay)]])
    rz = np.array([[c(az), -s(az), 0], [s(az), c(az), 0], [0, 0, 1]])
    np.testing.assert_allclose(estimate.matrix[:3, :3], rz @ rx @ ry, atol=1e-12)
    np.testing.assert_allclose(estimate.matrix[:3, 3], [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metric": "cosine"}, "unsupported registration metric"),
        ({"iterations": 0}, "iterations must be positive"),
        ({"sampling_percentage": 0.0}, "sampling_percentage"),
        ({"shrink_factors": (2, 1), "smoothing_sigmas": (1.0,)}, "equal length"),
        ({"shrink_factors": (0,), "smoothing_sigmas": (0.0,)}, "invalid multi-resolution"),
        ({"learning_rate": 0.0}, "optimizer steps must be positive"),
    ],
)
def test_an_impossible_registration_is_refused(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        RigidRegistration(**kwargs)


@pytest.mark.parametrize(
    ("fixed", "moving", "kwargs", "message"),
    [
        (np.zeros((8, 8)), np.zeros((8, 8, 8)), {}, "equal dimensionality"),
        (np.zeros(8), np.zeros(8), {}, "one 2D or 3D image"),
        (np.full((8, 8), np.nan), np.zeros((8, 8)), {}, "finite"),
        (np.zeros((8, 8)), np.zeros((8, 8)), {"spacing": (1.0,)}, "one positive value"),
        (
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            {"initial": RigidMotionEstimate(parameters=np.zeros(6), center=np.zeros(3))},
            "initial transform dimensionality",
        ),
    ],
)
def test_images_a_registration_cannot_take_are_refused(fixed, moving, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        RigidRegistration()(fixed, moving, **kwargs)


def test_resampling_through_a_transform_of_the_other_dimensionality_is_refused() -> None:
    estimate = RigidMotionEstimate(parameters=np.zeros(6), center=np.zeros(3))
    with pytest.raises(ValueError, match="does not match the moving image"):
        RigidRegistration().resample(np.zeros((8, 8)), estimate)


#: Axial, coronal and sagittal planes: (row, column) directions in (x, y, z).
ORTHOGONAL = (
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
)


def test_the_first_navigator_is_the_reference_and_its_pose_is_zero() -> None:
    tracker = NavigatorMotionTracker()
    planes = [phantom(32)] * 3
    pose = tracker.track(planes, ORTHOGONAL)
    assert tracker.reference is not None and len(tracker.reference) == 3
    np.testing.assert_array_equal(pose.parameters, np.zeros(6))
    assert tracker.registration is tracker.filter.registration
    np.testing.assert_array_equal(tracker.pose.parameters, np.zeros(6))


def test_a_translation_seen_in_three_planes_is_the_translation_of_the_pose() -> None:
    tracker = NavigatorMotionTracker(registration=converged(), measurement_noise=1e-8)
    reference = phantom()
    tracker.track([reference] * 3, ORTHOGONAL)
    shift = np.array([-2.0, 3.0, 1.0])
    # A plane sees the shift along its rows and along its columns.
    planes = [
        np.roll(reference, (int(shift @ row), int(shift @ column)), axis=(0, 1))
        for row, column in np.asarray(ORTHOGONAL)
    ]

    pose = tracker.track(planes, ORTHOGONAL, spacing=1.0)

    np.testing.assert_allclose(pose.translation, shift, atol=0.05)
    np.testing.assert_allclose(pose.angles, 0.0, atol=5e-3)


def test_a_turn_seen_only_in_the_axial_plane_is_a_rotation_about_z() -> None:
    tracker = NavigatorMotionTracker(registration=converged(), measurement_noise=1e-8)
    reference = phantom()
    tracker.track([reference] * 3, ORTHOGONAL)
    # The content turns from x towards y, which is a positive turn about z.
    pose = tracker.track([rotated(reference, 0.1), reference, reference], ORTHOGONAL)
    np.testing.assert_allclose(pose.angles, [0.0, 0.0, 0.1], atol=3e-3)


def test_reset_discards_the_reference_navigator() -> None:
    tracker = NavigatorMotionTracker()
    tracker.track([phantom(32)] * 3, ORTHOGONAL)
    tracker.reset()
    assert tracker.reference is None
    assert not tracker.filter.initialized


@pytest.mark.parametrize(
    ("axes", "message"),
    [
        (ORTHOGONAL[:2], "a row and a column direction"),
        ((((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),) + ORTHOGONAL[1:], "non-zero"),
        ((((1.0, 1.0, 0.0), (1.0, 0.0, 0.0)),) + ORTHOGONAL[1:], "perpendicular"),
    ],
)
def test_plane_axes_that_are_not_a_perpendicular_pair_per_plane_are_refused(axes, message):
    with pytest.raises(ValueError, match=message):
        NavigatorMotionTracker().track([phantom(32)] * 3, axes)


def test_a_navigator_with_another_number_of_planes_than_the_reference_is_refused() -> None:
    tracker = NavigatorMotionTracker()
    tracker.track([phantom(32)] * 3, ORTHOGONAL)
    with pytest.raises(ValueError, match="2 planes and the reference has 3"):
        tracker.track([phantom(32)] * 2, ORTHOGONAL[:2])


def test_planes_that_leave_a_rotation_unmeasured_are_refused() -> None:
    tracker = NavigatorMotionTracker(registration=RigidRegistration(iterations=1))
    tracker.track([phantom(32)] * 2, ORTHOGONAL[:2])
    with pytest.raises(ValueError, match="do not span the rotation"):
        tracker.track([phantom(32)] * 2, ORTHOGONAL[:2])
