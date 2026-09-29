"""The constant-velocity Kalman filter over rigid poses, held to the textbook filter.

None of this needs SimpleITK: the filter takes measured poses, and only
:meth:`RigidMotionEKF.step` asks a registration for one, which a stand-in
answers here.
"""

from __future__ import annotations

import numpy as np
import pytest

from bartorch.tools import RigidMotionEKF, RigidMotionEstimate


class RecordingRegistration:
    """A registration that returns a fixed pose and records the start it was given."""

    def __init__(self, parameters) -> None:
        self.parameters = np.asarray(parameters, dtype=np.float64)
        self.calls: list[dict] = []

    def estimate(self, fixed, moving, *, initial=None, spacing=None) -> RigidMotionEstimate:
        self.calls.append({"initial": initial, "spacing": spacing})
        return RigidMotionEstimate(parameters=self.parameters, center=np.zeros(3))


def _filter(**kwargs) -> RigidMotionEKF:
    return RigidMotionEKF(RecordingRegistration(np.zeros(6)), **kwargs)


def _constant_velocity_blocks(dt: float, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transition and process noise of a piecewise-constant acceleration, written out.

    Per coordinate, the acceleration enters through ``G = (dt^2 / 2, dt)``, so
    ``Q = G q G^T``; the state is every pose coordinate followed by every rate.
    """
    dof = q.size
    transition = np.eye(2 * dof)
    noise = np.zeros((2 * dof, 2 * dof))
    for i in range(dof):
        transition[i, dof + i] = dt
        gain = np.array([dt**2 / 2.0, dt])
        block = q[i] * np.outer(gain, gain)
        for a, row in enumerate((i, dof + i)):
            for b, column in enumerate((i, dof + i)):
                noise[row, column] = block[a, b]
    return transition, noise


def test_predict_moves_the_pose_by_its_velocity_and_grows_the_covariance_in_closed_form():
    q = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3])
    kalman = _filter(process_noise=q)
    rng = np.random.default_rng(0)
    pose = np.array([0.1, -0.2, 0.05, 1.0, -2.0, 3.0])
    velocity = np.array([0.01, 0.02, -0.03, 0.5, -0.25, 0.125])
    kalman.update(RigidMotionEstimate(parameters=pose, center=np.zeros(3)))
    kalman.state[6:] = velocity
    spread = rng.standard_normal((12, 12))
    kalman.covariance = spread @ spread.T + np.eye(12)
    before = kalman.covariance.copy()

    dt = 0.7
    predicted = kalman.predict(dt)

    transition, noise = _constant_velocity_blocks(dt, q)
    np.testing.assert_allclose(predicted.parameters, pose + dt * velocity, atol=1e-15)
    np.testing.assert_allclose(kalman.velocity, velocity, atol=0)
    np.testing.assert_allclose(kalman.covariance, transition @ before @ transition.T + noise)
    np.testing.assert_allclose(predicted.covariance, kalman.covariance[:6, :6])


def test_the_joseph_update_is_the_textbook_kalman_update():
    kalman = _filter(measurement_noise=[1e-2, 2e-2, 3e-2, 0.1, 0.2, 0.3])
    kalman.update(RigidMotionEstimate(parameters=np.zeros(6), center=np.zeros(3)))
    rng = np.random.default_rng(1)
    kalman.state = rng.normal(scale=0.1, size=12)
    spread = rng.standard_normal((12, 12))
    kalman.covariance = 0.1 * spread @ spread.T + 0.5 * np.eye(12)
    state, covariance = kalman.state.copy(), kalman.covariance.copy()
    measured = rng.normal(scale=0.1, size=6)
    noise = np.diag([1e-2, 2e-2, 3e-2, 0.1, 0.2, 0.3])

    kalman.update(measured)

    observation = np.hstack([np.eye(6), np.zeros((6, 6))])
    gain = (
        covariance @ observation.T @ np.linalg.inv(observation @ covariance @ observation.T + noise)
    )
    np.testing.assert_allclose(kalman.state, state + gain @ (measured - observation @ state))
    np.testing.assert_allclose(
        kalman.covariance, (np.eye(12) - gain @ observation) @ covariance, atol=1e-12
    )


def test_a_measurement_covariance_takes_precedence_over_the_configured_noise():
    stated = np.diag([4.0, 4.0, 4.0])
    loose = RigidMotionEKF(None, dimension=2, measurement_noise=1e-6)
    tight = RigidMotionEKF(None, dimension=2, measurement_noise=1e-6)
    for kalman in (loose, tight):
        kalman.update(np.zeros(3))
    loose.update(
        RigidMotionEstimate(parameters=[0.0, 1.0, 1.0], center=np.zeros(2), covariance=stated)
    )
    tight.update(np.array([0.0, 1.0, 1.0]))

    # One unit of prior variance against four of measurement: a fifth of the way.
    np.testing.assert_allclose(loose.pose.translation, [0.2, 0.2])
    np.testing.assert_allclose(tight.pose.translation, [1.0, 1.0], atol=1e-5)


def test_a_constant_velocity_sequence_is_tracked():
    kalman = _filter(process_noise=1e-8, measurement_noise=1e-6)
    velocity = np.array([0.002, -0.001, 0.003, 0.4, -0.2, 0.1])
    start = np.array([0.01, 0.02, -0.01, 1.0, 2.0, -3.0])
    for step in range(12):
        if kalman.initialized:
            kalman.predict(1.0)
        filtered = kalman.update(start + step * velocity)

    np.testing.assert_allclose(filtered.parameters, start + 11 * velocity, atol=1e-5)
    np.testing.assert_allclose(kalman.velocity, velocity, atol=1e-5)


def test_the_first_measurement_initializes_the_pose_at_rest():
    kalman = _filter(initial_covariance=2.0)
    stated = 0.5 * np.eye(6)
    measured = RigidMotionEstimate(
        parameters=np.arange(6.0) / 10, center=[1.0, 2.0, 3.0], covariance=stated, metric=-0.9
    )

    pose = kalman.update(measured)

    assert kalman.initialized
    np.testing.assert_array_equal(pose.parameters, measured.parameters)
    np.testing.assert_array_equal(pose.center, measured.center)
    np.testing.assert_array_equal(kalman.velocity, np.zeros(6))
    np.testing.assert_array_equal(kalman.covariance[:6, :6], stated)
    np.testing.assert_array_equal(kalman.covariance[6:, 6:], 2.0 * np.eye(6))
    assert pose.metric == -0.9


def test_bare_parameters_are_measured_about_the_filters_centre():
    kalman = _filter()
    kalman.update(RigidMotionEstimate(parameters=np.zeros(6), center=[5.0, 6.0, 7.0]))
    kalman.predict()
    pose = kalman.update(np.zeros(6))
    np.testing.assert_array_equal(pose.center, [5.0, 6.0, 7.0])


def test_reset_forgets_the_pose_or_starts_from_the_one_given():
    kalman = _filter(initial_covariance=3.0)
    kalman.update(np.ones(6))
    kalman.predict()
    kalman.update(2 * np.ones(6))

    kalman.reset()
    assert not kalman.initialized
    assert kalman.last_measurement is None
    np.testing.assert_array_equal(kalman.state, np.zeros(12))
    np.testing.assert_array_equal(kalman.covariance, 3.0 * np.eye(12))
    assert kalman.pose.metric is None

    start = RigidMotionEstimate(parameters=np.full(6, 0.25), center=[1.0, 0.0, 0.0])
    kalman.reset(start)
    assert kalman.initialized
    np.testing.assert_array_equal(kalman.pose.parameters, start.parameters)
    np.testing.assert_array_equal(kalman.pose.center, start.center)


def test_an_angle_crossing_pi_is_predicted_and_filtered_on_the_circle():
    kalman = RigidMotionEKF(None, dimension=2, process_noise=1e-8, measurement_noise=1.0)
    kalman.update(np.array([np.pi - 0.05, 0.0, 0.0]))
    kalman.state[3] = 0.1

    predicted = kalman.predict(1.0)
    np.testing.assert_allclose(predicted.angles, [-np.pi + 0.05], atol=1e-12)

    # A measurement 0.07 short of the prediction, on the other side of the
    # cut: the innovation is -0.07, not 2 pi - 0.07.  One unit of variance on
    # the pose and one on its rate make the predicted variance two, against one
    # of measurement, so the gain is two thirds.
    filtered = kalman.update(np.array([np.pi - 0.02, 0.0, 0.0]))
    np.testing.assert_allclose(filtered.angles, [-np.pi + 0.05 - 2 / 3 * 0.07], atol=1e-6)
    assert -np.pi <= filtered.angles[0] < np.pi


def test_step_registers_from_the_prediction_and_fuses_what_it_measured():
    registration = RecordingRegistration([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    kalman = RigidMotionEKF(registration, process_noise=1e-8, measurement_noise=1e-8)

    first = kalman.step("fixed", "moving", spacing=(1.0, 1.0, 1.0))
    assert registration.calls[0]["initial"] is None
    assert registration.calls[0]["spacing"] == (1.0, 1.0, 1.0)
    np.testing.assert_array_equal(first.parameters, registration.parameters)

    kalman.state[6:] = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    kalman.step("fixed", "moving", dt=2.0)
    np.testing.assert_allclose(
        registration.calls[1]["initial"].parameters, [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dimension": 4}, "two or three"),
        ({"initial_covariance": 0.0}, "initial_covariance"),
        ({"process_noise": [1.0, 2.0]}, "process_noise"),
        ({"measurement_noise": -1.0}, "measurement_noise"),
    ],
)
def test_a_filter_with_an_impossible_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _filter(**kwargs)


def test_a_non_positive_time_step_is_refused():
    with pytest.raises(ValueError, match="dt must be positive"):
        _filter().predict(0.0)


def test_a_measurement_of_the_other_dimensionality_is_refused():
    with pytest.raises(ValueError, match="dimensionality"):
        _filter().update(RigidMotionEstimate(parameters=np.zeros(3), center=np.zeros(2)))


def test_a_measurement_covariance_of_the_wrong_shape_is_refused():
    kalman = _filter()
    kalman.update(np.zeros(6))
    with pytest.raises(ValueError, match="wrong shape"):
        kalman.update(np.zeros(6), covariance=np.eye(3))


@pytest.mark.parametrize(
    ("parameters", "center", "covariance", "message"),
    [
        (np.zeros(4), np.zeros(2), None, "three 2D or six 3D"),
        (np.zeros(3), np.zeros(3), None, "center dimensionality"),
        (np.zeros(6), np.zeros(3), np.eye(3), "covariance shape"),
    ],
)
def test_an_inconsistent_estimate_is_refused(parameters, center, covariance, message):
    with pytest.raises(ValueError, match=message):
        RigidMotionEstimate(parameters=parameters, center=center, covariance=covariance)


def test_an_estimate_splits_its_parameters_into_angles_and_translation():
    planar = RigidMotionEstimate(parameters=[0.1, 2.0, 3.0], center=np.zeros(2))
    spatial = RigidMotionEstimate(parameters=[0.1, 0.2, 0.3, 4.0, 5.0, 6.0], center=np.zeros(3))
    assert (planar.dimension, spatial.dimension) == (2, 3)
    np.testing.assert_array_equal(planar.angles, [0.1])
    np.testing.assert_array_equal(planar.translation, [2.0, 3.0])
    np.testing.assert_array_equal(spatial.angles, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(spatial.translation, [4.0, 5.0, 6.0])
