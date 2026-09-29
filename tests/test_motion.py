"""Rigid registration, the filter over it, and the thread count it holds."""

from __future__ import annotations

import numpy as np
import pytest

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
