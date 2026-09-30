"""Rigid motion estimation from navigators: plane reconstruction, registration and filtering.

Registration is SimpleITK's, an optional dependency (``pip install 'bartorch[motion]'``)
imported when a registration runs.
"""

from __future__ import annotations

from bartorch.tools._motion._navigator import reconstruct_navigator
from bartorch.tools._motion._registration import (
    NavigatorMotionTracker,
    RigidMotionEKF,
    RigidMotionEstimate,
    RigidRegistration,
)

__all__ = [
    "NavigatorMotionTracker",
    "RigidMotionEKF",
    "RigidMotionEstimate",
    "RigidRegistration",
    "reconstruct_navigator",
]
