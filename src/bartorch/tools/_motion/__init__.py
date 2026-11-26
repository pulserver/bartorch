"""Rigid motion: from navigators, and between the shots of a segmented acquisition.

Registration is SimpleITK's, imported when a registration runs.
"""

from __future__ import annotations

from bartorch.tools._motion._navigator import reconstruct_navigator
from bartorch.tools._motion._registration import (
    NavigatorMotionTracker,
    RigidMotionEKF,
    RigidMotionEstimate,
    RigidRegistration,
)
from bartorch.tools._motion._shots import estimate_rigid_motion, rigid_encoding

__all__ = [
    "NavigatorMotionTracker",
    "RigidMotionEKF",
    "RigidMotionEstimate",
    "RigidRegistration",
    "estimate_rigid_motion",
    "reconstruct_navigator",
    "rigid_encoding",
]
