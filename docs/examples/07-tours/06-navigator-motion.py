"""
=================================
Rigid head motion from navigators
=================================

In this tour you measure how a head moves during a scan, from navigators. If
the head moves, the anatomy is in a different place for the readouts that
encode it, and the image is blurred and ghosted. Prospective correction
moves the imaging field of view with the head before each readout, and for
that it needs the rigid pose of the head, six degrees of freedom, measured
again and again during the scan. A navigator is a short, low-resolution
acquisition interleaved with the imaging readouts; registering it against
the first navigator of the scan tells you how the head has moved since
[#ehman]_.

A 2D navigator plane, registered against its reference, measures three
degrees of freedom: two shifts in the plane and one rotation about its
normal. Three orthogonal planes give nine measurements of the six degrees of
freedom of the pose, which a least-squares fit combines. Over a scan, a
Kalman filter smooths the poses, as in PROMO [#promo]_, which tracks the
head with three orthogonal spiral navigators.

Prerequisites: lesson 7,
:doc:`../04-non-cartesian/01-trajectories-and-transforms`.

**Learning objectives**

- Relate each navigator plane to the three degrees of freedom it measures,
  and the rigid pose to the nine measurements of three orthogonal planes.
- Reconstruct radial navigator planes with
  :func:`~bartorch.tools.reconstruct_navigator` and measure the pose with
  :class:`~bartorch.tools.NavigatorMotionTracker`.
- Check a measured pose by comparing the navigator of the moved head with
  the navigator of the head moved by that pose.
- Set the filter's process and measurement noise against the precision of
  the navigator and the dynamics of the motion.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
import numpy as np
from brainweb_dl import get_mri
from scipy import ndimage


def show(axis, values, title, vmin=0.0, vmax=1.0, cmap="gray"):
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axis.set_title(title)
    axis.set_axis_off()
    return handle


def nrmse(estimate, target):
    return float((estimate - target).norm() / target.norm())


def as_vector(pose):
    """A pose as (rotation vector [rad], translation [mm])."""
    rotation = Rotation.from_matrix(np.asarray(pose.matrix)[:3, :3])
    return np.concatenate([rotation.as_rotvec(), np.asarray(pose.translation)])


# The head: BrainWeb's T1-weighted volume at 2 mm, padded to 112^3.
volume = ndimage.zoom(get_mri(sub_id=0, contrast="T1").astype(np.float32), 0.5, order=1)
SIZE = 112
volume = np.pad(volume, [((SIZE - n) // 2, SIZE - n - (SIZE - n) // 2) for n in volume.shape])
volume /= volume.max()
centre = (np.array(volume.shape) - 1) / 2
VOXEL_MM = 2.0


def move(rotvec, translation_mm):
    """The volume after a rigid motion about its centre."""
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    shift = np.asarray(translation_mm) / VOXEL_MM
    return ndimage.affine_transform(
        volume, rotation.T, offset=centre - rotation.T @ (centre + shift), order=1
    )


# sphinx_gallery_end_ignore

# %%
# The head
# --------
#
# Your head is the BrainWeb T1-weighted volume [#brainweb]_, downsampled to
# 2 mm isotropic and padded to :math:`112^3`. Array axis 0 runs inferior to
# superior (S/I), axis 1 posterior to anterior (A/P) and axis 2 left to
# right (L/R). You state poses in that frame: a rotation vector in radians
# about the centre of the volume, printed in degrees, and a translation in
# millimetres. ``move(rotvec, translation_mm)``, hidden on this page, returns
# the volume after such a motion.
#
# The navigator
# -------------
#
# Each navigator is three central planes, axial, coronal and sagittal, each
# acquired along 96 golden-angle radial spokes of 56 samples. The trajectory
# is in grid units of a :math:`56^2` matrix, so the spokes reach half the
# imaging resolution and you reconstruct each plane at 4 mm over the full
# field of view. :func:`~bartorch.tools.reconstruct_navigator` grids a plane
# by the density-compensated adjoint NUFFT, with weights from
# :func:`bartorch.estimate_density`.
import numpy as np
import torch

import bartorch
import bartorch.tools as bt

from scipy.spatial.transform import Rotation

NAV, SPOKES = 56, 96
golden = np.pi * (3 - np.sqrt(5))
angle = np.arange(SPOKES)[:, None] * golden
radius = np.arange(NAV) - NAV / 2
spokes = np.stack([np.cos(angle) * radius, np.sin(angle) * radius], axis=-1)
trajectory = torch.tensor(spokes.reshape(-1, 2), dtype=torch.float32)
density = bartorch.estimate_density(trajectory, (NAV, NAV))

# %%
# ``AXES`` tells the tracker, for each plane, which direction in the volume
# its image's rows and columns run along. That is what relates each plane's
# in-plane measurement to the 3D pose.
AXES = [
    ((0, 1, 0), (0, 0, 1)),  # axial: rows A/P, columns L/R
    ((1, 0, 0), (0, 0, 1)),  # coronal: rows S/I, columns L/R
    ((1, 0, 0), (0, 1, 0)),  # sagittal: rows S/I, columns A/P
]
SPACING_MM = VOXEL_MM * SIZE / NAV


def navigator(head, noise=0.0, generator=None):
    """The three navigator planes of ``head``, with k-space noise."""
    middle = SIZE // 2
    traj = torch.nn.functional.pad(trajectory, (0, 1))[None]
    planes = []
    for plane in (head[middle], head[:, middle], head[:, :, middle]):
        plane = torch.tensor(plane, dtype=torch.complex64)
        samples = bartorch.nufft(plane, traj).reshape(1, -1)
        if noise:
            samples = samples + noise * torch.randn(
                samples.shape,
                dtype=torch.complex64,
                generator=generator,
            )
        nav = bt.reconstruct_navigator(
            samples,
            trajectory[None],
            (NAV, NAV),
            density=density,
        )
        planes.append(nav[0])
    return planes


reference = navigator(volume)

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
for axis, plane, title in zip(axes, reference, ("axial", "coronal", "sagittal"), strict=True):
    axis.imshow(plane, cmap="gray", origin="lower")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
# These are the three reference planes at 4 mm resolution. Rigid
# registration relies on the outline of the head and the large structures,
# which this resolution retains.
#
# One pose
# --------
#
# :class:`~bartorch.tools.NavigatorMotionTracker` takes the first navigator
# you give it as the reference and returns the pose of every later one
# relative to it. It registers each plane in 2D with
# :class:`~bartorch.tools.RigidRegistration`, and solves for the pose that
# fits the three planes in the least-squares sense. You move the head by a
# few degrees and millimetres along every axis and see what the tracker
# measures.
rotvec = np.array([0.02, -0.03, 0.05])  # rad
translation = np.array([3.0, -2.0, 4.0])  # mm

tracker = bt.NavigatorMotionTracker(measurement_noise=1e-3)
tracker.track(reference, AXES, spacing=SPACING_MM)
moved = navigator(move(rotvec, translation))
pose = tracker.track(moved, AXES, spacing=SPACING_MM)

rotation = Rotation.from_matrix(np.asarray(pose.matrix)[:3, :3])
measured_rotvec = rotation.as_rotvec()
measured_translation = np.asarray(pose.translation)

# sphinx_gallery_start_ignore
NAMES = ("S/I", "A/P", "L/R")
print(f"{'':>14} {'truth':>7} {'measured':>9}")
for axis, name in enumerate(NAMES):
    print(
        f"rotation {name:>5} {np.degrees(rotvec[axis]):6.2f}° "
        f"{np.degrees(measured_rotvec[axis]):8.2f}°"
    )
for axis, name in enumerate(NAMES):
    print(f"shift    {name:>5} {translation[axis]:5.1f} mm {measured_translation[axis]:6.2f} mm")
# sphinx_gallery_end_ignore

# %%
# In a real scan the true pose is unknown, so you check a measured pose on
# the navigator itself: move the head by the measured pose, acquire
# its navigator, and compare that with the navigator after the real motion.
# Without any correction, the difference is that of the motion; with the
# measured pose, what is left is the error of the pose.
realigned = navigator(move(measured_rotvec, measured_translation))

# sphinx_gallery_start_ignore
for plane, before, after, target in zip(
    ("axial", "coronal", "sagittal"), reference, realigned, moved, strict=True
):
    print(
        f"{plane:>8}: NRMSE against the moved navigator, "
        f"reference {nrmse(before, target):.3f}, "
        f"measured pose {nrmse(after, target):.3f}"
    )

figure, axes = plt.subplots(2, 3, figsize=(7.8, 5.0))
for column, (plane, before, after, target) in enumerate(
    zip(("axial", "coronal", "sagittal"), reference, realigned, moved, strict=True)
):
    scale = float(target.abs().max())
    show(axes[0, column], (target - before) / scale, plane, -0.5, 0.5, "RdBu_r")
    handle = show(axes[1, column], (target - after) / scale, "", -0.5, 0.5, "RdBu_r")
for row, label in enumerate(("moved - reference", "moved - measured")):
    axes[row, 0].text(
        -0.06, 0.5, label, rotation=90, ha="right", va="center", transform=axes[row, 0].transAxes
    )
figure.colorbar(handle, ax=axes, shrink=0.8, label="difference / peak")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The top row is the motion as the navigator sees it: strong edges all
# around the head. In the bottom row, after the measured pose, only a faint
# difference at the edges of the head is left. The pose is off by a fraction
# of a degree and of a millimetre, a small fraction of the 4 mm navigator
# voxel. That error has two sources: the coarse resolution of the
# navigator, and the through-plane motion, which each plane sees as a change
# of the anatomy in it rather than as a rigid motion within it.
#
# A scan
# ------
#
# Over a scan, the head nods, a rotation about the
# left-right axis, by up to 2.3° with a period of 8 s, and drifts by 3 mm
# along the superior-inferior axis over 12 s. You acquire a navigator every
# 0.5 s, and each one carries complex Gaussian k-space noise, so each
# measured pose carries registration error.
DT, COUNT = 0.5, 24
seconds = DT * np.arange(1, COUNT + 1)
nod = 0.04 * np.sin(2 * np.pi * seconds / 8)  # rad, about L/R
drift = 3.0 * seconds / seconds[-1]  # mm, along S/I
truth = np.zeros((COUNT, 6))
truth[:, 2], truth[:, 3] = nod, drift

generator = torch.Generator().manual_seed(0)
scan = []
for a, d in zip(nod, drift):
    head = move((0.0, 0.0, a), (d, 0.0, 0.0))
    scan.append(navigator(head, noise=0.5, generator=generator))

# %%
# The tracker filters the poses with a constant-velocity extended Kalman
# filter, and two numbers set it. ``process_noise`` is the variance of the
# acceleration the filter allows between navigators, per pose coordinate, in
# rad²/s⁴ and mm²/s⁴. ``measurement_noise`` is the variance it assigns to
# each measured coordinate. You filter the same scan with three values of
# ``process_noise``; the largest one leaves the filter at the measurements.


def track(process_noise):
    tracker = bt.NavigatorMotionTracker(
        process_noise=process_noise,
        measurement_noise=1e-4,
    )
    tracker.track(reference, AXES, spacing=SPACING_MM)
    poses = []
    for planes in scan:
        poses.append(tracker.track(planes, AXES, dt=DT, spacing=SPACING_MM))
    return poses


traces = {noise: track(noise) for noise in (1e2, 1e-3, 1e-4)}

# sphinx_gallery_start_ignore
traces = {noise: np.array([as_vector(p) for p in poses]) for noise, poses in traces.items()}
rms = {noise: np.sqrt(np.mean((trace - truth) ** 2, axis=0)) for noise, trace in traces.items()}
for noise, error in rms.items():
    print(
        f"process_noise {noise:<6g} rms error: rotation "
        + ", ".join(f"{n} {np.degrees(e):.2f}°" for n, e in zip(NAMES, error[:3], strict=True))
        + "; shift "
        + ", ".join(f"{n} {e:.2f} mm" for n, e in zip(NAMES, error[3:], strict=True))
    )

LABELS = {1e2: "process_noise 1e2", 1e-3: "1e-3", 1e-4: "1e-4"}
figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.8))
for axis, index, scale, title, label in (
    (axes[0], 2, np.degrees(1.0), "nod, about L/R", "rotation [°]"),
    (axes[1], 3, 1.0, "drift, along S/I", "shift [mm]"),
):
    axis.plot(seconds, scale * truth[:, index], color="0.55", linewidth=3.0, label="head")
    for noise, style, colour in ((1e2, ".", "C0"), (1e-3, "-", "C1"), (1e-4, "-", "C2")):
        axis.plot(
            seconds, scale * traces[noise][:, index], style, color=colour, label=LABELS[noise]
        )
    axis.set_xlabel("time [s]")
    axis.set_ylabel(label)
    axis.set_title(title)
figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
plt.show()
# sphinx_gallery_end_ignore

# %%
# A large ``process_noise`` leaves the filter at the measurements (the
# dots). A small one makes it trust its constant-velocity prediction, which
# smooths the registration error but lags a change of direction: at ``1e-4``
# the error of the nod grows while that of the other coordinates falls. An
# error common to every navigator, such as the offset of the A/P shift, is a
# bias of the measurement, and no value of ``process_noise`` reduces it. The
# value that gives the smallest error depends on the motion and on the
# navigator's precision, so you set it against a motion trace of your own
# application.
#
# As a spec
# ---------
#
# What this tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Track rigid head motion from navigators with bartorch. Each navigator
#    is three orthogonal central planes (axial, coronal, sagittal), each 96
#    golden-angle radial spokes of 56 samples, reconstructed on a 56^2 grid
#    with bt.reconstruct_navigator and bartorch.estimate_density weights.
#    Pass the plane orientations as row and column directions in the volume
#    (S/I, A/P, L/R) and the pixel spacing in mm to
#    bt.NavigatorMotionTracker; the first navigator is the reference.
#    Report each pose as a rotation vector and a translation in mm, check it
#    by re-navigating the head moved by the measured pose, and filter a
#    24-navigator scan (one every 0.5 s) with process_noise 1e2, 1e-3 and
#    1e-4, reporting the RMS error per coordinate against the true motion.

# %%
#
# References
# ----------
#
# .. [#ehman] Ehman RL, Felmlee JP. Adaptive technique for high-definition MR
#    imaging of moving structures. *Radiology* 173(1):255-263 (1989).
#    https://doi.org/10.1148/radiology.173.1.2781017
#
# .. [#promo] White N, Roddey C, Shankaranarayanan A, Han E, Rettmann D,
#    Santos J, Kuperman J, Dale A. PROMO: Real-time prospective motion
#    correction in MRI using image-based tracking. *Magn Reson Med*
#    63(1):91-105 (2010). https://doi.org/10.1002/mrm.22176
#
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ,
#    Holmes CJ, Evans AC. Design and construction of a realistic digital
#    brain phantom. *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
