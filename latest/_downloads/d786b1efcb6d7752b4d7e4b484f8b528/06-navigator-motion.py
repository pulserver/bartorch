"""
=================================
Rigid head motion from navigators
=================================

Head motion during a scan changes the position of the anatomy between the
readouts that encode it, and the image acquires blurring and ghosting.
Prospective correction updates the imaging field of view with the measured
head pose before each readout; it requires a measurement of the rigid pose
with six degrees of freedom, repeated during the scan. A navigator is a
short, low-resolution acquisition interleaved with the imaging readouts, and
its registration against the first navigator of the scan measures how the
head has moved since [#ehman]_.

This example measures the rigid pose of a BrainWeb head from three
orthogonal radial navigator planes, compares the navigator after motion and
after the measured pose is applied with the reference navigator, and tracks a
nodding and drifting head over a 12 s scan with a constant-velocity extended
Kalman filter, as in PROMO [#promo]_, which tracks the head with three
orthogonal spiral navigators.

**Learning objectives**

* Relate each 2D navigator plane to the three degrees of freedom it measures,
  and the rigid pose to the nine measurements of three orthogonal planes.
* Reconstruct radial navigator planes with
  :func:`~bartorch.tools.reconstruct_navigator` and measure the pose with
  :class:`~bartorch.tools.NavigatorMotionTracker`.
* Assess a measured pose by the residual between the navigator and the
  navigator of the head moved by that pose.
* Set the filter's process and measurement noise against the precision of the
  navigator and the dynamics of the motion.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

TRUTH, MEASURED, FAST, SLOW = "#8a8a8a", "#8a8a8a", "#e8a33d", "#3dbde8"


def show(axis, values, title, vmin=0.0, vmax=1.0, cmap="gray"):
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axis.set_title(title)
    axis.set_axis_off()
    return handle


# sphinx_gallery_end_ignore
import numpy as np
import torch
from brainweb_dl import get_mri
from scipy import ndimage
from scipy.spatial.transform import Rotation

import bartorch
import bartorch.tools as bt

# %%
#
# The head
# --------
#
# A BrainWeb T1-weighted volume [#brainweb]_, downsampled to 2 mm isotropic
# and padded to :math:`112^3`. Array axis 0 runs inferior to superior (S/I),
# axis 1 posterior to anterior (A/P) and axis 2 left to right (L/R). Poses are
# stated in that frame: a rotation vector in radians about the centre of the
# volume, printed in degrees, and a translation in millimetres.

# sphinx_gallery_start_ignore
volume = ndimage.zoom(get_mri(sub_id=0, contrast="T1").astype(np.float32), 0.5, order=1)
SIZE = 112
volume = np.pad(volume, [((SIZE - n) // 2, SIZE - n - (SIZE - n) // 2) for n in volume.shape])
volume /= volume.max()
centre = (np.array(volume.shape) - 1) / 2
# sphinx_gallery_end_ignore
VOXEL_MM = 2.0


def move(rotvec, translation_mm):
    """The volume after a rigid motion about its centre."""
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    shift = np.asarray(translation_mm) / VOXEL_MM
    return ndimage.affine_transform(
        volume, rotation.T, offset=centre - rotation.T @ (centre + shift), order=1
    )


# %%
#
# The navigator
# -------------
#
# Each navigator is three central planes, axial, coronal and sagittal, each
# acquired along 96 golden-angle radial spokes of 56 samples. The trajectory
# is in grid units of a :math:`56^2` matrix, so the spokes reach half the
# imaging resolution and each plane is reconstructed at 4 mm over the full
# field of view. :func:`~bartorch.tools.reconstruct_navigator` grids a
# plane by the density-compensated adjoint NUFFT, with weights from
# :func:`bartorch.estimate_density`.
#
# ``AXES`` states, for each plane, the direction in the volume of its image's
# rows and columns; it is what relates each plane's in-plane measurement to
# the 3D pose.

NAV, SPOKES = 56, 96
golden = np.pi * (3 - np.sqrt(5))
angle = np.arange(SPOKES)[:, None] * golden
radius = np.arange(NAV) - NAV / 2
trajectory = torch.tensor(
    np.stack([np.cos(angle) * radius, np.sin(angle) * radius], axis=-1).reshape(-1, 2),
    dtype=torch.float32,
)
density = bartorch.estimate_density(trajectory, (NAV, NAV))

AXES = [
    ((0, 1, 0), (0, 0, 1)),  # axial: rows posterior-anterior, columns left-right
    ((1, 0, 0), (0, 0, 1)),  # coronal: rows inferior-superior, columns left-right
    ((1, 0, 0), (0, 1, 0)),  # sagittal: rows inferior-superior, columns posterior-anterior
]
SPACING_MM = VOXEL_MM * SIZE / NAV


def navigator(head, noise=0.0, generator=None):
    """The three navigator planes of ``head``, with complex Gaussian k-space noise."""
    middle = SIZE // 2
    planes = []
    for plane in (head[middle], head[:, middle], head[:, :, middle]):
        samples = bartorch.nufft(
            torch.tensor(plane, dtype=torch.complex64),
            torch.nn.functional.pad(trajectory, (0, 1))[None],
        ).reshape(1, -1)
        if noise:
            samples = samples + noise * torch.randn(
                samples.shape, dtype=torch.complex64, generator=generator
            )
        planes.append(
            bt.reconstruct_navigator(samples, trajectory[None], (NAV, NAV), density=density)[0]
        )
    return planes


reference = navigator(volume)

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
for axis, plane, title in zip(axes, reference, ("axial", "coronal", "sagittal"), strict=True):
    axis.imshow(plane, cmap="gray", origin="lower")
    axis.set_title(f"{title}, {NAV}$^2$ at {SPACING_MM:.0f} mm")
    axis.set_xticks([])
    axis.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# One pose
# --------
#
# :class:`~bartorch.tools.NavigatorMotionTracker` takes the first
# navigator it is given as the reference and returns the pose of every later
# one relative to it. Each plane is registered in 2D by
# :class:`~bartorch.tools.RigidRegistration`; the pose is the
# least-squares solution over the three planes.

rotvec = np.array([0.02, -0.03, 0.05])  # rad
translation = np.array([3.0, -2.0, 4.0])  # mm

tracker = bt.NavigatorMotionTracker(measurement_noise=1e-3)
tracker.track(reference, AXES, spacing=SPACING_MM)
moved = navigator(move(rotvec, translation))
pose = tracker.track(moved, AXES, spacing=SPACING_MM)

measured_rotvec = Rotation.from_matrix(np.asarray(pose.matrix)[:3, :3]).as_rotvec()
measured_translation = np.asarray(pose.translation)
NAMES = ("S/I", "A/P", "L/R")
print(f"{'':>14} {'truth':>7} {'measured':>9}")
for axis, name in enumerate(NAMES):
    print(
        f"rotation {name:>5} {np.degrees(rotvec[axis]):6.2f}° "
        f"{np.degrees(measured_rotvec[axis]):8.2f}°"
    )
for axis, name in enumerate(NAMES):
    print(f"shift    {name:>5} {translation[axis]:5.1f} mm {measured_translation[axis]:6.2f} mm")

# %%
#
# The measured pose is assessed on the navigator itself: the head moved by the
# measured pose, navigated again, is compared with the navigator after the
# motion. Without correction, the difference is that of the motion; with the
# measured pose, what remains is the error of the pose.

realigned = navigator(move(measured_rotvec, measured_translation))


def nrmse(estimate, target):
    return float((estimate - target).norm() / target.norm())


for plane, before, after, target in zip(
    ("axial", "coronal", "sagittal"), reference, realigned, moved, strict=True
):
    print(
        f"{plane:>8}: NRMSE against the moved navigator, reference {nrmse(before, target):.3f}, "
        f"measured pose {nrmse(after, target):.3f}"
    )

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(2, 3, figsize=(8.4, 5.6))
for column, (plane, before, after, target) in enumerate(
    zip(("axial", "coronal", "sagittal"), reference, realigned, moved, strict=True)
):
    scale = float(target.abs().max())
    show(
        axes[0, column],
        (target - before) / scale,
        f"{plane}: moved - reference\nNRMSE {nrmse(before, target):.3f}",
        -0.5,
        0.5,
        "RdBu_r",
    )
    handle = show(
        axes[1, column],
        (target - after) / scale,
        f"moved - measured pose\nNRMSE {nrmse(after, target):.3f}",
        -0.5,
        0.5,
        "RdBu_r",
    )
figure.colorbar(handle, ax=axes, fraction=0.03, label="difference / peak")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The residual error of the pose, a fraction of a degree and of a millimetre,
# has two sources: the navigator's 4 mm resolution, and the
# through-plane motion, which each plane sees as a change of the anatomy in it
# rather than as a rigid motion within it. It is a small fraction of the 4 mm
# navigator voxel, and the residual difference is confined to the edges of the
# head.
#
# A scan
# ------
#
# The head nods, a rotation about the left-right axis, by up to 2.3° with a
# period of 8 s, and drifts by 3 mm along the superior-inferior axis over
# 12 s, with a navigator every 0.5 s. Each
# navigator carries complex Gaussian k-space noise, so each measured pose
# carries registration error. The filter's ``process_noise`` is the variance
# of the acceleration it allows between navigators, per pose coordinate, in
# rad²/s⁴ and mm²/s⁴; ``measurement_noise`` is the variance it assigns to each
# measured coordinate. The trace is filtered with three values of
# ``process_noise``; the largest leaves the filter at the measurements.

DT, COUNT = 0.5, 24
seconds = DT * np.arange(1, COUNT + 1)
nod = 0.04 * np.sin(2 * np.pi * seconds / 8)  # rad, about L/R
drift = 3.0 * seconds / seconds[-1]  # mm, along S/I
truth = np.zeros((COUNT, 6))
truth[:, 2], truth[:, 3] = nod, drift

generator = torch.Generator().manual_seed(0)
scan = [
    navigator(move((0.0, 0.0, a), (d, 0.0, 0.0)), noise=0.5, generator=generator)
    for a, d in zip(nod, drift, strict=True)
]


def track(process_noise):
    tracker = bt.NavigatorMotionTracker(process_noise=process_noise, measurement_noise=1e-4)
    tracker.track(reference, AXES, spacing=SPACING_MM)
    poses = [tracker.track(planes, AXES, dt=DT, spacing=SPACING_MM) for planes in scan]
    return np.array(
        [
            np.concatenate(
                [
                    Rotation.from_matrix(np.asarray(p.matrix)[:3, :3]).as_rotvec(),
                    np.asarray(p.translation),
                ]
            )
            for p in poses
        ]
    )


traces = {noise: track(noise) for noise in (1e2, 1e-3, 1e-4)}
rms = {noise: np.sqrt(np.mean((trace - truth) ** 2, axis=0)) for noise, trace in traces.items()}
for noise, error in rms.items():
    print(
        f"process_noise {noise:<6g} rms error: rotation "
        + ", ".join(f"{n} {np.degrees(e):.2f}°" for n, e in zip(NAMES, error[:3], strict=True))
        + "; shift "
        + ", ".join(f"{n} {e:.2f} mm" for n, e in zip(NAMES, error[3:], strict=True))
    )

# %%

# sphinx_gallery_start_ignore
LABELS = {1e2: "process_noise 1e2", 1e-3: "1e-3", 1e-4: "1e-4"}
figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
for axis, index, scale, label, unit in (
    (axes[0], 2, np.degrees(1.0), "nod: rotation about L/R", "°"),
    (axes[1], 3, 1.0, "drift: shift along S/I", "mm"),
):
    axis.plot(seconds, scale * truth[:, index], color=TRUTH, linewidth=3.0, label="head")
    for noise, style, colour in ((1e2, ".", MEASURED), (1e-3, "-", FAST), (1e-4, "-", SLOW)):
        axis.plot(
            seconds,
            scale * traces[noise][:, index],
            style,
            color=colour,
            label=f"{LABELS[noise]}, rms {scale * rms[noise][index]:.2f} {unit}",
        )
    axis.set_xlabel("time [s]")
    axis.set_ylabel(f"{label.split(':')[1].split()[0]} [{unit}]")
    axis.set_title(label)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# A large ``process_noise`` leaves the filter at the measurements; a small one
# makes it trust its constant-velocity prediction, which smooths the
# registration error and lags a change of direction: at ``1e-4`` the error of
# the nod grows while that of the other coordinates falls. An error common to
# every navigator, such as the offset of the A/P shift, is a bias of the
# measurement and is not reduced by any value. The value that minimizes the
# error depends on the motion and on the navigator's precision, and is set
# against a motion trace of the application.
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
# .. [#brainweb] Collins DL, Zijdenbos AP, Kollokian V, Sled JG, Kabani NJ, Holmes CJ,
#    Evans AC. Design and construction of a realistic digital brain phantom.
#    *IEEE Trans Med Imaging* 17(3):463-468 (1998).
#    https://doi.org/10.1109/42.712135
