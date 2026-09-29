"""
=================================
Rigid head motion from navigators
=================================

Six-degree-of-freedom rigid head motion measured from three orthogonal
radial navigator planes, and carried across a scan by a Kalman filter, with
the rigid-motion functions of :mod:`bartorch.tools`.

A navigator is a short, low-resolution acquisition interleaved with the
imaging readouts, reconstructed and registered against the first navigator of
the scan to measure how the head has moved since [#ehman]_. A 2D plane
measures three of the six degrees of freedom: the rotation about its normal
and the translations along its two in-plane axes. Three orthogonal planes
measure each rotation once and each translation twice, and the rigid pose is
solved from the nine measurements by least squares. Between navigators, a
constant-velocity extended Kalman filter predicts the pose and weighs each new
measurement against the prediction, as in PROMO [#promo]_, which tracks the
head with three orthogonal spiral navigators.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 10,
        "figure.constrained_layout.use": True,
    }
)
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
# and padded to :math:`112^3`. Array axis 0 runs inferior to superior, axis 1
# posterior to anterior and axis 2 left to right; poses below are stated in
# that frame, rotations in radians about the volume's centre and translations
# in millimetres.

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

rotvec = np.array([0.02, -0.03, 0.05])
translation = np.array([3.0, -2.0, 4.0])

tracker = bt.NavigatorMotionTracker(measurement_noise=1e-3)
tracker.track(reference, AXES, spacing=SPACING_MM)
pose = tracker.track(navigator(move(rotvec, translation)), AXES, spacing=SPACING_MM)

measured_rotvec = Rotation.from_matrix(np.asarray(pose.matrix)[:3, :3]).as_rotvec()
print("rotation vector, rad   truth", rotvec, " measured", measured_rotvec.round(4))
print(
    "translation, mm        truth", translation, " measured", np.asarray(pose.translation).round(2)
)

# %%
#
# The residual error, a few milliradians and a few tenths of a millimetre, has
# two sources: the navigator's 4 mm resolution, and the out-of-plane motion,
# which each plane sees as a change of the anatomy in it rather than as a
# rigid motion within it.
#
# A scan
# ------
#
# The head nods about the left-right axis and drifts along the
# inferior-superior axis over 12 s, with a navigator every 0.5 s. Each
# navigator carries complex Gaussian k-space noise, so each measured pose
# carries registration error. The filter's ``process_noise`` is the variance
# of the acceleration it allows between navigators, per pose coordinate, in
# rad²/s⁴ and mm²/s⁴; ``measurement_noise`` is the variance it assigns to each
# measured coordinate. The trace is filtered with three values of
# ``process_noise``; the largest leaves the filter at the measurements.

DT, COUNT = 0.5, 24
seconds = DT * np.arange(1, COUNT + 1)
nod = 0.04 * np.sin(2 * np.pi * seconds / 8)  # rad, about axis 2
drift = 3.0 * seconds / seconds[-1]  # mm, along axis 0
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
for noise, trace in traces.items():
    rms = np.sqrt(np.mean((trace - truth) ** 2, axis=0))
    print(
        f"process_noise {noise:<6g} rotation rms error {rms[:3].round(4)} rad"
        f"   translation rms error {rms[3:].round(2)} mm"
    )

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))
for axis, index, scale, label in (
    (axes[0], 2, 1e3, "rotation about axis 2, mrad"),
    (axes[1], 3, 1.0, "translation along axis 0, mm"),
):
    axis.plot(seconds, scale * truth[:, index], "k-", label="head")
    axis.plot(seconds, scale * traces[1e2][:, index], ".", color="0.5", label="process_noise 1e2")
    axis.plot(
        seconds, scale * traces[1e-3][:, index], "-", color="crimson", label="process_noise 1e-3"
    )
    axis.plot(
        seconds, scale * traces[1e-4][:, index], "-", color="steelblue", label="process_noise 1e-4"
    )
    axis.set_xlabel("s")
    axis.set_title(label)
axes[0].legend(fontsize=7)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# A large ``process_noise`` leaves the filter at the measurements; a small one
# makes it trust its constant-velocity prediction, which smooths the
# registration error and lags a change of direction: at ``1e-4`` the error of
# the nod grows while that of the other coordinates falls. An error
# common to every navigator, such as the offset along axis 1, is a bias of the
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
