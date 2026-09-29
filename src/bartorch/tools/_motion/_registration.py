"""Rigid registration, a constant-velocity Kalman filter over it, and navigator tracking."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = [
    "NavigatorMotionTracker",
    "RigidMotionEKF",
    "RigidMotionEstimate",
    "RigidRegistration",
]


def _simpleitk():
    try:
        import SimpleITK
    except ImportError as error:
        raise ImportError(
            "rigid registration requires SimpleITK: pip install 'bartorch[motion]'"
        ) from error
    return SimpleITK


@contextmanager
def _itk_threads(count: int) -> Iterator[None]:
    """Hold ITK's process-wide default thread count at ``count``, restoring it on exit."""
    process = _simpleitk().ProcessObject
    restore = process.GetGlobalDefaultNumberOfThreads()
    process.SetGlobalDefaultNumberOfThreads(int(count))
    try:
        yield
    finally:
        process.SetGlobalDefaultNumberOfThreads(restore)


@dataclass(frozen=True)
class RigidMotionEstimate:
    """A 2D or 3D rigid transform in SimpleITK physical coordinates.

    Parameters
    ----------
    parameters : array_like
        ``(angle, tx, ty)`` in 2D or ``(angle_x, angle_y, angle_z, tx, ty, tz)``
        in 3D, as SimpleITK's ``Euler2DTransform`` and ``Euler3DTransform``
        take them.  Angles are in radians, translations in the unit of the
        image spacing.
    center : array_like
        Centre of rotation, in physical ``(x, y[, z])`` order.
    covariance : array_like, default=None
        Covariance of ``parameters``, in the same order.
    metric : float, default=None
        Final registration metric value.
    stop_condition : str, default=None
        SimpleITK optimizer stop description.

    Notes
    -----
    SimpleITK indexes an image in ``(x, y[, z])`` order, the reverse of the
    array's axes: ``tx`` is along the last array axis.  The 3D angles compose as
    :math:`R_z R_x R_y`.  The arrays are stored as float64 NumPy copies.
    """

    parameters: np.ndarray
    center: np.ndarray
    covariance: np.ndarray | None = None
    metric: float | None = None
    stop_condition: str | None = None

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=np.float64).reshape(-1)
        if parameters.size not in {3, 6}:
            raise ValueError("rigid motion requires three 2D or six 3D parameters")
        dimension = 2 if parameters.size == 3 else 3
        center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if center.size != dimension:
            raise ValueError("center dimensionality does not match parameters")
        covariance = self.covariance
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64)
            if covariance.shape != (parameters.size, parameters.size):
                raise ValueError("covariance shape does not match motion parameters")
        object.__setattr__(self, "parameters", parameters.copy())
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "covariance", None if covariance is None else covariance.copy())

    @property
    def dimension(self) -> int:
        """Two or three spatial dimensions."""
        return 2 if self.parameters.size == 3 else 3

    @property
    def angles(self) -> np.ndarray:
        """Euler angles in radians."""
        return self.parameters[: 1 if self.dimension == 2 else 3].copy()

    @property
    def translation(self) -> np.ndarray:
        """Translation in physical ``(x, y[, z])`` order."""
        return self.parameters[1 if self.dimension == 2 else 3 :].copy()

    @property
    def matrix(self) -> np.ndarray:
        """The transform about its centre as a homogeneous ``(d + 1, d + 1)`` matrix."""
        transform = _transform_from_estimate(self)
        rotation = np.asarray(transform.GetMatrix()).reshape(self.dimension, self.dimension)
        offset = self.center + self.translation - rotation @ self.center
        matrix = np.eye(self.dimension + 1)
        matrix[: self.dimension, : self.dimension] = rotation
        matrix[: self.dimension, self.dimension] = offset
        return matrix


class RigidRegistration:
    """Multi-resolution rigid registration of magnitude images with SimpleITK.

    Regular-step gradient descent over an Euler transform initialized at the
    images' geometric centres, with linear interpolation.  Complex inputs are
    registered by their modulus, and both images are normalized to zero mean
    and unit standard deviation.

    Parameters
    ----------
    metric : {"correlation", "mutual_information", "mean_squares"}, default='correlation'
        Similarity metric; mutual information uses 32 histogram bins.
    iterations : int, default=50
        Optimizer iterations per resolution level.
    sampling_percentage : float, default=0.2
        Fraction of voxels the metric samples at random, with a fixed seed.
        One samples every voxel.
    shrink_factors : sequence of int, default=(4, 2, 1)
        Pyramid shrink factor per level, coarsest first.
    smoothing_sigmas : sequence of float, default=(2.0, 1.0, 0.0)
        Gaussian smoothing per level, in physical units.
    learning_rate : float, default=1.0
        Initial optimizer step.
    min_step : float, default=0.0001
        Step at which the optimizer stops.
    threads : int, default=1
        ITK threads for the registration.  ITK's thread count is process-wide;
        it is set for the call and restored after it.

    Examples
    --------
    >>> estimate = RigidRegistration()(fixed, moving, spacing=(1.0, 1.0))
    >>> estimate.parameters  # (angle, tx, ty)
    """

    def __init__(
        self,
        *,
        metric: str = "correlation",
        iterations: int = 50,
        sampling_percentage: float = 0.2,
        shrink_factors: Sequence[int] = (4, 2, 1),
        smoothing_sigmas: Sequence[float] = (2.0, 1.0, 0.0),
        learning_rate: float = 1.0,
        min_step: float = 1e-4,
        threads: int = 1,
    ) -> None:
        if metric not in {"correlation", "mutual_information", "mean_squares"}:
            raise ValueError("unsupported registration metric")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if not 0.0 < sampling_percentage <= 1.0:
            raise ValueError("sampling_percentage must lie in (0, 1]")
        shrink = tuple(int(value) for value in shrink_factors)
        smoothing = tuple(float(value) for value in smoothing_sigmas)
        if not shrink or len(shrink) != len(smoothing):
            raise ValueError("pyramid factors and sigmas must have equal length")
        if any(value < 1 for value in shrink) or any(value < 0 for value in smoothing):
            raise ValueError("invalid multi-resolution pyramid")
        if learning_rate <= 0.0 or min_step <= 0.0:
            raise ValueError("optimizer steps must be positive")
        if int(threads) < 1:
            raise ValueError("threads must be at least one")
        self.metric = metric
        self.iterations = int(iterations)
        self.sampling_percentage = float(sampling_percentage)
        self.shrink_factors = shrink
        self.smoothing_sigmas = smoothing
        self.learning_rate = float(learning_rate)
        self.min_step = float(min_step)
        self.threads = int(threads)

    def estimate(
        self,
        fixed: Any,
        moving: Any,
        *,
        initial: RigidMotionEstimate | None = None,
        spacing: Sequence[float] | None = None,
    ) -> RigidMotionEstimate:
        """Register ``moving`` to ``fixed``.

        Parameters
        ----------
        fixed, moving : torch.Tensor or numpy.ndarray
            One 2D or 3D image each, after squeezing singleton axes.
        initial : RigidMotionEstimate, default=None
            Starting parameters; the geometric-centre initialization otherwise.
        spacing : sequence of float, default=None
            Voxel size per axis in SimpleITK ``(x, y[, z])`` order; one
            otherwise.  Translations come back in its unit.

        Returns
        -------
        RigidMotionEstimate
            The transform mapping points of ``fixed`` into ``moving``, with the
            final metric value and the optimizer's stop condition.
        """
        sitk = _simpleitk()
        fixed_image = _sitk_image(fixed, spacing)
        moving_image = _sitk_image(moving, spacing)
        if fixed_image.GetDimension() != moving_image.GetDimension():
            raise ValueError("fixed and moving images must have equal dimensionality")
        dimension = fixed_image.GetDimension()
        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler2DTransform() if dimension == 2 else sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        if initial is not None:
            if initial.dimension != dimension:
                raise ValueError("initial transform dimensionality does not match images")
            transform.SetParameters(tuple(map(float, initial.parameters)))

        registration = sitk.ImageRegistrationMethod()
        if self.metric == "correlation":
            registration.SetMetricAsCorrelation()
        elif self.metric == "mutual_information":
            registration.SetMetricAsMattesMutualInformation(32)
        else:
            registration.SetMetricAsMeanSquares()
        if self.sampling_percentage < 1.0:
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            registration.SetMetricSamplingPercentage(self.sampling_percentage, seed=42)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetShrinkFactorsPerLevel(self.shrink_factors)
        registration.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=self.learning_rate,
            minStep=self.min_step,
            numberOfIterations=self.iterations,
            gradientMagnitudeTolerance=1e-8,
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetInitialTransform(transform, inPlace=True)
        with _itk_threads(self.threads):
            result = registration.Execute(fixed_image, moving_image)
        if result.GetName() == "CompositeTransform":
            if result.GetNumberOfTransforms() != 1:
                raise RuntimeError("registration returned an unexpected transform stack")
            result = result.GetNthTransform(0)
        return RigidMotionEstimate(
            parameters=np.asarray(result.GetParameters()),
            center=np.asarray(result.GetCenter()),
            metric=float(registration.GetMetricValue()),
            stop_condition=registration.GetOptimizerStopConditionDescription(),
        )

    def resample(
        self,
        moving: Any,
        estimate: RigidMotionEstimate,
        *,
        reference: Any | None = None,
        spacing: Sequence[float] | None = None,
        default_value: float = 0.0,
    ) -> torch.Tensor:
        """Resample ``moving`` onto the grid of ``reference`` through ``estimate``.

        Parameters
        ----------
        moving : torch.Tensor or numpy.ndarray
            Image to resample; complex values are resampled by their modulus.
        estimate : RigidMotionEstimate
            Transform from the reference grid into ``moving``, as
            :meth:`estimate` returns it.
        reference : torch.Tensor or numpy.ndarray, default=None
            Image whose grid is resampled onto; the grid of ``moving`` otherwise.
        spacing : sequence of float, default=None
            Voxel size, as in :meth:`estimate`.
        default_value : float, default=0.0
            Value outside ``moving``.

        Returns
        -------
        torch.Tensor
            Real ``float32`` image on the grid of ``reference``, on the device
            of ``moving`` when it is a tensor.
        """
        sitk = _simpleitk()
        moving_image = _sitk_image(moving, spacing, normalize=False)
        reference_image = (
            moving_image if reference is None else _sitk_image(reference, spacing, normalize=False)
        )
        if estimate.dimension != moving_image.GetDimension():
            raise ValueError("motion dimensionality does not match the moving image")
        output = sitk.Resample(
            moving_image,
            reference_image,
            _transform_from_estimate(estimate),
            sitk.sitkLinear,
            float(default_value),
            sitk.sitkFloat32,
        )
        device = moving.device if isinstance(moving, torch.Tensor) else None
        return torch.as_tensor(sitk.GetArrayFromImage(output), device=device)

    def __call__(self, fixed: Any, moving: Any, **kwargs: Any) -> RigidMotionEstimate:
        """:meth:`estimate`."""
        return self.estimate(fixed, moving, **kwargs)


class RigidMotionEKF:
    """Constant-velocity extended Kalman filter over registered rigid poses.

    The state is the pose and its rate of change, ``2 * dof`` entries with
    ``dof`` three in 2D and six in 3D.  The process noise is that of a
    piecewise-constant acceleration of variance ``process_noise`` per pose
    coordinate; the measurement is the pose a registration returns.  Angle
    innovations and angles are wrapped to :math:`[-\\pi, \\pi)`.

    Parameters
    ----------
    registration : RigidRegistration
        Measurement model used by :meth:`step`.
    dimension : {2, 3}, default=3
        Spatial dimensions.
    process_noise : float or sequence of float, default=0.0001
        Acceleration variance, one value or one per pose coordinate.
    measurement_noise : float or sequence of float, default=0.01
        Measurement variance, one value or one per pose coordinate, used when a
        measurement carries no covariance.
    initial_covariance : float, default=1.0
        Diagonal of the state covariance at initialization.

    Attributes
    ----------
    state : numpy.ndarray
        Pose followed by its rate.
    covariance : numpy.ndarray
        State covariance.
    initialized : bool
        Whether a measurement has been fused; the first one initializes the
        state rather than updating it.
    """

    def __init__(
        self,
        registration: RigidRegistration,
        *,
        dimension: int = 3,
        process_noise: float | Sequence[float] = 1e-4,
        measurement_noise: float | Sequence[float] = 1e-2,
        initial_covariance: float = 1.0,
    ) -> None:
        if dimension not in {2, 3}:
            raise ValueError("dimension must be two or three")
        if initial_covariance <= 0.0:
            raise ValueError("initial_covariance must be positive")
        self.registration = registration
        self.dimension = int(dimension)
        self.dof = 3 if dimension == 2 else 6
        self.rotation_count = 1 if dimension == 2 else 3
        self.process_noise = _noise_vector(process_noise, self.dof, "process_noise")
        self.measurement_noise = _noise_vector(measurement_noise, self.dof, "measurement_noise")
        self.initial_covariance = float(initial_covariance)
        self.state = np.zeros(2 * self.dof, dtype=np.float64)
        self.covariance = np.eye(2 * self.dof) * self.initial_covariance
        self.center = np.zeros(dimension, dtype=np.float64)
        self.initialized = False
        self.last_measurement: RigidMotionEstimate | None = None

    @property
    def pose(self) -> RigidMotionEstimate:
        """Filtered pose, with its covariance and the last measurement's metric."""
        last = self.last_measurement
        return RigidMotionEstimate(
            parameters=self.state[: self.dof],
            center=self.center,
            covariance=self.covariance[: self.dof, : self.dof],
            metric=None if last is None else last.metric,
            stop_condition=None if last is None else last.stop_condition,
        )

    @property
    def velocity(self) -> np.ndarray:
        """Angular and translational rates, in pose order, per unit of ``dt``."""
        return self.state[self.dof :].copy()

    def reset(self, initial: RigidMotionEstimate | None = None) -> None:
        """Return to the uninitialized state, or initialize at ``initial``."""
        self.state.fill(0.0)
        self.covariance = np.eye(2 * self.dof) * self.initial_covariance
        self.center.fill(0.0)
        self.initialized = False
        self.last_measurement = None
        if initial is not None:
            self._initialize(initial)

    def predict(self, dt: float = 1.0) -> RigidMotionEstimate:
        """Propagate the state by ``dt`` and return the predicted pose.

        Parameters
        ----------
        dt : float, default=1.0
            Time step, in the unit the noise variances are stated per.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        transition = np.eye(2 * self.dof)
        transition[: self.dof, self.dof :] = dt * np.eye(self.dof)
        noise = np.zeros_like(self.covariance)
        diagonal = np.diag(self.process_noise)
        noise[: self.dof, : self.dof] = 0.25 * dt**4 * diagonal
        noise[: self.dof, self.dof :] = 0.5 * dt**3 * diagonal
        noise[self.dof :, : self.dof] = 0.5 * dt**3 * diagonal
        noise[self.dof :, self.dof :] = dt**2 * diagonal
        self.state = transition @ self.state
        self.state[: self.rotation_count] = _wrap_angles(self.state[: self.rotation_count])
        self.covariance = transition @ self.covariance @ transition.T + noise
        return self.pose

    def update(
        self,
        measurement: RigidMotionEstimate | Sequence[float] | np.ndarray,
        *,
        covariance: Any | None = None,
    ) -> RigidMotionEstimate:
        """Fuse a measured pose and return the filtered pose.

        The covariance update is the Joseph form.

        Parameters
        ----------
        measurement : RigidMotionEstimate or array_like
            Measured pose; bare parameters are taken about the filter's centre.
        covariance : array_like, default=None
            Measurement covariance; that of ``measurement``, or
            ``measurement_noise`` on the diagonal, otherwise.
        """
        if not isinstance(measurement, RigidMotionEstimate):
            measurement = RigidMotionEstimate(
                parameters=np.asarray(measurement), center=self.center
            )
        if measurement.dimension != self.dimension:
            raise ValueError("measurement dimensionality does not match the filter")
        if not self.initialized:
            self._initialize(measurement)
            return self.pose
        observation = np.zeros((self.dof, 2 * self.dof))
        observation[:, : self.dof] = np.eye(self.dof)
        innovation = measurement.parameters - observation @ self.state
        innovation[: self.rotation_count] = _wrap_angles(innovation[: self.rotation_count])
        selected = covariance
        if selected is None:
            selected = measurement.covariance
        if selected is None:
            selected = np.diag(self.measurement_noise)
        selected = np.asarray(selected, dtype=np.float64)
        if selected.shape != (self.dof, self.dof):
            raise ValueError("measurement covariance has the wrong shape")
        innovation_covariance = observation @ self.covariance @ observation.T + selected
        gain = np.linalg.solve(innovation_covariance, observation @ self.covariance).T
        self.state = self.state + gain @ innovation
        self.state[: self.rotation_count] = _wrap_angles(self.state[: self.rotation_count])
        residual = np.eye(2 * self.dof) - gain @ observation
        self.covariance = residual @ self.covariance @ residual.T + gain @ selected @ gain.T
        self.center = measurement.center.copy()
        self.last_measurement = measurement
        return self.pose

    def step(
        self,
        fixed: Any,
        moving: Any,
        *,
        dt: float = 1.0,
        spacing: Sequence[float] | None = None,
    ) -> RigidMotionEstimate:
        """Predict, register ``moving`` to ``fixed`` from the prediction, and update.

        Parameters
        ----------
        fixed, moving : torch.Tensor or numpy.ndarray
            Images, as for :meth:`RigidRegistration.estimate`.
        dt : float, default=1.0
            Time since the previous step.
        spacing : sequence of float, default=None
            Voxel size, as for :meth:`RigidRegistration.estimate`.
        """
        initial = self.predict(dt) if self.initialized else None
        measurement = self.registration.estimate(fixed, moving, initial=initial, spacing=spacing)
        return self.update(measurement)

    def _initialize(self, measurement: RigidMotionEstimate) -> None:
        self.state.fill(0.0)
        self.state[: self.dof] = measurement.parameters
        self.center = measurement.center.copy()
        self.covariance = np.eye(2 * self.dof) * self.initial_covariance
        if measurement.covariance is not None:
            self.covariance[: self.dof, : self.dof] = measurement.covariance
        self.last_measurement = measurement
        self.initialized = True


class NavigatorMotionTracker:
    """Six-degree-of-freedom rigid pose from a navigator's 2D planes, filtered over time.

    Each plane is registered in 2D against the same plane of the first
    navigator.  A plane measures the rotation about its normal and the
    translation along its two in-plane axes; these are linear in the rotation
    vector and translation of the 3D pose, which are solved for by least
    squares over the planes and passed to a :class:`RigidMotionEKF`.  The
    planes' normals must span the three rotations and their in-plane axes the
    three translations; three orthogonal planes are the usual set.  The
    in-plane model holds while the out-of-plane motion is small enough that
    each plane still intersects the anatomy of its reference.

    Parameters
    ----------
    registration : RigidRegistration, default=None
        Registration of each plane; a :class:`RigidRegistration` with its
        defaults otherwise.
    process_noise : float or sequence of float, default=0.0001
        Passed to :class:`RigidMotionEKF`.
    measurement_noise : float or sequence of float, default=0.01
        Passed to :class:`RigidMotionEKF`.
    initial_covariance : float, default=1.0
        Passed to :class:`RigidMotionEKF`.

    Attributes
    ----------
    filter : RigidMotionEKF
        The filter carrying the pose.
    reference : list or None
        Planes of the first navigator; ``None`` before the first :meth:`track`.
    """

    def __init__(
        self,
        *,
        registration: RigidRegistration | None = None,
        process_noise: float | Sequence[float] = 1e-4,
        measurement_noise: float | Sequence[float] = 1e-2,
        initial_covariance: float = 1.0,
    ) -> None:
        self.filter = RigidMotionEKF(
            RigidRegistration() if registration is None else registration,
            dimension=3,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_covariance=initial_covariance,
        )
        self.reference: list[Any] | None = None

    @property
    def registration(self) -> RigidRegistration:
        """Registration each plane is measured with."""
        return self.filter.registration

    @property
    def pose(self) -> RigidMotionEstimate:
        """Current filtered pose, about the navigator's centre."""
        return self.filter.pose

    def reset(self) -> None:
        """Discard the reference navigator and the filtered pose."""
        self.filter.reset()
        self.reference = None

    def track(
        self,
        planes: Sequence[Any],
        axes: Sequence[Sequence[Sequence[float]]],
        *,
        dt: float = 1.0,
        spacing: float | Sequence[float] = 1.0,
    ) -> RigidMotionEstimate:
        """Register one navigator against the reference and return the filtered pose.

        Parameters
        ----------
        planes : sequence of torch.Tensor or numpy.ndarray
            One 2D image per plane, in the same order at every call.
        axes : sequence
            Per plane, ``(row_axis, column_axis)``: unit 3-vectors, in the frame
            the pose is wanted in, along which the image's first and second
            array axes run.  The two must be perpendicular.
        dt : float, default=1.0
            Time since the previous navigator.
        spacing : float or sequence of float, default=1.0
            In-plane pixel size, one value or one per plane.  Translations are
            returned in its unit.

        Returns
        -------
        RigidMotionEstimate
            The filtered 3D pose relative to the reference navigator, about the
            origin of the ``axes`` frame; zero for the first navigator.

        Raises
        ------
        ValueError
            If the number of planes differs from the reference, the axes are
            not a perpendicular pair per plane, or the planes leave a rotation
            or translation component unmeasured.
        """
        row_axes, column_axes = _plane_axes(axes, len(planes))
        spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float64).reshape(-1), (len(planes),))
        if self.reference is None:
            self.reference = [_copy(plane) for plane in planes]
            return self.filter.update(
                RigidMotionEstimate(parameters=np.zeros(6), center=np.zeros(3))
            )
        if len(planes) != len(self.reference):
            raise ValueError(
                f"this navigator has {len(planes)} planes and the reference "
                f"has {len(self.reference)}"
            )
        predicted = self.filter.predict(dt) if self.filter.initialized else None
        measurements = [
            self.filter.registration.estimate(
                reference,
                plane,
                initial=None if predicted is None else _project_onto_plane(predicted, row, column),
                spacing=(float(step), float(step)),
            )
            for reference, plane, row, column, step in zip(
                self.reference, planes, row_axes, column_axes, spacing, strict=True
            )
        ]
        return self.filter.update(_pose_from_planes(measurements, row_axes, column_axes))


def _copy(plane: Any) -> Any:
    return plane.detach().clone() if isinstance(plane, torch.Tensor) else np.array(plane)


def _as_magnitude_array(value: Any, *, normalize: bool) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        value = (value.abs() if value.is_complex() else value).numpy()
    array = np.asarray(value)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = np.squeeze(array)
    if array.ndim not in {2, 3}:
        raise ValueError("rigid registration requires one 2D or 3D image")
    if not np.isfinite(array).all():
        raise ValueError("registration images must contain only finite values")
    array = array.astype(np.float32, copy=False)
    if not normalize:
        return array
    return (array - float(array.mean())) / max(float(array.std()), 1e-6)


def _sitk_image(value: Any, spacing: Sequence[float] | None, *, normalize: bool = True) -> Any:
    image = _simpleitk().GetImageFromArray(_as_magnitude_array(value, normalize=normalize))
    if spacing is not None:
        selected = tuple(float(item) for item in spacing)
        if len(selected) != image.GetDimension() or any(item <= 0 for item in selected):
            raise ValueError("spacing must contain one positive value per dimension")
        image.SetSpacing(selected)
    return image


def _transform_from_estimate(estimate: RigidMotionEstimate) -> Any:
    sitk = _simpleitk()
    transform = sitk.Euler2DTransform() if estimate.dimension == 2 else sitk.Euler3DTransform()
    transform.SetCenter(tuple(map(float, estimate.center)))
    transform.SetParameters(tuple(map(float, estimate.parameters)))
    return transform


def _plane_axes(
    axes: Sequence[Sequence[Sequence[float]]], count: int
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(axes, dtype=np.float64)
    if selected.shape != (count, 2, 3):
        raise ValueError(
            "axes must give a row and a column direction, as three-vectors, for every plane"
        )
    norms = np.linalg.norm(selected, axis=-1, keepdims=True)
    if not np.all(norms > 0.0):
        raise ValueError("plane axes must be non-zero directions")
    selected = selected / norms
    if not np.allclose(np.sum(selected[:, 0] * selected[:, 1], axis=-1), 0.0, atol=1e-6):
        raise ValueError("a plane's row and column directions must be perpendicular")
    return selected[:, 0], selected[:, 1]


def _pose_from_planes(
    measurements: Sequence[RigidMotionEstimate],
    row_axes: np.ndarray,
    column_axes: np.ndarray,
) -> RigidMotionEstimate:
    """Least-squares 3D pose from per-plane ``(angle, tx, ty)`` measurements.

    A plane's SimpleITK ``x`` is its column direction and ``y`` its row
    direction.  Its angle rotates the column direction towards the row
    direction, which about the normal ``row x column`` is the negative sense.
    """
    parameters = np.asarray([m.parameters for m in measurements], dtype=np.float64)
    normals = np.cross(row_axes, column_axes)
    rotation = _solve(-normals, parameters[:, 0], "rotation")
    translation = _solve(
        np.concatenate([column_axes, row_axes]),
        np.concatenate([parameters[:, 1], parameters[:, 2]]),
        "translation",
    )
    return RigidMotionEstimate(
        parameters=np.concatenate([_rotvec_to_euler(rotation), translation]),
        center=np.zeros(3),
    )


def _project_onto_plane(
    pose: RigidMotionEstimate, row_axis: np.ndarray, column_axis: np.ndarray
) -> RigidMotionEstimate:
    """The ``(angle, tx, ty)`` one plane measures of the 3D ``pose``."""
    rotation = _euler_to_rotvec(pose.angles)
    translation = pose.translation
    return RigidMotionEstimate(
        parameters=(
            -float(rotation @ np.cross(row_axis, column_axis)),
            float(translation @ column_axis),
            float(translation @ row_axis),
        ),
        center=np.zeros(2),
    )


def _solve(rows: np.ndarray, values: np.ndarray, unknown: str) -> np.ndarray:
    solution, _, rank, _ = np.linalg.lstsq(rows, values, rcond=None)
    if rank < rows.shape[1]:
        raise ValueError(
            f"the navigator's planes do not span the {unknown}: they leave at "
            f"least one of its three components unmeasured"
        )
    return solution


# SimpleITK's Euler3DTransform composes its angles as Rz @ Rx @ Ry, which SciPy
# spells "yxz", with the angles in that order rather than in the transform's.
_EULER_SEQUENCE = "yxz"
_EULER_ORDER = (1, 0, 2)


def _rotvec_to_euler(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    angles = Rotation.from_rotvec(rotation).as_euler(_EULER_SEQUENCE)
    return angles[list(np.argsort(_EULER_ORDER))]


def _euler_to_rotvec(angles: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler(_EULER_SEQUENCE, np.asarray(angles)[list(_EULER_ORDER)]).as_rotvec()


def _noise_vector(value: float | Sequence[float], size: int, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=np.float64)
    if selected.ndim == 0:
        selected = np.full(size, float(selected))
    selected = selected.reshape(-1)
    if selected.size != size or np.any(selected <= 0.0):
        raise ValueError(f"{name} must be positive and have one value per parameter")
    return selected


def _wrap_angles(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2 * np.pi) - np.pi
