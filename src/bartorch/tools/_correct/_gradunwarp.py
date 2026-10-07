"""Gradient nonlinearity correction from a coil's spherical-harmonic coefficients.

The spherical-harmonic conversion and coordinate conventions are adapted from
the MIT-licensed HCP ``gradunwarp`` package and its ``.dat`` converter.  No
coefficient table is distributed with this package or written by it.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import torch

__all__ = ["CoefficientAccessor", "GradientCoefficients", "Gradunwarp"]


def _simpleitk():
    try:
        import SimpleITK
    except ImportError as error:
        raise ImportError(
            "gradient nonlinearity correction requires SimpleITK: pip install 'bartorch[correct]'"
        ) from error
    return SimpleITK


HarmonicBasis = Literal["unnormalized", "normalized"]
CoefficientFormat = Literal["auto", "dat", "grad", "coef"]
CoefficientPayload: TypeAlias = str | bytes | bytearray

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_DAT_SCALE_RE = re.compile(
    rf"^\s*SCALE(?P<axis>[XYZ])(?P<order>\d+)\s+(?P<value>{_FLOAT_PATTERN})",
    re.IGNORECASE | re.MULTILINE,
)
_SIEMENS_GRAD_RE = re.compile(
    rf"^\s*\d+\s+(?P<kind>[AB])\s*\(\s*(?P<n>\d+)\s*,\s*(?P<m>\d+)"
    rf"\s*\)\s+(?P<value>{_FLOAT_PATTERN})\s+(?P<axis>[xyz])\b",
    re.IGNORECASE,
)
_SIEMENS_R0_RE = re.compile(
    rf"(?P<radius>{_FLOAT_PATTERN})\s*m\s*=\s*R0\b",
    re.IGNORECASE,
)
_NEUTRAL_COEF_RE = re.compile(
    rf"^\s*(?P<kind>Alpha|Beta)_(?P<axis>[xyz])\s+"
    rf"(?P<n>\d+)\s+(?P<m>\d+)\s+(?P<value>{_FLOAT_PATTERN})",
    re.IGNORECASE,
)

# The original converter divides stored values by the multiplicative factors
# documented in the source table. The 6th--9th order factors extend the beta
# converter's original 1st--5th order table. SCALE10 has no documented
# conversion and is therefore accepted only when it is zero.
_DAT_XY_FACTORS = np.asarray(
    [
        0.0,
        1.0,
        1.0 / 3.0,
        2.0 / 3.0,
        2.0 / 5.0,
        8.0 / 15.0,
        8.0 / 21.0,
        16.0 / 7.0,
        16.0 / 9.0,
        128.0 / 45.0,
        0.0,
    ],
    dtype=np.float64,
)
_DAT_Z_FACTORS = np.asarray(
    [0.0, 1.0, -2.0, -2.0, -8.0, -8.0, -16.0, -16.0, -128.0, -128.0, 0.0],
    dtype=np.float64,
)


@runtime_checkable
class CoefficientAccessor(Protocol):
    """Source of a coefficient table that is not a file.

    An object with a ``read_coefficients()`` method returning the table's text
    or bytes, for a table held by a scanner or a service.
    :meth:`GradientCoefficients.from_file` and :class:`Gradunwarp` accept one
    wherever they accept a path; an optional ``name`` attribute labels it.
    """

    def read_coefficients(self) -> CoefficientPayload:
        """The contents of the coefficient table, as text or UTF-8 bytes."""


CoefficientSource: TypeAlias = (
    CoefficientPayload | str | Path | io.TextIOBase | io.BufferedIOBase | CoefficientAccessor
)


def _source_text(source: CoefficientSource) -> tuple[str, str | None]:
    """Resolve a coefficient source to text and a non-sensitive source label."""
    if isinstance(source, CoefficientAccessor):
        payload = source.read_coefficients()
        label = getattr(source, "name", type(source).__name__)
    elif hasattr(source, "read"):
        payload = source.read()
        label = getattr(source, "name", type(source).__name__)
    elif isinstance(source, Path):
        payload = source.read_bytes()
        label = source.name
    elif isinstance(source, str):
        if "\n" not in source and "\r" not in source:
            candidate = Path(source)
            if candidate.is_file():
                payload = candidate.read_bytes()
                label = candidate.name
            else:
                payload = source
                label = None
        else:
            payload = source
            label = None
    else:
        payload = source
        label = None

    if isinstance(payload, (bytes, bytearray, np.bytes_)):
        text = bytes(payload).decode("utf-8")
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError(
            f"Coefficient accessors must return str or bytes, not {type(payload).__name__}."
        )
    return text, label


@dataclass(frozen=True)
class GradientCoefficients:
    r"""Spherical-harmonic coefficients of a gradient coil's departure from linearity.

    ``alpha[axis, n, m]`` and ``beta[axis, n, m]`` are the cosine and sine
    coefficients of order ``n`` and degree ``m`` for the scanner X, Y and Z
    gradients, shape ``(3, order + 1, order + 1)``.  The arrays are read-only
    and excluded from ``repr``, so that a table is not written to a log.

    Parameters
    ----------
    basis : {"unnormalized", "normalized"}
        ``"unnormalized"``: GE ``.dat`` solid harmonics of the position in
        centimetres.  ``"normalized"``: Siemens ``.grad`` harmonics,
        normalized as HCP ``gradunwarp`` normalizes them, of the position
        relative to ``reference_radius_mm``.
    alpha, beta : array_like
        Cosine and sine coefficients.
    reference_radius_mm : float, default=None
        Radius ``R0`` of a normalized expansion, in millimetres.
    source_name : str, default=None
        Label of the table's source, such as a file name.

    Examples
    --------
    >>> table = "0.25 m = R0\n  1 A( 1, 0)  1.0  x\n  2 A( 3, 0)  -0.025  x\n"
    >>> coefficients = GradientCoefficients.from_file(table)
    >>> coefficients.reference_radius_mm, coefficients.max_order
    (250.0, 3)
    """

    basis: HarmonicBasis
    alpha: np.ndarray = field(repr=False)
    beta: np.ndarray = field(repr=False)
    reference_radius_mm: float | None = None
    source_name: str | None = None

    def __post_init__(self) -> None:
        alpha = np.array(self.alpha, dtype=np.float64, copy=True)
        beta = np.array(self.beta, dtype=np.float64, copy=True)
        if alpha.ndim != 3 or alpha.shape[0] != 3 or alpha.shape != beta.shape:
            raise ValueError("alpha and beta must have matching shape (3, order + 1, order + 1).")
        if alpha.shape[1] != alpha.shape[2]:
            raise ValueError("Harmonic coefficient matrices must be square.")
        if self.basis not in ("unnormalized", "normalized"):
            raise ValueError("basis must be 'unnormalized' or 'normalized'.")
        if self.basis == "normalized" and (
            self.reference_radius_mm is None or self.reference_radius_mm <= 0
        ):
            raise ValueError("Normalized coefficients require a positive reference_radius_mm.")
        alpha.setflags(write=False)
        beta.setflags(write=False)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def max_order(self) -> int:
        """Largest represented spherical-harmonic order."""
        return self.alpha.shape[1] - 1

    @property
    def mapping_sign(self) -> float:
        """Sign of the evaluated field in the map from corrected to acquired positions."""
        return -1.0 if self.basis == "unnormalized" else 1.0

    @classmethod
    def from_file(
        cls,
        source: CoefficientSource,
        *,
        coefficient_format: CoefficientFormat = "auto",
        reference_radius_mm: float | None = None,
    ) -> GradientCoefficients:
        """Parse a coefficient table.

        Parameters
        ----------
        source : str, bytes, pathlib.Path, file object or CoefficientAccessor
            A path, the table's own text or bytes (a string containing a line
            break, or naming no file, is taken as text), an open file, or an
            accessor.
        coefficient_format : {"auto", "dat", "grad", "coef"}, default='auto'
            GE ``.dat`` (``SCALE{X,Y,Z}{n}`` entries), Siemens ``.grad``
            (``A(n, m)`` and ``B(n, m)`` entries and ``R0``), or a neutral
            ``Alpha_x n m value`` table.  ``"auto"`` reads ``.dat`` when a
            ``SCALE`` entry is present and ``.grad`` or ``.coef`` otherwise.
        reference_radius_mm : float, default=None
            ``R0`` for a normalized table that does not state it.

        Returns
        -------
        GradientCoefficients

        Raises
        ------
        ValueError
            If the table has no entries, a ``.dat`` table has a nonzero
            ``DELTA``, an unsupported ``GRADWARPTYPE`` or a nonzero tenth
            order, or a normalized table has no reference radius.
        """
        text, label = _source_text(source)
        if coefficient_format == "auto":
            coefficient_format = "dat" if _DAT_SCALE_RE.search(text) else "grad"
        if coefficient_format == "dat":
            return cls._from_dat_text(text, label)
        if coefficient_format in ("grad", "coef"):
            return cls._from_normalized_text(text, label, reference_radius_mm)
        raise ValueError("coefficient_format must be 'auto', 'dat', 'grad', or 'coef'.")

    @classmethod
    def _from_dat_text(
        cls,
        text: str,
        label: str | None,
    ) -> GradientCoefficients:
        gradwarp_type: int | None = None
        delta = 0.0
        scales: dict[tuple[int, int], float] = {}

        for line in text.splitlines():
            fields = line.split("#", 1)[0].strip().split()
            if not fields:
                continue
            key = fields[0].upper()
            if key == "GRADWARPTYPE" and len(fields) >= 2:
                gradwarp_type = int(fields[1])
                continue
            if key == "DELTA" and len(fields) >= 2:
                delta = float(fields[1].replace("D", "E").replace("d", "e"))
                continue
            match = _DAT_SCALE_RE.match(line)
            if match:
                axis = "XYZ".index(match.group("axis").upper())
                order = int(match.group("order"))
                value = float(match.group("value").replace("D", "E").replace("d", "e"))
                scales[(axis, order)] = value

        if gradwarp_type not in (None, 1):
            raise ValueError(f"Only GRADWARPTYPE 1 is supported, got {gradwarp_type}.")
        if not scales:
            raise ValueError("No SCALE{X,Y,Z}{N} entries found in .dat payload.")
        if not math.isclose(delta, 0.0, abs_tol=1e-5):
            raise ValueError(f".dat DELTA must be zero, got {delta}.")

        max_order = max(order for _, order in scales)
        if max_order > 10:
            raise ValueError(f".dat harmonic order {max_order} is not supported.")
        for axis in range(3):
            value = scales.get((axis, 10), 0.0)
            if not math.isclose(value, 0.0, abs_tol=0.0):
                raise ValueError(
                    "Non-zero SCALE{X,Y,Z}10 cannot be converted because its "
                    "neutral normalization is undocumented."
                )

        represented_order = min(max(max_order, 1), 9)
        alpha = np.zeros((3, represented_order + 1, represented_order + 1))
        beta = np.zeros_like(alpha)
        for order in range(1, represented_order + 1):
            alpha[0, order, 1] = scales.get((0, order), 0.0) * _DAT_XY_FACTORS[order]
            beta[1, order, 1] = scales.get((1, order), 0.0) * _DAT_XY_FACTORS[order]
            alpha[2, order, 0] = scales.get((2, order), 0.0) * _DAT_Z_FACTORS[order]
        return cls("unnormalized", alpha, beta, source_name=label)

    @classmethod
    def _from_normalized_text(
        cls,
        text: str,
        label: str | None,
        reference_radius_mm: float | None = None,
    ) -> GradientCoefficients:
        entries: list[tuple[str, int, int, int, float]] = []
        parsed_radius = reference_radius_mm

        for line in text.splitlines():
            radius_match = _SIEMENS_R0_RE.search(line)
            if radius_match:
                parsed_radius = 1000.0 * float(
                    radius_match.group("radius").replace("D", "E").replace("d", "e")
                )
            grad_match = _SIEMENS_GRAD_RE.match(line)
            if grad_match:
                entries.append(
                    (
                        grad_match.group("kind").upper(),
                        "xyz".index(grad_match.group("axis").lower()),
                        int(grad_match.group("n")),
                        int(grad_match.group("m")),
                        float(grad_match.group("value").replace("D", "E").replace("d", "e")),
                    )
                )
                continue
            neutral_match = _NEUTRAL_COEF_RE.match(line)
            if neutral_match:
                entries.append(
                    (
                        "A" if neutral_match.group("kind").lower() == "alpha" else "B",
                        "xyz".index(neutral_match.group("axis").lower()),
                        int(neutral_match.group("n")),
                        int(neutral_match.group("m")),
                        float(neutral_match.group("value").replace("D", "E").replace("d", "e")),
                    )
                )

        if not entries:
            raise ValueError("No spherical-harmonic entries found.")
        if parsed_radius is None or parsed_radius <= 0:
            raise ValueError(
                "Neutral .coef data requires reference_radius_mm; .grad data "
                "normally contains '<R0> m = R0'."
            )

        max_order = max(n for _, _, n, _, _ in entries)
        alpha = np.zeros((3, max_order + 1, max_order + 1))
        beta = np.zeros_like(alpha)
        for kind, axis, order, degree, value in entries:
            if degree > order:
                raise ValueError(f"Invalid harmonic degree m={degree} for n={order}.")
            (alpha if kind == "A" else beta)[axis, order, degree] = value
        return cls(
            "normalized",
            alpha,
            beta,
            reference_radius_mm=float(parsed_radius),
            source_name=label,
        )


@dataclass(frozen=True)
class _ImageGeometry:
    """Physical geometry of an image's trailing two or three array axes.

    ``direction[:, axis]`` is the unit scanner-coordinate vector followed when
    the corresponding NumPy image index increases. ``fov_mm`` is independent of
    matrix size, so a zero-filled reconstruction uses its *actual* output
    ``shape`` with the unchanged physical field of view.
    """

    shape: tuple[int, ...]
    fov_mm: tuple[float, ...]
    direction: np.ndarray = field(repr=False)
    center_mm: np.ndarray

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        fov = tuple(float(value) for value in self.fov_mm)
        direction = np.array(self.direction, dtype=np.float64, copy=True)
        center = np.array(self.center_mm, dtype=np.float64, copy=True)
        if len(shape) not in (2, 3):
            raise ValueError("Geometry supports two or three spatial axes.")
        if len(fov) != len(shape) or any(value <= 0 for value in fov):
            raise ValueError("fov_mm must contain one positive value per axis.")
        if any(value <= 0 for value in shape):
            raise ValueError("shape entries must be positive.")
        if direction.shape != (3, len(shape)):
            raise ValueError(f"direction must have shape (3, {len(shape)}), got {direction.shape}.")
        if center.shape != (3,):
            raise ValueError("center_mm must have shape (3,).")
        # Summed by numpy rather than BLAS: on macOS, matmul on these finite
        # columns raised a floating-point "invalid value" warning.
        gram = np.einsum("ij,ik->jk", direction, direction)
        if not np.allclose(gram, np.eye(len(shape)), atol=1e-6):
            raise ValueError("direction columns must be orthonormal.")
        direction.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "fov_mm", fov)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "center_mm", center)

    @property
    def ndim(self) -> int:
        """Number of spatial image axes."""
        return len(self.shape)

    @property
    def voxel_size_mm(self) -> np.ndarray:
        """Physical voxel spacing based on the actual matrix size."""
        return np.asarray(self.fov_mm) / np.asarray(self.shape)

    def indices_to_scanner(self, indices: np.ndarray) -> np.ndarray:
        """Map array indices to scanner coordinates in millimetres."""
        indices = np.asarray(indices, dtype=np.float64)
        if indices.shape[-1] != self.ndim:
            raise ValueError(f"Expected indices[..., {self.ndim}], got {indices.shape}.")
        centered = indices - (np.asarray(self.shape) - 1.0) / 2.0
        offsets = centered * self.voxel_size_mm
        return self.center_mm + offsets @ self.direction.T

    def scanner_to_indices(self, coordinates_mm: np.ndarray) -> np.ndarray:
        """Map scanner coordinates in millimetres to floating image indices."""
        coordinates = np.asarray(coordinates_mm, dtype=np.float64)
        if coordinates.shape[-1] != 3:
            raise ValueError("Scanner coordinates must have a final size-3 axis.")
        offsets = (coordinates - self.center_mm) @ self.direction
        return offsets / self.voxel_size_mm + (np.asarray(self.shape) - 1.0) / 2.0

    def scanner_grid(self) -> np.ndarray:
        """Return the physical coordinate of every voxel centre."""
        indices = np.moveaxis(np.indices(self.shape, dtype=np.float64), 0, -1)
        return self.indices_to_scanner(indices)

    @classmethod
    def from_affine(
        cls,
        affine: np.ndarray,
        shape: tuple[int, ...],
    ) -> _ImageGeometry:
        """Create geometry from a voxel-centre-to-scanner affine."""
        shape = tuple(int(value) for value in shape)
        if len(shape) not in (2, 3):
            raise ValueError("shape must have two or three entries.")
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError("affine must have shape (4, 4).")
        basis = affine[:3, : len(shape)]
        spacing = np.linalg.norm(basis, axis=0)
        if np.any(spacing <= 0):
            raise ValueError("affine contains a zero-length spatial axis.")
        direction = basis / spacing
        center_index = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
        center_h = np.concatenate((center_index, np.zeros(3 - len(shape)), [1.0]))
        center = (affine @ center_h)[:3]
        return cls(
            shape,
            tuple((spacing * np.asarray(shape)).tolist()),
            direction,
            center,
        )


def _associated_legendre(
    order: int,
    degree: int,
    cosine: np.ndarray,
) -> np.ndarray:
    """Unnormalised associated Legendre polynomial with Condon-Shortley phase."""
    if degree < 0 or degree > order:
        raise ValueError(f"Invalid associated Legendre P_{order}^{degree}.")
    p_mm = np.ones_like(cosine)
    if degree:
        root = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
        factor = 1.0
        for _ in range(1, degree + 1):
            p_mm *= -factor * root
            factor += 2.0
    if order == degree:
        return p_mm
    p_m1m = (2 * degree + 1) * cosine * p_mm
    if order == degree + 1:
        return p_m1m
    previous, current = p_mm, p_m1m
    for n in range(degree + 2, order + 1):
        following = ((2 * n - 1) * cosine * current - (n + degree - 1) * previous) / (n - degree)
        previous, current = current, following
    return current


def _evaluate_harmonics(
    coefficients: GradientCoefficients,
    coordinates_mm: np.ndarray,
) -> np.ndarray:
    """Evaluate displacement/error fields directly on scanner coordinates."""
    coordinates = np.asarray(coordinates_mm, dtype=np.float64)
    if coefficients.basis == "unnormalized" and _has_dat_structure(coefficients):
        return _evaluate_dat_harmonics(coefficients, coordinates)

    x, y, z = np.moveaxis(coordinates, -1, 0)
    radius_mm = np.sqrt(x * x + y * y + z * z)
    cosine = np.divide(
        z,
        radius_mm,
        out=np.ones_like(radius_mm),
        where=radius_mm != 0,
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    azimuth = np.arctan2(y, x)
    result = np.zeros(coordinates.shape, dtype=np.float64)

    if coefficients.basis == "unnormalized":
        radial = radius_mm / 10.0  # Unnormalised .dat coefficients use centimetres.
        output_scale = 10.0  # centimetres back to millimetres.
    else:
        reference = coefficients.reference_radius_mm
        if reference is None:
            raise ValueError(
                "a normalized basis is stated over a reference radius, and "
                "these coefficients carry none"
            )
        radial = radius_mm / reference
        output_scale = reference

    for n in range(coefficients.max_order + 1):
        radial_term = np.power(radial, n)
        for m in range(n + 1):
            alpha = coefficients.alpha[:, n, m]
            beta = coefficients.beta[:, n, m]
            if not np.any(alpha) and not np.any(beta):
                continue
            legendre = _associated_legendre(n, m, cosine)
            if coefficients.basis == "normalized" and m > 0:
                normalization = ((-1.0) ** m) * math.sqrt(
                    (2 * n + 1) * math.factorial(n - m) / (2.0 * math.factorial(n + m))
                )
                legendre = legendre * normalization
            angular = (
                alpha.reshape((1,) * radius_mm.ndim + (3,)) * np.cos(m * azimuth)[..., None]
                + beta.reshape((1,) * radius_mm.ndim + (3,)) * np.sin(m * azimuth)[..., None]
            )
            result += output_scale * radial_term[..., None] * legendre[..., None] * angular
    return result


def _has_dat_structure(coefficients: GradientCoefficients) -> bool:
    """Return whether only the three harmonic families stored by .dat are set."""
    alpha = coefficients.alpha.copy()
    beta = coefficients.beta.copy()
    for order in range(1, coefficients.max_order + 1):
        alpha[0, order, 1] = 0.0
        beta[1, order, 1] = 0.0
        alpha[2, order, 0] = 0.0
    return not (np.any(alpha) or np.any(beta))


def _evaluate_dat_harmonics(
    coefficients: GradientCoefficients,
    coordinates_mm: np.ndarray,
) -> np.ndarray:
    """Evaluate .dat solid harmonics using Cartesian recurrences.

    This avoids spherical coordinates, trigonometric functions, and repeated
    associated-Legendre evaluation. It is algebraically equivalent for the
    three coefficient families represented by the source format.
    """
    x, y, z = np.moveaxis(coordinates_mm / 10.0, -1, 0)
    radius_squared = x * x + y * y + z * z
    result = np.zeros_like(coordinates_mm)

    zonal_previous_2 = np.ones_like(x)
    zonal_previous_1 = z
    order_one_previous_2 = np.zeros_like(x)
    order_one_previous_1 = np.ones_like(x)

    for order in range(1, coefficients.max_order + 1):
        if order == 1:
            zonal = zonal_previous_1
            order_one = order_one_previous_1
        else:
            zonal = (
                (2 * order - 1) * z * zonal_previous_1
                - (order - 1) * radius_squared * zonal_previous_2
            ) / order
            order_one = (
                (2 * order - 1) * z * order_one_previous_1
                - order * radius_squared * order_one_previous_2
            ) / (order - 1)
            zonal_previous_2, zonal_previous_1 = zonal_previous_1, zonal
            order_one_previous_2, order_one_previous_1 = (
                order_one_previous_1,
                order_one,
            )

        result[..., 0] -= coefficients.alpha[0, order, 1] * x * order_one
        result[..., 1] -= coefficients.beta[1, order, 1] * y * order_one
        result[..., 2] += coefficients.alpha[2, order, 0] * zonal
    return 10.0 * result


_COEFFICIENT_PARAMETER_NAMES = (
    "gradientcoefficients",
    "gradient_coefficients",
    "gradunwarp_coefficients",
    "coeff_dat",
    "coefficients",
)


def _coefficients_from_header(header: Any) -> str | None:
    """Find a coil table among an MRD header's user parameters."""
    parameters = getattr(getattr(header, "userParameters", None), "userParameterString", None)
    for parameter in parameters or ():
        name = str(getattr(parameter, "name", "")).strip().lower()
        if name in _COEFFICIENT_PARAMETER_NAMES or (
            "coef" in name and ("grad" in name or "coil" in name)
        ):
            value = getattr(parameter, "value", None)
            if value:
                return str(value)
    return None


class Gradunwarp:
    r"""Gradient nonlinearity correction of images from a coil's coefficient table.

    The coil's spherical-harmonic field is evaluated at the scanner position
    :math:`r` of every voxel of the target grid, giving the position
    :math:`r + d(r)` at which that voxel was acquired, and the image is
    resampled there by cubic B-spline interpolation (SimpleITK), optionally
    multiplied by the Jacobian determinant :math:`|\det(I + \nabla d)|`.
    Geometry is in scanner coordinates, in millimetres; voxel centres are at
    ``center_mm + ((index - (n - 1) / 2) * fov_mm / n) @ orientation.T``.

    Parameters
    ----------
    coefficients : str, bytes, pathlib.Path, CoefficientAccessor or GradientCoefficients
        The coil's table, as :meth:`GradientCoefficients.from_file` reads it,
        or already parsed.
    shape : sequence of int
        Matrix of the acquired image, ``(z, y, x)`` or ``(y, x)``.
    fov_mm : sequence of float
        Field of view per axis of ``shape``, in millimetres; independent of
        the matrix, so a zero-filled reconstruction gives its own ``shape`` and
        the acquired field of view.
    orientation : numpy.ndarray, default=None
        ``(3, ndim)`` direction cosines: column ``i`` is the unit scanner
        vector along which array axis ``i`` increases.  ``None`` is the
        identity's first ``ndim`` columns.
    center_mm : sequence of float, default=(0.0, 0.0, 0.0)
        Scanner position of the image centre.
    target_shape : sequence of int, default=None
        Matrix of the grid corrected onto; ``shape`` if ``None``.  The target
        grid shares ``orientation`` and ``center_mm``, and correction and
        reslicing are one interpolation.
    target_fov_mm : sequence of float, default=None
        Field of view of the target grid; ``fov_mm`` if ``None``.
    jacobian : bool, default=True
        Multiply by the Jacobian determinant of the mapping, which conserves
        the signal of a voxel whose volume the nonlinearity changes.
    coefficient_format : {"auto", "dat", "grad", "coef"}, default='auto'
        As in :meth:`GradientCoefficients.from_file`.
    reference_radius_mm : float, default=None
        As in :meth:`GradientCoefficients.from_file`.

    Attributes
    ----------
    coefficients : GradientCoefficients
        The parsed table.

    Examples
    --------
    >>> correct = Gradunwarp("coil.grad", shape=(8, 16, 16), fov_mm=(80.0, 240.0, 240.0))
    >>> corrected = correct(volume)  # (..., 8, 16, 16)
    """

    def __init__(
        self,
        coefficients: CoefficientSource | GradientCoefficients,
        shape: Sequence[int],
        fov_mm: Sequence[float],
        orientation: np.ndarray | None = None,
        center_mm: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        target_shape: Sequence[int] | None = None,
        target_fov_mm: Sequence[float] | None = None,
        jacobian: bool = True,
        coefficient_format: CoefficientFormat = "auto",
        reference_radius_mm: float | None = None,
    ) -> None:
        if not isinstance(coefficients, GradientCoefficients):
            coefficients = GradientCoefficients.from_file(
                coefficients,
                coefficient_format=coefficient_format,
                reference_radius_mm=reference_radius_mm,
            )
        self.coefficients = coefficients

        shape = tuple(int(value) for value in shape)
        if orientation is None:
            orientation = np.eye(3)[:, : len(shape)]
        acquired = _ImageGeometry(shape, tuple(fov_mm), orientation, center_mm)
        target = _ImageGeometry(
            shape if target_shape is None else tuple(int(v) for v in target_shape),
            tuple(fov_mm if target_fov_mm is None else target_fov_mm),
            orientation,
            center_mm,
        )
        if acquired.ndim != target.ndim:
            raise ValueError("The acquired and target grids must have the same rank.")
        self._acquired = acquired
        self._target = target
        self.jacobian = bool(jacobian)
        self._source_indices: np.ndarray | None = None
        self._jacobian_multiplier: np.ndarray | None = None
        self._simpleitk_cache: tuple[object, object] | None = None

    @classmethod
    def from_mrd(
        cls,
        header: Any,
        acquisition: Any,
        *,
        coefficients: CoefficientSource | GradientCoefficients | None = None,
        **kwargs: Any,
    ) -> Gradunwarp:
        """Correction for the image of one encoding of an MRD stream.

        The header and the acquisition are read by attribute, as the ISMRMRD
        Python objects carry them; no MRD library is imported.

        Parameters
        ----------
        header : object
            MRD XML header.  ``header.encoding[k].reconSpace`` gives the matrix
            and field of view; the table is read from the first entry of
            ``header.userParameters.userParameterString`` named for gradient
            coefficients (``GradientCoefficients`` and similar).
        acquisition : object
            An acquisition or its head: ``encoding_space_ref`` selects the
            encoding, ``slice_dir``, ``phase_dir`` and ``read_dir`` the
            orientation of the ``(z, y, x)`` array axes, ``position`` the
            centre.
        coefficients : str, pathlib.Path, CoefficientAccessor or GradientCoefficients, default=None
            The table, which takes precedence over the header's; text is also
            accepted as bytes.
        **kwargs
            Passed to :class:`Gradunwarp`.

        Returns
        -------
        Gradunwarp

        Raises
        ------
        ValueError
            If the header has no such encoding, or no table is given or found.
        """
        # An MRD Acquisition proxies its head's fields; a bare head does not.
        head = acquisition
        if not hasattr(head, "read_dir") and not hasattr(head, "col_dir"):
            head = getattr(acquisition, "head", acquisition)
        index = int(getattr(head, "encoding_space_ref", 0) or 0)
        encodings = list(getattr(header, "encoding", []) or [])
        if index >= len(encodings):
            raise ValueError(
                f"The acquisition names encoding {index}, and the header has {len(encodings)}."
            )
        recon = encodings[index].reconSpace
        matrix, extent = recon.matrixSize, recon.fieldOfView_mm

        def direction(*names: str) -> np.ndarray:
            for name in names:
                value = getattr(head, name, None)
                if value is not None:
                    vector = np.asarray(value, dtype=np.float64).ravel()
                    if vector.shape == (3,):
                        return vector
            raise ValueError(f"The acquisition is missing {names[0]!r}.")

        # MRD states x along the readout; a reconstructed array is indexed
        # (slice, line, column), which is the reverse.
        shape = (int(matrix.z), int(matrix.y), int(matrix.x))
        fov_mm = (float(extent.z), float(extent.y), float(extent.x))
        orientation = np.stack(
            [
                direction("slice_dir"),
                direction("phase_dir", "line_dir"),
                direction("read_dir", "col_dir"),
            ],
            axis=1,
        )
        center = getattr(head, "position", (0.0, 0.0, 0.0))

        if coefficients is None:
            coefficients = _coefficients_from_header(header)
        if coefficients is None:
            raise ValueError(
                "No gradient coefficients in the header's userParameters; pass coefficients=."
            )
        return cls(
            coefficients,
            shape,
            fov_mm,
            orientation,
            np.asarray(center, dtype=np.float64).ravel(),
            **kwargs,
        )

    @classmethod
    def from_affine(
        cls,
        coefficients: CoefficientSource | GradientCoefficients,
        affine: np.ndarray,
        shape: Sequence[int],
        **kwargs: Any,
    ) -> Gradunwarp:
        """Correction for an image whose voxel centres map to scanner millimetres by ``affine``.

        Parameters
        ----------
        coefficients : str, bytes, pathlib.Path, CoefficientAccessor or GradientCoefficients
            The coil's table.
        affine : numpy.ndarray
            ``(4, 4)`` map from array indices ``(i, j[, k], 1)`` to scanner
            position in millimetres.
        shape : sequence of int
            Matrix of the image.
        **kwargs
            Passed to :class:`Gradunwarp`.

        Returns
        -------
        Gradunwarp
        """
        geometry = _ImageGeometry.from_affine(affine, shape)
        return cls(
            coefficients,
            geometry.shape,
            geometry.fov_mm,
            geometry.direction,
            geometry.center_mm,
            **kwargs,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Matrix size of the corrected image."""
        return self._target.shape

    @property
    def source_grid(self) -> np.ndarray:
        """Floating indices into the acquired image, one per corrected voxel."""
        return self._prepare_grid()

    @property
    def target_grid(self) -> np.ndarray:
        """Scanner coordinates in millimetres of the grid being corrected onto."""
        return self._target.scanner_grid()

    @property
    def jacobian_grid(self) -> np.ndarray:
        """The intensity multiplier, or ones when ``jacobian`` is off."""
        self._prepare_grid()
        if self._jacobian_multiplier is None:
            return np.ones(self._target.shape, dtype=np.float64)
        return self._jacobian_multiplier

    def clear_cache(self) -> None:
        """Drop the cached physical sampling grid and Jacobian."""
        self._source_indices = None
        self._jacobian_multiplier = None
        self._simpleitk_cache = None

    def _prepare_grid(self) -> np.ndarray:
        """Return the sampling grid, computing it the first time it is asked for."""
        if self._source_indices is not None:
            return self._source_indices
        coordinates = self._target.scanner_grid()
        field = _evaluate_harmonics(self.coefficients, coordinates)
        displacement = self.coefficients.mapping_sign * field
        source_coordinates = coordinates + displacement
        source_indices = self._acquired.scanner_to_indices(source_coordinates)
        source_indices.setflags(write=False)
        self._source_indices = source_indices

        if self.jacobian:
            multiplier = self._full_jacobian(coordinates, displacement)
            multiplier.setflags(write=False)
            self._jacobian_multiplier = multiplier
        return source_indices

    def _full_jacobian(
        self,
        coordinates: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray:
        geometry = self._target
        if geometry.ndim == 3:
            return self._simpleitk_jacobian(displacement)

        directional = np.empty((*geometry.shape, 3, 3), dtype=np.float64)
        basis = np.empty((3, 3), dtype=np.float64)

        for axis in range(geometry.ndim):
            basis[:, axis] = geometry.direction[:, axis]
            if geometry.shape[axis] > 1:
                edge_order = 2 if geometry.shape[axis] > 2 else 1
                for component in range(3):
                    directional[..., component, axis] = np.gradient(
                        displacement[..., component],
                        geometry.voxel_size_mm[axis],
                        axis=axis,
                        edge_order=edge_order,
                    )
            else:
                epsilon = max(geometry.voxel_size_mm[axis] * 1e-3, 1e-3)
                offset = geometry.direction[:, axis] * epsilon
                plus = self.coefficients.mapping_sign * _evaluate_harmonics(
                    self.coefficients, coordinates + offset
                )
                minus = self.coefficients.mapping_sign * _evaluate_harmonics(
                    self.coefficients, coordinates - offset
                )
                directional[..., :, axis] = (plus - minus) / (2.0 * epsilon)

        if geometry.ndim == 2:
            normal = np.cross(geometry.direction[:, 0], geometry.direction[:, 1])
            normal /= np.linalg.norm(normal)
            basis[:, 2] = normal
            epsilon = max(float(np.min(geometry.voxel_size_mm)) * 1e-3, 1e-3)
            plus = self.coefficients.mapping_sign * _evaluate_harmonics(
                self.coefficients, coordinates + normal * epsilon
            )
            minus = self.coefficients.mapping_sign * _evaluate_harmonics(
                self.coefficients, coordinates - normal * epsilon
            )
            directional[..., :, 2] = (plus - minus) / (2.0 * epsilon)

        derivative_scanner = np.einsum(
            "...ob,sb->...os",
            directional,
            basis,
        )
        mapping_jacobian = derivative_scanner + np.eye(3)
        return np.abs(np.linalg.det(mapping_jacobian))

    def _simpleitk_jacobian(self, displacement: np.ndarray) -> np.ndarray:
        """Compute a 3D physical Jacobian with SimpleITK's compiled filter."""
        sitk = _simpleitk()

        geometry = self._target
        displacement_in_image_basis = displacement @ geometry.direction
        field = sitk.GetImageFromArray(
            np.ascontiguousarray(displacement_in_image_basis[..., ::-1]),
            isVector=True,
        )
        field.SetSpacing(tuple(geometry.voxel_size_mm[::-1]))
        determinant = sitk.DisplacementFieldJacobianDeterminantFilter().Execute(field)
        return np.abs(sitk.GetArrayFromImage(determinant))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Correct ``image`` onto the target grid.

        Parameters
        ----------
        image : torch.Tensor
            Real or complex image ``(..., *shape)``; the real and imaginary
            parts are resampled separately.

        Returns
        -------
        torch.Tensor
            ``(..., *target_shape)``, with the dtype and device of ``image``
            (``float32`` for an integer image).
        """
        self._prepare_grid()
        tensor = torch.as_tensor(image)
        dtype = tensor.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.float32
        result = self._call_simpleitk(tensor.detach().cpu().numpy())
        return torch.as_tensor(result, device=tensor.device).to(dtype)

    def _validate_shape(self, shape: tuple[int, ...]) -> None:
        ndim = self._acquired.ndim
        if len(shape) < ndim or tuple(shape[-ndim:]) != self._acquired.shape:
            raise ValueError(
                "The image's trailing shape must match the acquired matrix "
                f"{self._acquired.shape}, got {shape}."
            )

    def _call_simpleitk(self, image: np.ndarray) -> np.ndarray:
        sitk = _simpleitk()

        array = np.asarray(image)
        self._validate_shape(array.shape)
        if not (np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array)):
            array = array.astype(np.float32)
        ndim = self._acquired.ndim
        leading_shape = array.shape[:-ndim]
        batch = math.prod(leading_shape) if leading_shape else 1
        values = array.reshape((batch, *self._acquired.shape))

        if self._simpleitk_cache is None:
            output_indices = np.moveaxis(
                np.indices(self._target.shape, dtype=np.float64),
                0,
                -1,
            )
            displacement = self._source_indices - output_indices
            field = sitk.GetImageFromArray(
                np.ascontiguousarray(displacement[..., ::-1]),
                isVector=True,
            )
            transform = sitk.DisplacementFieldTransform(field)
            reference = sitk.Image(
                list(reversed(self._target.shape)),
                sitk.sitkFloat32,
            )
            self._simpleitk_cache = (transform, reference)
        transform, reference = self._simpleitk_cache

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(0.0)

        def sample(value: np.ndarray) -> np.ndarray:
            source = sitk.GetImageFromArray(np.ascontiguousarray(value))
            return sitk.GetArrayFromImage(resampler.Execute(source))

        output = []
        for value in values:
            if np.iscomplexobj(value):
                output.append(sample(value.real) + 1j * sample(value.imag))
            else:
                output.append(sample(value))
        result = np.stack(output).reshape(leading_shape + self._target.shape)
        if self._jacobian_multiplier is not None:
            result *= self._jacobian_multiplier.astype(result.dtype, copy=False)
        return result
