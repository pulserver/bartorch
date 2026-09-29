"""Gradient nonlinearity correction from a coil's spherical-harmonic table."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bartorch.tools import GradientCoefficients, Gradunwarp
from bartorch.tools._correct._gradunwarp import _evaluate_harmonics

needs_simpleitk = pytest.mark.skipif(
    importlib.util.find_spec("SimpleITK") is None,
    reason="resampling needs SimpleITK, the correct extra",
)


class StubAccessor:
    """A coefficient source that is not a file, as an integration would be.

    The scanner's own accessor reads an MRD header and lives with the server
    that speaks MRD. What belongs here is only that ``from_file`` takes one at
    all, which is what the protocol promises.
    """

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read_coefficients(self) -> str:
        return self._payload


def _zero_coefficients(order: int = 1) -> GradientCoefficients:
    values = np.zeros((3, order + 1, order + 1))
    return GradientCoefficients("unnormalized", values, values)


def _dat_payload(*, scale10: float = 0.0) -> str:
    lines = ["GRADWARPTYPE 1"]
    for axis, multiplier in zip("XYZ", (1.0, 2.0, 3.0), strict=True):
        for order in range(1, 11):
            value = multiplier * order if order < 10 else scale10
            lines.append(f"SCALE{axis}{order} {value}")
    lines.append("DELTA 0.0")
    return "\n".join(lines)


def test_dat_conversion_extends_beta_converter_to_ninth_order() -> None:
    coefficients = GradientCoefficients.from_file(_dat_payload())

    assert coefficients.max_order == 9
    assert coefficients.alpha[0, 3, 1] == pytest.approx(3.0 * 2.0 / 3.0)
    assert coefficients.beta[1, 7, 1] == pytest.approx((2.0 * 7.0) * 16.0 / 7.0)
    assert coefficients.alpha[2, 9, 0] == pytest.approx((3.0 * 9.0) * -128.0)
    assert coefficients.basis == "unnormalized"
    assert "alpha" not in repr(coefficients)
    assert "beta" not in repr(coefficients)


def test_dat_conversion_rejects_undocumented_nonzero_tenth_order() -> None:
    with pytest.raises(ValueError, match=r"SCALE.*10"):
        GradientCoefficients.from_file(_dat_payload(scale10=1e-12))


def test_coefficient_path_and_serialized_text_are_equivalent(tmp_path) -> None:
    path = tmp_path / "coefficients.dat"
    path.write_text(_dat_payload())

    from_path = GradientCoefficients.from_file(path)
    from_text = GradientCoefficients.from_file(_dat_payload())

    np.testing.assert_array_equal(from_path.alpha, from_text.alpha)
    np.testing.assert_array_equal(from_path.beta, from_text.beta)


def test_normalized_coefficient_parser_reads_radius_and_general_degrees() -> None:
    payload = """
    0.25 m = R0, lnorm = 4
      1 A( 3, 0) -0.01 z
    101 A( 3, 1)  0.02 x
    201 B( 5, 3) -0.03 y
    """

    coefficients = GradientCoefficients.from_file(payload)

    assert coefficients.reference_radius_mm == 250.0
    assert coefficients.basis == "normalized"
    assert coefficients.alpha[2, 3, 0] == -0.01
    assert coefficients.alpha[0, 3, 1] == 0.02
    assert coefficients.beta[1, 5, 3] == -0.03


def _mrd_header(table: str | None = "", encodings: int = 1):
    """An MRD XML header, with as much of it as this correction reads."""
    user = SimpleNamespace(
        userParameterString=[SimpleNamespace(name="GradientCoefficients", value=table)]
        if table is not None
        else []
    )
    encoding = [
        SimpleNamespace(
            reconSpace=SimpleNamespace(
                matrixSize=SimpleNamespace(x=16 + index, y=12, z=4),
                fieldOfView_mm=SimpleNamespace(x=240.0, y=180.0, z=60.0),
            )
        )
        for index in range(encodings)
    ]
    return SimpleNamespace(encoding=encoding, userParameters=user)


def _mrd_acquisition(encoding: int = 0):
    return SimpleNamespace(
        encoding_space_ref=encoding,
        read_dir=(-1.0, 0.0, 0.0),
        phase_dir=(0.0, 1.0, 0.0),
        slice_dir=(0.0, 0.0, 1.0),
        position=(10.0, -20.0, 4.0),
    )


def test_from_mrd_reads_the_recon_space_of_the_encoding_the_acquisition_names():
    """The matrix and field of view come from the encoding, not the acquisition."""
    correct = Gradunwarp.from_mrd(
        _mrd_header(_dat_payload(), encodings=3), _mrd_acquisition(encoding=2)
    )

    # MRD states x along the readout; the array is indexed (slice, line, column).
    assert correct.shape == (4, 12, 18)
    np.testing.assert_allclose(correct.target_grid.shape, (4, 12, 18, 3))


def test_from_mrd_takes_the_orientation_and_centre_from_the_acquisition() -> None:
    correct = Gradunwarp.from_mrd(_mrd_header(_dat_payload()), _mrd_acquisition())

    # Columns follow (slice, line, column), so read_dir is the last of them.
    np.testing.assert_allclose(
        correct._acquired.direction,
        np.asarray(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))),
    )
    np.testing.assert_allclose(correct._acquired.center_mm, (10.0, -20.0, 4.0))


def test_from_mrd_reads_the_coil_table_out_of_the_user_parameters() -> None:
    correct = Gradunwarp.from_mrd(_mrd_header(_dat_payload()), _mrd_acquisition())

    assert correct.coefficients.max_order == 9


def test_from_mrd_accepts_a_table_a_stream_does_not_carry() -> None:
    """A site whose coefficients travel separately passes them in."""
    correct = Gradunwarp.from_mrd(
        _mrd_header(table=None),
        _mrd_acquisition(),
        coefficients=StubAccessor(_dat_payload()),
    )

    assert correct.coefficients.max_order == 9


def test_from_mrd_refuses_a_stream_with_no_table_anywhere() -> None:
    with pytest.raises(ValueError, match="No gradient coefficients"):
        Gradunwarp.from_mrd(_mrd_header(table=None), _mrd_acquisition())


def test_from_mrd_refuses_an_encoding_the_header_does_not_have() -> None:
    with pytest.raises(ValueError, match="names encoding 2"):
        Gradunwarp.from_mrd(_mrd_header(_dat_payload()), _mrd_acquisition(encoding=2))


def test_affine_geometry_preserves_oblique_physical_grid() -> None:
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag((1.2, 1.5, 2.0))
    affine[:3, 3] = (4.0, -5.0, 6.0)
    shape = (8, 10, 12)

    correct = Gradunwarp.from_affine(_zero_coefficients(), affine, shape)
    index = (2, 3, 4)

    expected = (affine @ np.append(np.asarray(index, dtype=float), 1.0))[:3]
    np.testing.assert_allclose(correct.target_grid[index], expected, atol=1e-10)
    np.testing.assert_allclose(correct._acquired.scanner_to_indices(expected), index)
    np.testing.assert_allclose(correct._acquired.fov_mm, (9.6, 15.0, 24.0))


@needs_simpleitk
def test_zero_coefficients_are_identity_for_real_and_complex_batches() -> None:
    angle = np.deg2rad(20.0)
    direction = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
            (0.0, 0.0),
        )
    )
    correct = Gradunwarp(_zero_coefficients(), (7, 8), (70.0, 80.0), direction)
    rng = np.random.default_rng(4)
    image = torch.as_tensor(rng.normal(size=(2, *correct.shape)).astype(np.float32))
    complex_image = image + 1j * image.flip(0)

    torch.testing.assert_close(correct(image), image, atol=2e-5, rtol=0)
    torch.testing.assert_close(correct(complex_image), complex_image, atol=2e-5, rtol=0)
    np.testing.assert_allclose(correct.jacobian_grid, 1.0)


@needs_simpleitk
def test_the_corrected_image_keeps_the_dtype_and_device_of_the_input(device) -> None:
    direction = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    correct = Gradunwarp(_zero_coefficients(), (7, 8), (70.0, 80.0), direction)
    image = torch.ones(3, 7, 8, dtype=torch.complex128, device=device)
    result = correct(image)
    assert result.dtype == image.dtype
    assert result.device == image.device


@needs_simpleitk
def test_a_target_grid_corrects_and_reslices_in_one_step() -> None:
    direction = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    correct = Gradunwarp(
        _zero_coefficients(),
        (9, 9),
        (90.0, 90.0),
        direction,
        target_shape=(17, 17),
        target_fov_mm=(80.0, 80.0),
        jacobian=False,
    )
    acquired = Gradunwarp(_zero_coefficients(), (9, 9), (90.0, 90.0), direction)
    physical = acquired.target_grid
    image = (2.0 * physical[..., 0] - 3.0 * physical[..., 1] + 7.0).astype(np.float32)

    result = correct(torch.as_tensor(image)).numpy()
    expected = acquired._acquired.scanner_to_indices(correct.target_grid)

    assert result.shape == (17, 17)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(correct.source_grid, expected, atol=1e-12)


def test_full_3d_jacobian_is_used_for_a_2d_plane() -> None:
    alpha = np.zeros((3, 2, 2))
    beta = np.zeros_like(alpha)
    alpha[0, 1, 1] = 0.01
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    correct = Gradunwarp(
        coefficients,
        (11, 12),
        (110.0, 120.0),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
    )

    # P_1^1(cos(theta)) cos(phi) * r = -x. The output-to-source
    # convention therefore yields source_x = 1.01*x and determinant 1.01.
    np.testing.assert_allclose(correct.jacobian_grid, 1.01, atol=2e-8)


def test_cartesian_dat_recurrence_matches_second_order_solid_harmonics() -> None:
    alpha = np.zeros((3, 3, 3))
    beta = np.zeros_like(alpha)
    alpha[0, 2, 1] = 0.2
    beta[1, 2, 1] = -0.3
    alpha[2, 2, 0] = 0.4
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    coordinates = np.asarray(((20.0, -30.0, 40.0), (-10.0, 50.0, -20.0)))
    x, y, z = (coordinates / 10.0).T
    radius_squared = x * x + y * y + z * z
    expected = 10.0 * np.stack(
        (
            0.2 * (-3.0 * x * z),
            -0.3 * (-3.0 * y * z),
            0.4 * (3.0 * z * z - radius_squared) / 2.0,
        ),
        axis=-1,
    )

    np.testing.assert_allclose(
        _evaluate_harmonics(coefficients, coordinates),
        expected,
        atol=1e-12,
    )


@needs_simpleitk
def test_compiled_3d_jacobian_uses_all_three_physical_derivatives() -> None:
    alpha = np.zeros((3, 2, 2))
    beta = np.zeros_like(alpha)
    alpha[0, 1, 1] = 0.01
    beta[1, 1, 1] = 0.02
    alpha[2, 1, 0] = 0.03
    coefficients = GradientCoefficients("unnormalized", alpha, beta)
    angle = np.deg2rad(23.0)
    direction = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    determinant = Gradunwarp(
        coefficients, (9, 10, 11), (90.0, 100.0, 110.0), direction
    ).jacobian_grid

    np.testing.assert_allclose(
        determinant[1:-1, 1:-1, 1:-1],
        1.01 * 1.02 * 0.97,
        atol=1e-12,
    )


def test_the_constructor_accepts_a_coefficient_accessor_directly() -> None:
    correct = Gradunwarp(
        StubAccessor(_dat_payload()),
        (8, 9),
        (80.0, 90.0),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
    )

    assert correct.coefficients.max_order == 9


#: A body-coil table in the console syntax a GE scanner emits, with invented
#: third- and fifth-order values: no vendor table is bundled here.
EMITTED_COIL_DESCRIPTION = (
    "GRADWARPTYPE 1\n"
    + "".join(
        f"SCALE{axis}{order} {value:.9e}\n"
        for axis, third, fifth in (
            ("X", -1.5e-04, -8.0e-08),
            ("Y", -1.25e-04, -9.0e-08),
            ("Z", -1.0e-04, -1.0e-08),
        )
        for order, value in (
            (order, {3: third, 5: fifth}.get(order, 0.0)) for order in range(1, 11)
        )
    )
    + "DELTA 0.000000000e+00\n"
)


def test_a_console_coil_description_parses_with_the_documented_factors():
    coefficients = GradientCoefficients.from_file(EMITTED_COIL_DESCRIPTION)

    assert coefficients.basis == "unnormalized"
    assert coefficients.max_order == 9
    assert coefficients.alpha[0, 3, 1] == pytest.approx(-1.5e-4 * 2.0 / 3.0)
    assert coefficients.beta[1, 3, 1] == pytest.approx(-1.25e-4 * 2.0 / 3.0)
    assert coefficients.alpha[2, 3, 0] == pytest.approx(-1.0e-4 * -2.0)
    assert coefficients.alpha[0, 5, 1] == pytest.approx(-8.0e-8 * 8.0 / 15.0)
    assert coefficients.beta[1, 5, 1] == pytest.approx(-9.0e-8 * 8.0 / 15.0)
    assert coefficients.alpha[2, 5, 0] == pytest.approx(-1.0e-8 * -8.0)


# --- the general spherical-harmonic expansion ------------------------------------


def _single(basis: str, kind: str, axis: int, order: int, degree: int, value: float, **kwargs):
    alpha = np.zeros((3, order + 1, order + 1))
    beta = np.zeros_like(alpha)
    (alpha if kind == "alpha" else beta)[axis, order, degree] = value
    return GradientCoefficients(basis, alpha, beta, **kwargs)


POINTS = np.asarray(
    ((20.0, -30.0, 40.0), (-10.0, 50.0, -20.0), (35.0, 5.0, 0.0), (0.0, 0.0, -45.0))
)


def test_a_normalized_zonal_term_is_the_solid_harmonic_written_out() -> None:
    radius = 250.0
    x, y, z = POINTS.T
    r2 = x * x + y * y + z * z

    linear = _evaluate_harmonics(
        _single("normalized", "alpha", 2, 1, 0, 0.1, reference_radius_mm=radius), POINTS
    )
    cubic = _evaluate_harmonics(
        _single("normalized", "alpha", 0, 3, 0, 0.1, reference_radius_mm=radius), POINTS
    )

    # R0 (r / R0)^n P_n(cos theta): z, and (5 z^3 - 3 z r^2) / (2 R0^2).
    np.testing.assert_allclose(linear, np.stack([0 * z, 0 * z, 0.1 * z], axis=-1), atol=1e-12)
    np.testing.assert_allclose(
        cubic[:, 0], 0.1 * (5 * z**3 - 3 * z * r2) / (2 * radius**2), atol=1e-12
    )
    np.testing.assert_array_equal(cubic[:, 1:], 0.0)


def test_a_normalized_sectoral_term_is_the_solid_harmonic_written_out() -> None:
    """``P_2^2`` with HCP ``gradunwarp``'s normalization, ``sqrt(5 / 48)``, over ``R0``."""
    radius = 200.0
    x, y, _ = POINTS.T
    cosine = _evaluate_harmonics(
        _single("normalized", "alpha", 1, 2, 2, 0.1, reference_radius_mm=radius), POINTS
    )
    sine = _evaluate_harmonics(
        _single("normalized", "beta", 1, 2, 2, 0.1, reference_radius_mm=radius), POINTS
    )
    # r^2 sin^2(theta) cos(2 phi) = x^2 - y^2, and sin(2 phi) gives 2 x y.
    scale = 0.1 * 3.0 * np.sqrt(5.0 / 48.0) / radius
    np.testing.assert_allclose(cosine[:, 1], scale * (x * x - y * y), atol=1e-12)
    np.testing.assert_allclose(sine[:, 1], scale * 2 * x * y, atol=1e-12)


@pytest.mark.parametrize(("order", "degree"), [(1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 0)])
def test_the_associated_legendre_functions_are_scipys(order, degree) -> None:
    from scipy.special import lpmv

    from bartorch.tools._correct._gradunwarp import _associated_legendre

    cosine = np.linspace(-1.0, 1.0, 41)
    np.testing.assert_allclose(
        _associated_legendre(order, degree, cosine),
        lpmv(degree, order, cosine),
        rtol=1e-10,
        atol=1e-10,
    )


def test_a_legendre_degree_above_the_order_is_refused() -> None:
    from bartorch.tools._correct._gradunwarp import _associated_legendre

    with pytest.raises(ValueError, match="Invalid associated Legendre"):
        _associated_legendre(2, 3, np.zeros(3))


def test_an_unnormalized_term_outside_the_dat_families_is_evaluated_in_centimetres() -> None:
    x, y, _ = POINTS.T
    field = _evaluate_harmonics(_single("unnormalized", "alpha", 0, 2, 2, 0.1), POINTS)
    # 10 (r / 10)^2 * 3 sin^2(theta) cos(2 phi) = 0.3 (x^2 - y^2), in millimetres.
    np.testing.assert_allclose(field[:, 0], 0.1 * 0.3 * (x * x - y * y), atol=1e-12)


def test_the_isocentre_is_not_displaced() -> None:
    coefficients = _single("normalized", "alpha", 0, 3, 1, 0.1, reference_radius_mm=250.0)
    np.testing.assert_array_equal(_evaluate_harmonics(coefficients, np.zeros((1, 3))), 0.0)


def test_the_cartesian_recurrence_and_the_legendre_expansion_agree_to_ninth_order(
    monkeypatch,
) -> None:
    from bartorch.tools._correct import _gradunwarp

    coefficients = GradientCoefficients.from_file(EMITTED_COIL_DESCRIPTION)
    recurrence = _evaluate_harmonics(coefficients, POINTS)
    monkeypatch.setattr(_gradunwarp, "_has_dat_structure", lambda _: False)
    expansion = _evaluate_harmonics(coefficients, POINTS)
    np.testing.assert_allclose(expansion, recurrence, rtol=1e-10, atol=1e-12)


def test_a_normalized_table_maps_each_voxel_to_where_its_field_puts_it() -> None:
    """A normalized field adds to the position: ``z (1 + 0.02)`` along z."""
    coefficients = _single("normalized", "alpha", 2, 1, 0, 0.02, reference_radius_mm=250.0)
    direction = np.eye(3)
    correct = Gradunwarp(coefficients, (5, 6, 7), (50.0, 60.0, 70.0), direction, jacobian=False)
    grid = correct.target_grid
    source = grid.copy()
    source[..., 2] *= 1.02
    np.testing.assert_allclose(
        correct.source_grid, correct._acquired.scanner_to_indices(source), atol=1e-12
    )
    np.testing.assert_array_equal(correct.jacobian_grid, 1.0)


def test_a_plane_one_voxel_thick_takes_its_jacobian_from_the_field_itself() -> None:
    alpha = np.zeros((3, 2, 2))
    alpha[0, 1, 1] = 0.01
    correct = Gradunwarp(
        GradientCoefficients("unnormalized", alpha, np.zeros_like(alpha)),
        (1, 12),
        (10.0, 120.0),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
    )
    np.testing.assert_allclose(correct.jacobian_grid, 1.01, atol=1e-8)


def test_clearing_the_cache_recomputes_the_same_grid() -> None:
    correct = Gradunwarp(_zero_coefficients(), (4, 5), (40.0, 50.0))
    first = correct.source_grid
    correct.clear_cache()
    assert correct._source_indices is None
    np.testing.assert_array_equal(correct.source_grid, first)


@needs_simpleitk
def test_an_integer_image_comes_back_as_float32() -> None:
    correct = Gradunwarp(_zero_coefficients(), (6, 7), (60.0, 70.0))
    result = correct(torch.ones(6, 7, dtype=torch.int32))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.ones(6, 7), atol=2e-5, rtol=0)


@needs_simpleitk
def test_an_image_of_another_matrix_is_refused() -> None:
    correct = Gradunwarp(_zero_coefficients(), (6, 7), (60.0, 70.0))
    with pytest.raises(ValueError, match="trailing shape must match"):
        correct(torch.ones(7, 6))


# --- reading a table ----------------------------------------------------------------


def test_an_open_file_and_a_path_string_read_as_the_text_they_hold(tmp_path) -> None:
    path = tmp_path / "coil.dat"
    path.write_text(_dat_payload())
    with open(path, "rb") as handle:
        from_handle = GradientCoefficients.from_file(handle)
    from_string = GradientCoefficients.from_file(str(path))
    expected = GradientCoefficients.from_file(_dat_payload())
    np.testing.assert_array_equal(from_handle.alpha, expected.alpha)
    np.testing.assert_array_equal(from_string.alpha, expected.alpha)
    assert from_string.source_name == "coil.dat"
    assert from_handle.source_name == str(path)


def test_comments_and_blank_lines_in_a_dat_table_are_skipped() -> None:
    annotated = GradientCoefficients.from_file(
        "# gradient coil\n\nGRADWARPTYPE 1  # type\nSCALEX3 3.0\n   \nDELTA 0.0\n"
    )
    plain = GradientCoefficients.from_file("SCALEX3 3.0\n")
    np.testing.assert_array_equal(annotated.alpha, plain.alpha)
    assert annotated.alpha[0, 3, 1] == pytest.approx(3.0 * 2.0 / 3.0)


def test_a_one_line_table_that_names_no_file_is_read_as_text() -> None:
    coefficients = GradientCoefficients.from_file(
        "Alpha_z 1 0 0.5", coefficient_format="coef", reference_radius_mm=200.0
    )
    assert coefficients.alpha[2, 1, 0] == 0.5
    assert coefficients.source_name is None


def test_a_table_given_as_bytes_is_utf8_text() -> None:
    coefficients = GradientCoefficients.from_file(b"Beta_y 2 1 -0.25\n", reference_radius_mm=200.0)
    assert coefficients.beta[1, 2, 1] == -0.25


def test_an_accessor_returning_neither_text_nor_bytes_is_refused() -> None:
    class Numbers:
        def read_coefficients(self):
            return 42

    with pytest.raises(TypeError, match="must return str or bytes"):
        GradientCoefficients.from_file(Numbers())


@pytest.mark.parametrize(
    ("payload", "kwargs", "message"),
    [
        ("SCALEX1 1.0\nDELTA 0.5\n", {}, "DELTA must be zero"),
        ("GRADWARPTYPE 2\nSCALEX1 1.0\n", {}, "Only GRADWARPTYPE 1"),
        ("GRADWARPTYPE 1\n", {"coefficient_format": "dat"}, "No SCALE"),
        ("SCALEX11 1.0\n", {}, "order 11 is not supported"),
        ("0.25 m = R0\n", {}, "No spherical-harmonic entries"),
        ("Alpha_x 1 0 1.0\n", {"coefficient_format": "coef"}, "requires reference_radius_mm"),
        ("0.25 m = R0\n  1 A( 1, 2)  1.0  x\n", {}, "Invalid harmonic degree"),
        ("Alpha_x 1 0 1.0\n", {"coefficient_format": "json"}, "coefficient_format must be"),
    ],
)
def test_a_table_that_cannot_be_read_is_refused(payload, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        GradientCoefficients.from_file(payload, **kwargs)


@pytest.mark.parametrize(
    ("alpha", "beta", "basis", "radius", "message"),
    [
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), "unnormalized", None, "matching shape"),
        (np.zeros((3, 2, 3)), np.zeros((3, 2, 3)), "unnormalized", None, "square"),
        (np.zeros((3, 2, 2)), np.zeros((3, 2, 2)), "spherical", None, "basis must be"),
        (np.zeros((3, 2, 2)), np.zeros((3, 2, 2)), "normalized", None, "reference_radius_mm"),
    ],
)
def test_coefficients_of_an_impossible_shape_or_basis_are_refused(
    alpha, beta, basis, radius, message
) -> None:
    with pytest.raises(ValueError, match=message):
        GradientCoefficients(basis, alpha, beta, reference_radius_mm=radius)


# --- geometry -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "fov", "orientation", "centre", "message"),
    [
        ((4,), (40.0,), np.eye(3)[:, :1], (0.0, 0.0, 0.0), "two or three spatial axes"),
        ((4, 5), (40.0,), None, (0.0, 0.0, 0.0), "one positive value per axis"),
        ((0, 5), (40.0, 50.0), None, (0.0, 0.0, 0.0), "shape entries must be positive"),
        ((4, 5), (40.0, 50.0), np.eye(3), (0.0, 0.0, 0.0), r"direction must have shape \(3, 2\)"),
        ((4, 5), (40.0, 50.0), None, (0.0, 0.0), "center_mm must have shape"),
        ((4, 5), (40.0, 50.0), np.ones((3, 2)), (0.0, 0.0, 0.0), "orthonormal"),
    ],
)
def test_an_impossible_image_geometry_is_refused(shape, fov, orientation, centre, message) -> None:
    with pytest.raises(ValueError, match=message):
        Gradunwarp(_zero_coefficients(), shape, fov, orientation, centre)


@pytest.mark.parametrize(
    ("affine", "shape", "message"),
    [
        (np.eye(4), (4,), "two or three entries"),
        (np.eye(3), (4, 5), r"shape \(4, 4\)"),
        (np.diag([1.0, 0.0, 1.0, 1.0]), (4, 5), "zero-length"),
    ],
)
def test_an_affine_that_is_not_a_voxel_grid_is_refused(affine, shape, message) -> None:
    with pytest.raises(ValueError, match=message):
        Gradunwarp.from_affine(_zero_coefficients(), affine, shape)


def test_from_mrd_reads_the_head_of_an_acquisition_that_carries_one() -> None:
    acquisition = SimpleNamespace(head=_mrd_acquisition())
    correct = Gradunwarp.from_mrd(_mrd_header(_dat_payload()), acquisition)
    np.testing.assert_allclose(correct._acquired.center_mm, (10.0, -20.0, 4.0))


def test_from_mrd_refuses_an_acquisition_without_a_direction() -> None:
    head = _mrd_acquisition()
    del head.slice_dir
    with pytest.raises(ValueError, match="missing 'slice_dir'"):
        Gradunwarp.from_mrd(_mrd_header(_dat_payload()), head)
