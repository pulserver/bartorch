"""``reslice`` is pinned against linear functions of patient position sampled at voxel centres."""

import numpy as np
import pytest
import torch

pytest.importorskip("SimpleITK")

from bartorch.tools import reslice  # noqa: E402


def rotation(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def geometry(position, fov, matrix, rot=None):
    rot = np.eye(3) if rot is None else rot
    return {
        "position": np.asarray(position, dtype=float),
        "read_dir": rot[:, 0],
        "phase_dir": rot[:, 1],
        "slice_dir": rot[:, 2],
        "field_of_view": np.asarray(fov, dtype=float),
        "matrix_size": np.asarray(matrix),
    }


def centres(g):
    """Patient position of every voxel centre, shape (z, y, x, 3)."""
    n = np.asarray(g["matrix_size"])
    d = np.asarray(g["field_of_view"]) / n
    axes = [(np.arange(n[i]) - n[i] // 2) * d[i] for i in range(3)]
    kk, jj, ii = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return (
        g["position"]
        + ii[..., None] * g["read_dir"]
        + jj[..., None] * g["phase_dir"]
        + kk[..., None] * g["slice_dir"]
    )


def linear(r, a, b):
    return r @ np.asarray(a) + b


SRC = geometry([5.0, -3.0, 2.0], [40.0, 48.0, 30.0], [20, 24, 10], rotation(0.1, -0.2, 0.3))
A = [0.013, -0.021, 0.034]


def source_image():
    return linear(centres(SRC), A, 0.7).astype(np.float32)


def test_identical_geometry_returns_input():
    image = np.random.default_rng(0).standard_normal((10, 24, 20)).astype(np.float32)
    assert np.allclose(reslice(image, SRC, SRC), image, atol=1e-5)


def test_integer_voxel_shift_along_read_shifts_image_along_x():
    image = np.random.default_rng(1).standard_normal((10, 24, 20)).astype(np.float32)
    dx = SRC["field_of_view"][0] / SRC["matrix_size"][0]
    shifted = dict(SRC, position=SRC["position"] + 3 * dx * SRC["read_dir"])
    out = reslice(image, SRC, shifted, fill=-9.0)
    assert np.allclose(out[..., :-3], image[..., 3:], atol=1e-4)
    assert np.all(out[..., -3:] == -9.0)


def test_linear_function_is_exact_on_oblique_target():
    target = geometry([5.5, -3.5, 2.5], [20.0, 16.0, 8.0], [13, 11, 7], rotation(-0.3, 0.2, 0.5))
    out = reslice(source_image(), SRC, target)
    expected = linear(centres(target), A, 0.7)
    assert np.allclose(out, expected, rtol=1e-4, atol=1e-4)


def test_plane_sequence_equals_volume_geometry():
    n = SRC["matrix_size"][2]
    dz = SRC["field_of_view"][2] / n
    planes = [
        dict(SRC, position=SRC["position"] + (k - n // 2) * dz * SRC["slice_dir"]) for k in range(n)
    ]
    target = geometry([5.5, -3.5, 2.5], [20.0, 16.0, 8.0], [13, 11, 7], rotation(-0.3, 0.2, 0.5))
    assert np.allclose(
        reslice(source_image(), planes, target), reslice(source_image(), SRC, target), atol=1e-5
    )


def test_unequal_plane_spacing_raises():
    planes = [dict(SRC, position=SRC["position"] + k**2 * SRC["slice_dir"]) for k in range(10)]
    with pytest.raises(ValueError, match="equally spaced"):
        reslice(source_image(), planes, SRC)


def test_complex_parts_resampled_independently_with_leading_axes():
    target = geometry([5.5, -3.5, 2.5], [20.0, 16.0, 8.0], [13, 11, 7], rotation(-0.3, 0.2, 0.5))
    b = [-0.02, 0.011, 0.005]
    f = linear(centres(SRC), A, 0.7)
    g = linear(centres(SRC), b, -1.3)
    image = np.stack([(c + 1) * f + 1j * (c - 2) * g for c in range(3)]).astype(np.complex64)
    out = reslice(image, SRC, target)
    ft, gt = linear(centres(target), A, 0.7), linear(centres(target), b, -1.3)
    assert out.shape == (3, 7, 11, 13) and out.dtype == np.complex64
    for c in range(3):
        assert np.allclose(out[c].real, (c + 1) * ft, rtol=1e-4, atol=1e-4)
        assert np.allclose(out[c].imag, (c - 2) * gt, rtol=1e-4, atol=1e-4)


def test_container_type_is_preserved():
    target = geometry([5.5, -3.5, 2.5], [20.0, 16.0, 8.0], [13, 11, 7])
    image = source_image()
    out_np = reslice(image, SRC, target)
    out_t = reslice(torch.from_numpy(image), SRC, target)
    assert isinstance(out_np, np.ndarray) and isinstance(out_t, torch.Tensor)
    assert out_t.dtype == torch.float32 and tuple(out_t.shape) == (7, 11, 13)
    assert np.allclose(out_t.numpy(), out_np)
    out_c = reslice(torch.from_numpy(image).to(torch.complex64), SRC, target)
    assert out_c.is_complex()


def test_mismatched_shape_and_unknown_interpolation_raise():
    with pytest.raises(ValueError, match="does not match"):
        reslice(np.zeros((9, 24, 20), np.float32), SRC, SRC)
    with pytest.raises(ValueError, match="interpolation"):
        reslice(source_image(), SRC, SRC, interpolation="cubic")
