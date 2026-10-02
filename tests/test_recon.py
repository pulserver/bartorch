"""Reconstruction tools end to end, against references computed outside BART."""

import numpy as np
import pytest
import torch

import bartorch
import bartorch._reference as ref
import bartorch.tools as bt


def _fully_sampled_coil_kspace(n=32, coils=4):
    ksp = bt.phantom([n, n], kspace=True, coils=coils)
    assert ksp.shape[-2:] == (n, n)
    return ksp


def test_analytical_phantom_kspace_transforms_back_to_the_sampled_phantom():
    # phantom -k is the closed-form Fourier transform of the ellipses, scaled
    # for BART's non-unitary inverse, so it agrees with the sampled image
    # only up to discretisation.
    n = 64
    img = bt.phantom([n, n]).squeeze()
    ksp = bt.phantom([n, n], kspace=True).squeeze()
    back = bartorch.ifft(ksp, axes=(-1, -2))
    a, b = back.abs().flatten().numpy(), img.abs().flatten().numpy()
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_ecalib_maps_are_normalised_where_the_object_is():
    ksp = _fully_sampled_coil_kspace()
    maps = bt.ecalib(ksp, calib_size=16, maps=1)
    coil_axis = -3 if maps.dim() == 3 else -4
    norm = (maps.abs() ** 2).sum(coil_axis).sqrt().squeeze()
    centre = norm[norm.shape[0] // 2, norm.shape[1] // 2]
    assert centre == pytest.approx(1.0, abs=1e-2)


def test_pics_on_fully_sampled_data_matches_the_direct_inverse():
    ksp = _fully_sampled_coil_kspace()
    maps = bt.ecalib(ksp, calib_size=16, maps=1)
    reco = ref.pics(ksp, maps, l2=0.001, maxiter=30, l=2).squeeze()
    coil_images = bartorch.ifft(ksp, axes=(-1, -2), unitary=True).squeeze()
    direct = (coil_images * maps.squeeze().conj()).sum(0)
    assert reco.shape == direct.shape
    scale = (reco.abs().max() / direct.abs().max()).item()
    err = (reco.abs() / scale - direct.abs()).norm() / direct.abs().norm()
    assert err < 0.05


def test_nufft_matches_an_explicit_dft_on_a_radial_trajectory():
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    ksp = bartorch.nufft(img, traj).squeeze()
    trj = traj.numpy().real  # (spokes, samples, 3): kx, ky, kz in grid units
    im = img.numpy().reshape(n, n)
    kx, ky = trj[..., 0], trj[..., 1]
    x = np.arange(n) - n // 2
    phase = np.exp(
        -2j
        * np.pi
        * (
            kx[..., None, None] * x[None, None, None, :] / n
            + ky[..., None, None] * x[None, None, :, None] / n
        )
    )
    ref = (phase * im[None, None]).sum(axis=(-1, -2)) / n
    assert ksp.shape == ref.shape
    diff = ksp.numpy() - ref
    assert np.linalg.norm(diff) / np.linalg.norm(ref) < 5e-3
    assert np.abs(diff).max() / np.abs(ref).max() < 2e-2


# BART's noise covariance is the sum over the noise samples divided by one less
# than their number, with no mean removed, and its whitening matrix is the inverse
# of the covariance's lower Cholesky factor.  The references are that in float64.
#
# BART computes in complex64 (eps 1.2e-7).  Through the Cholesky factorization of a
# covariance of condition number about 1e2 that is good to about 1e2 eps = 1e-5, and
# 1e-4 of the peak leaves a factor of ten for the BLAS in use.  It still tells K - 1
# from K, which moves the covariance by 1 / K = 4e-3 and the whitening by 1 / 2K = 2e-3
# for the K = 256 noise samples used.
WHITENING_TOLERANCE = 1e-4


def _channel_data(seed, shape):
    """Complex Gaussian ``(4, *shape)`` data whose channels are mixed by a random matrix."""
    rng = np.random.default_rng(seed)
    samples = int(np.prod(shape))
    mixing = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    white = rng.standard_normal((4, samples)) + 1j * rng.standard_normal((4, samples))
    return torch.from_numpy((mixing @ white).astype(np.complex64)).reshape(4, *shape)


def _whitening_references(noise):
    """The covariance of ``noise``, and ``W = L^-1`` for its Cholesky factor ``L``."""
    samples = noise.reshape(4, -1).numpy().astype(np.complex128)
    covariance = samples @ samples.conj().T / (samples.shape[1] - 1)
    return covariance, np.linalg.inv(np.linalg.cholesky(covariance))


def _assert_whitening_close(actual, expected):
    atol = WHITENING_TOLERANCE * np.abs(expected).max()
    np.testing.assert_allclose(actual.reshape(expected.shape).numpy(), expected, rtol=0, atol=atol)


def test_whitening_makes_noise_covariance_the_identity():
    """The noise whitened by its own covariance is ``L^-1 n`` for the Cholesky factor ``L`` of
    that covariance, and has the identity as its covariance by the same normalization."""
    noise = _channel_data(0, (1, 16, 16))
    _, whitening = _whitening_references(noise)

    white = bt.whiten(noise, noise)

    _assert_whitening_close(white, whitening @ noise.reshape(4, -1).numpy())
    w = white.reshape(4, -1).numpy()
    covariance = w @ w.conj().T / (w.shape[1] - 1)
    np.testing.assert_allclose(covariance, np.eye(4), rtol=0, atol=WHITENING_TOLERANCE)


@pytest.mark.parametrize(
    ("matrix", "covariance"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_whiten_returns_the_matrix_and_covariance_asked_for_in_barts_order(matrix, covariance):
    """BART's outputs are positional: the covariance is the third, and asked for alone it must
    not come back in the matrix's place.  The matrix is ``W``, acting on the channel vector;
    the covariance is transposed, its ``[i, j]`` being the sum over samples of
    ``conj(n_i) n_j`` divided by K - 1."""
    noise = _channel_data(0, (1, 16, 16))
    reference, whitening = _whitening_references(noise)
    expected = [whitening @ noise.reshape(4, -1).numpy()]
    if matrix:
        expected.append(whitening)
    if covariance:
        expected.append(reference.T)

    found = bt.whiten(noise, noise, return_matrix=matrix, return_covariance=covariance)

    assert isinstance(found, torch.Tensor) == (len(expected) == 1)
    found = (found,) if isinstance(found, torch.Tensor) else found
    assert len(found) == len(expected)
    for actual, wanted in zip(found, expected):
        _assert_whitening_close(actual, wanted)
    assert all(tuple(x.shape) == (4, 4, 1, 1, 1) for x in found[1:])


@pytest.mark.parametrize(("option", "index"), [("o", 1), ("c", 2)])
def test_a_returned_matrix_or_covariance_given_back_whitens_other_data_the_same_way(option, index):
    """``o`` and ``c`` take the arrays as returned.  The other data stand as their own noise
    measurement, which is not the one the arrays came from, so the array given back is what
    whitens them."""
    noise = _channel_data(0, (1, 16, 16))
    other = _channel_data(1, (1, 5, 7))
    _, whitening = _whitening_references(noise)
    returned = bt.whiten(noise, noise, return_matrix=True, return_covariance=True)

    white = bt.whiten(other, other, **{option: returned[index]})

    _assert_whitening_close(white, whitening @ other.reshape(4, -1).numpy())


def test_ecalib_of_one_slice_without_a_z_axis_gives_that_slice_s_maps():
    kspace = bt.phantom(64, coils=4, kspace=True)

    np.testing.assert_array_equal(
        bt.ecalib(kspace[:, 0], maps=1).numpy(), bt.ecalib(kspace, maps=1)[:, 0].numpy()
    )
