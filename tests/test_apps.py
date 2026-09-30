"""``bartorch.apps`` against the applications they are assembled from.

An app is a re-expression of a BART application, so on a grid the test is
``torch.equal``: a difference in the last place would mean some step was done
twice, once by BART and once by this package.
"""

from __future__ import annotations

import pytest
import torch

import bartorch
import bartorch._reference as ref
import bartorch.tools as bt
from bartorch import _dispatch, _finufft, apps, linop, priors
from bartorch.priors._terms import ImageNIHT

SIZE, COILS, ACCEL = 24, 4, 2


@pytest.fixture
def _whole_coil_operator():
    """``pics`` walks every coil at once; the package's default is a slab."""
    before = _dispatch.coil_batch()
    _dispatch.set_coil_batch(0)
    yield
    _dispatch.set_coil_batch(before)


def _cartesian():
    kspace = bt.phantom(SIZE, coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    mask = torch.zeros(SIZE, dtype=torch.complex64)
    mask[::ACCEL] = 1
    mask[SIZE // 2 - 2 : SIZE // 2 + 2] = 1
    return kspace * mask.reshape(SIZE, 1), maps


def _wavelet(**kwargs):
    return priors.Wavelet(axes=(-1, -2), weight=0.01, **kwargs)


def _tv(weight=0.01):
    return priors.TotalVariation(axes=(-1, -2), weight=weight)


def _llr():
    return priors.LocallyLowRank(axes=(-1, -2), weight=0.01, block=4)


def _tgv(weight=0.01):
    return priors.TotalGeneralizedVariation(axes=(-1, -2), weight=weight)


#: What the tool is given.  The app is given the same, which is the point: the
#: two take the same arguments and the app is the one without the command.
_CONFIGURATIONS = [
    ("plain", {}),
    ("tikhonov", {"l2": 0.1}),
    # `-R Q:` and `-r` name the same `L2IMG` term (grecon/optreg.c:386), and
    # conjugate gradients has no proximal step to apply it with, so the app
    # has to read one as the other.
    ("tikhonov as a term", {"regularizers": priors.L2(0.1)}),
    ("wavelet admm", {"regularizers": _wavelet(), "solver": "admm"}),
    ("wavelet fista", {"regularizers": _wavelet(), "solver": "fista"}),
    ("wavelet ist", {"regularizers": _wavelet(), "solver": "ist"}),
    ("no cycle spinning", {"regularizers": _wavelet(randshift=False), "solver": "fista"}),
    ("tv pridu", {"regularizers": _tv(), "solver": "pridu"}),
    ("tv admm", {"regularizers": _tv(0.005), "solver": "admm"}),
    ("locally low rank", {"regularizers": _llr(), "solver": "admm"}),
    ("two terms", {"regularizers": [_wavelet(), _tv(0.005)], "solver": "admm"}),
    ("tgv admm", {"regularizers": _tgv(), "solver": "admm"}),
    ("tgv pridu", {"regularizers": _tgv(), "solver": "pridu"}),
    # No solver named: the app chooses the one `italgo_choose` chooses.
    ("wavelet, chosen", {"regularizers": _wavelet()}),
    ("tv, chosen", {"regularizers": _tv()}),
    ("two terms, chosen", {"regularizers": [_wavelet(), _tv(0.005)]}),
]


@pytest.mark.parametrize(
    "arguments", [a for _, a in _CONFIGURATIONS], ids=[n for n, _ in _CONFIGURATIONS]
)
def test_the_app_is_the_tool_to_the_last_bit(arguments, _whole_coil_operator):
    kspace, maps = _cartesian()
    tool = ref.pics(kspace, maps, maxiter=20, **arguments).squeeze()
    ours = apps.pics(kspace, maps, maxiter=20, **arguments).squeeze()
    assert torch.equal(ours, tool), f"maximum difference {float((ours - tool).abs().max()):.3e}"


def test_the_iteration_is_the_one_the_terms_choose():
    """``italgo_choose`` (grecon/italgo.c), written out: an l2 penalty on the
    image leaves the choice where it was, the total variations take ADMM, and
    anything else takes FISTA first and ADMM after."""
    from bartorch.apps._pics import _chosen

    assert _chosen([]) == "cg"
    assert _chosen([priors.L2(0.01)]) == "cg"
    assert _chosen([_wavelet()]) == "fista"
    assert _chosen([_tv()]) == "admm"
    assert _chosen([_wavelet(), _tv()]) == "admm"
    assert _chosen([_tv(), _wavelet()]) == "admm"
    assert _chosen([ImageNIHT((-1, -2), 4)]) == "niht"


def test_an_l2_term_is_the_weight_once(_whole_coil_operator):
    kspace, maps = _cartesian()
    with pytest.raises(ValueError, match="given once"):
        apps.pics(kspace, maps, regularizers=priors.L2(0.1), l2=0.2)
    with pytest.raises(ValueError, match="not several"):
        apps.pics(kspace, maps, regularizers=[priors.L2(0.1), priors.L2(0.2)])


@pytest.mark.parametrize(
    "term", [priors.FourierL1((-1, -2), 0.01), priors.Laplace((-1, -2), 0.01)], ids=repr
)
def test_a_first_term_over_a_transform_is_not_thresholded_on_the_image(term, _whole_coil_operator):
    """The application chooses FISTA for these and hands it no transform
    (``trafos_cond`` in pics.c), so the tool thresholds the image itself.  The
    app refuses that choice and takes the term under ADMM, which applies it."""
    kspace, maps = _cartesian()
    with pytest.raises(ValueError, match="solver='admm'"):
        apps.pics(kspace, maps, regularizers=term)
    tool = ref.pics(kspace, maps, maxiter=20, regularizers=term, solver="admm").squeeze()
    ours = apps.pics(kspace, maps, maxiter=20, regularizers=term, solver="admm").squeeze()
    assert torch.equal(ours, tool)


def test_an_unknown_solver_is_refused():
    kspace, maps = _cartesian()
    with pytest.raises(ValueError, match="solver must be one of"):
        apps.pics(kspace, maps, solver="newton")


@pytest.mark.parametrize("solver", ["cg", "ist", "fista", "admm"])
def test_a_warm_start_reaches_the_iteration(solver, _whole_coil_operator):
    """And moves the answer, so the equality is not two cold starts agreeing.

    ``pics -W`` rescales the warm start only under ``-S``, where the answer is
    put back into the data's units at the end (pics.c:592).  Without it the
    solve stays in the scaled units and so does the start, which is the way
    the app takes one.
    """
    kspace, maps = _cartesian()
    arguments = {} if solver == "cg" else {"regularizers": _wavelet(), "solver": solver}
    warm = 0.5 * apps.pics(kspace, maps, maxiter=5)

    tool = ref.pics(kspace, maps, maxiter=20, W=warm, **arguments).squeeze()
    ours = apps.pics(kspace, maps, maxiter=20, initial=warm, **arguments).squeeze()
    cold = ref.pics(kspace, maps, maxiter=20, **arguments).squeeze()

    assert torch.equal(ours, tool)
    assert not torch.equal(tool, cold), "the warm start changed nothing, so this proves nothing"


def test_the_eigenvalue_step_is_the_tools():
    """``pics -e`` is ``eigen ? 30 : 0`` power iterations (pics.c:667).

    Held to round-off rather than to the bits: the starting vector comes from
    BART's process-global generator, so the tool's call and the app's do not
    start it from the same place.
    """
    kspace, maps = _cartesian()
    arguments = {"regularizers": _wavelet(), "solver": "fista"}
    tool = ref.pics(kspace, maps, maxiter=20, eigen_step=True, **arguments).squeeze()
    ours = apps.pics(kspace, maps, maxiter=20, eigen_step=True, **arguments).squeeze()
    plain = apps.pics(kspace, maps, maxiter=20, **arguments).squeeze()

    assert float((ours - tool).abs().max()) < 1e-5 * float(tool.abs().max())
    assert not torch.equal(ours, plain), "the eigenvalue step changed nothing"


def test_conjugate_gradients_takes_no_eigenvalue_step():
    kspace, maps = _cartesian()
    with pytest.raises(ValueError, match="cg does not take"):
        apps.pics(kspace, maps, eigen_step=True)


def _radial(size=32, coils=4, spokes=32):
    sens = bt.coils(t=bt.grid(D=(size, size, 1)), n=coils)[:, 0]
    sens = sens / bartorch.rss(sens, axes=(0,), keepdim=True)
    traj = bt.traj(readout=size, spokes=spokes, radial=True, golden=True)
    encoding = linop.NoncartesianSense(sens, (size, size), traj=traj)
    image = torch.as_tensor(bt.phantom(size)).to(torch.complex64)
    return traj, sens[:, None], bt.noise(encoding(image), n=1e-6, s=7)


def test_off_the_grid_the_app_is_the_tool_to_round_off():
    """Not the bits, and they cannot be: a non-uniform transform spread over
    threads sums in the order the threads finish in, so nothing off the grid is
    bit-reproducible.  ``pics`` is not reproducible against itself there --
    twice over the same data it differs by about 6e-07 of the peak, the same
    size as the difference asserted here -- so an equality would be a statement
    about the thread count rather than about the pipeline.
    """
    traj, maps, measured = _radial()
    term = _tv(0.001)
    tool = ref.pics(
        measured[..., None], maps, traj=traj, regularizers=term, solver="admm", maxiter=10
    ).squeeze()
    ours = apps.pics(
        measured, maps, traj=traj, regularizers=term, solver="admm", maxiter=10
    ).squeeze()
    assert float((ours - tool).abs().max()) < 1e-5 * float(tool.abs().max())


# --- mobafit ---------------------------------------------------------------
#
# The model the app fits is TorchSim's rather than BART's, so there is nothing
# to be equal to; what a fit has to answer for is the relaxation time the data
# was made from, and the decay and the recovery are written out here.

FIT_SIZE = 12
ECHO_TIMES = [12.5 * (echo + 1) for echo in range(8)]
INVERSION_TIMES = [50.0, 150.0, 400.0, 900.0, 1500.0, 2500.0, 4000.0]


def _two_halves(left: float, right: float, size: int = FIT_SIZE) -> torch.Tensor:
    """A map of two constant halves, so one fit answers two relaxation times."""
    values = torch.full((size, size), left)
    values[:, size // 2 :] = right
    return values


def test_a_decay_is_fitted_to_the_time_it_was_made_from():
    """``M0 exp(-TE / T2)``, written out here and recovered from the images."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)

    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    fitted = apps.mobafit(images, model, T2=80.0)["T2"]

    assert torch.allclose(fitted, t2, rtol=1e-3)


def test_a_recovery_is_fitted_to_the_time_it_was_made_from():
    from bartorch import nlop

    model = nlop.InversionRecovery(INVERSION_TIMES, (FIT_SIZE, FIT_SIZE))
    left = model(model.initial(T1=800.0))
    right = model(model.initial(T1=1400.0))
    images = torch.cat([left[..., : FIT_SIZE // 2], right[..., FIT_SIZE // 2 :]], dim=-1)

    fitted = apps.mobafit(images, model, T1=1000.0)["T1"]

    assert torch.allclose(fitted, _two_halves(800.0, 1400.0), rtol=1e-3)


def test_the_magnitude_is_fitted_where_the_phase_is_thrown_away():
    """``mobafit -a``: a decay whose phase varies across the image, given to
    the fit as a magnitude, is the same T2 as the decay itself."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    phase = torch.exp(1j * torch.linspace(-1.0, 1.0, FIT_SIZE))[None, :]
    images = (torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2) * phase).abs()

    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    fitted = apps.mobafit(images.to(torch.complex64), model, magnitude=True, T2=80.0)["T2"]

    assert torch.allclose(fitted, t2, rtol=1e-3)


def test_a_magnitude_fit_discards_the_phase_of_complex_data():
    """``mobafit -a`` takes the modulus of the data as well as of the model, so
    a phased complex decay fits its T2 and the modulus of its amplitude."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    phase = torch.exp(1j * torch.linspace(-1.0, 1.0, FIT_SIZE))[None, :]
    images = 2.0 * torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2) * phase

    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    fitted = apps.mobafit(images.to(torch.complex64), model, magnitude=True, T2=80.0)

    assert torch.allclose(fitted["T2"], t2, rtol=1e-3)
    assert torch.allclose(fitted["amplitude"].abs(), torch.full_like(t2, 2.0), rtol=1e-3)


def test_a_voxel_with_no_signal_keeps_the_value_it_started_from():
    """``mobafit`` skips a patch whose data is zero; an unconstrained voxel
    would otherwise walk wherever the bounds allow."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)
    images[:, 0] = 0.0

    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    fitted = apps.mobafit(images, model, T2=80.0)["T2"]

    assert torch.allclose(fitted[0], torch.full((FIT_SIZE,), 80.0), atol=1e-3)
    assert torch.allclose(fitted[1:], t2[1:], rtol=1e-3)


def test_mobafit_answers_the_same_decay_whatever_units_the_images_are_in():
    """Images in a scanner's arbitrary units fit to the T2 of the same images
    at unit peak, and the amplitude comes back in the units it went in."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)
    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))

    unit = apps.mobafit(images, model, T2=80.0)
    for scale in (1e3, 1e-3):
        scaled = apps.mobafit(images * scale, model, T2=80.0)
        torch.testing.assert_close(scaled["T2"], unit["T2"], rtol=1e-4, atol=0.0)
        torch.testing.assert_close(
            scaled["amplitude"] / scale, unit["amplitude"], rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(scaled["T2"], t2, rtol=1e-3, atol=0.0)


def test_an_amplitude_given_in_the_images_units_starts_the_fit_where_it_would_at_unit_peak():
    """A start of 1000 on images at a peak of 1000 is a start of one on the
    same images at unit peak, so even one step lands in the same place."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)
    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))

    unit = apps.mobafit(images, model, iterations=1, T2=40.0, amplitude=1.0)
    scaled = apps.mobafit(1e3 * images, model, iterations=1, T2=40.0, amplitude=1e3)

    torch.testing.assert_close(scaled["T2"], unit["T2"], rtol=1e-4, atol=0.0)
    torch.testing.assert_close(scaled["amplitude"], 1e3 * unit["amplitude"], rtol=1e-4, atol=1e-3)


def test_the_fit_is_taken_from_where_it_is_started():
    """The starting maps reach the loop: a fit stopped after one step is still
    near where it began, and a different beginning is a different answer."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)
    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))

    one = apps.mobafit(images, model, iterations=1, T2=40.0)["T2"]
    other = apps.mobafit(images, model, iterations=1, T2=200.0)["T2"]

    assert float((one - other).abs().max()) > 1.0
    assert float(one.median()) < float(other.median())


def test_mobafit_regularizes_towards_zero_in_the_models_variables_by_default():
    """Zero in the model's variables is the middle of each bound and no
    amplitude, so stating that as the reference changes nothing."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)
    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    low, high = model.model.bounds["T2"]

    default = apps.mobafit(images, model, iterations=3, T2=80.0)
    middle = {"T2": 0.5 * (low + high), "amplitude": 0.0}
    stated = apps.mobafit(images, model, iterations=3, T2=80.0, reference=middle)

    for name, value in default.items():
        torch.testing.assert_close(stated[name], value)


def test_a_heavily_weighted_reference_pulls_the_fitted_decay_towards_it():
    """With a weight that does not decay, a reference of no decay at all -- an
    R2 of zero, clamped inside the bound -- draws every voxel above the T2 a
    reference of a fast decay draws it to."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)
    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    settings = {"alpha": 1e3, "alpha_min": 1e3, "redu": 1.0, "T2": 80.0}

    slow = apps.mobafit(images, model, **settings, reference={"T2": float("inf"), "amplitude": 1.0})
    fast = apps.mobafit(images, model, **settings, reference={"T2": 20.0, "amplitude": 1.0})

    assert bool((slow["T2"] > fast["T2"]).all())
    assert bool((slow["T2"] > t2).all())
    assert bool((fast["T2"] < t2).all())


# --- moba ------------------------------------------------------------------
#
# The same decay as above, now behind coils and an FFT: the echo images are
# written out, the sensitivities are smooth analytic profiles, the k-space is
# torch's own centred unitary FFT of their product, and each echo keeps its own
# random half of the phase encodes.  What the fit answers for is the T2 the
# images were made from.

MOBA_SIZE, MOBA_COILS, MOBA_ECHOES = 16, 4, 6
MOBA_ECHO_TIMES = [12.5 * (echo + 1) for echo in range(MOBA_ECHOES)]
#: Fewer steps than the app's default, which is enough for a noiseless phantom
#: and keeps each fit to a few seconds.
MOBA_SETTINGS = {"iterations": 12, "cg_maxiter": 20}


def _moba_phantom(size: int = MOBA_SIZE):
    """Echo images of a disc of two T2 halves, smooth coils, and the support."""
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size), torch.linspace(-1.0, 1.0, size), indexing="ij"
    )
    support = x**2 + y**2 < 0.8
    t2 = _two_halves(60.0, 110.0, size)
    amplitude = torch.where(support, 1.0, 0.0) * torch.exp(0.5j * x)
    te = torch.tensor(MOBA_ECHO_TIMES)[:, None, None]
    images = (amplitude * torch.exp(-te / t2)).to(torch.complex64)
    angles = torch.arange(MOBA_COILS) * (2 * torch.pi / MOBA_COILS)
    coils = torch.stack(
        [
            torch.exp(-((x - 1.5 * torch.cos(a)) ** 2 + (y - 1.5 * torch.sin(a)) ** 2) / 18.0)
            * torch.exp(1j * a)
            for a in angles
        ]
    ).to(torch.complex64)
    return images, coils, t2, support, amplitude


def _moba_kspace(images, coils):
    """``(echoes, coils, y, x)``: a random half of the lines per echo, and the centre."""
    coil_images = images[:, None] * coils[None]
    kspace = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(coil_images, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    size = images.shape[-1]
    generator = torch.Generator().manual_seed(0)
    lines = torch.zeros(MOBA_ECHOES, 1, size, 1)
    for echo in range(MOBA_ECHOES):
        lines[echo, 0, torch.randperm(size, generator=generator)[: size // 2]] = 1.0
        lines[echo, 0, size // 2 - 3 : size // 2 + 3] = 1.0
    return kspace * lines


def _relative(fitted, reference, support):
    return (fitted - reference)[support].abs() / reference[support]


def test_moba_fits_the_decay_behind_known_coils():
    from bartorch import nlop

    images, coils, t2, support, _ = _moba_phantom()
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE))

    fitted = apps.moba(_moba_kspace(images, coils), model, coils, **MOBA_SETTINGS, T2=80.0)

    error = _relative(fitted["T2"], t2, support)
    assert float(error.median()) < 0.02
    assert float(error.max()) < 0.08


def test_moba_estimates_the_coils_with_the_decay():
    """Without sensitivities the coils are a second unknown; the T2 is still the
    one the images were made from, and the product of the fitted amplitude and
    coils -- which the data does fix, unlike either factor -- is the one the
    k-space was made from."""
    from bartorch import nlop

    images, coils, t2, support, amplitude = _moba_phantom()
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE))

    fitted, estimated = apps.moba(
        _moba_kspace(images, coils), model, return_sensitivities=True, **MOBA_SETTINGS, T2=80.0
    )

    error = _relative(fitted["T2"], t2, support)
    assert float(error.median()) < 0.03
    assert float(error.max()) < 0.08
    assert estimated.shape == coils.shape
    product = (fitted["amplitude"] * estimated)[:, support]
    truth = (amplitude * coils)[:, support]
    assert float((product - truth).norm() / truth.norm()) < 0.15


def test_moba_fits_a_model_without_an_amplitude_to_unscaled_data():
    """Only an amplitude can absorb a scaling of the data, so a model without
    one is fitted to the k-space as it stands: unit-amplitude echoes fit the
    T2 they were made from."""
    from bartorch import nlop

    _, coils, t2, _, _ = _moba_phantom()
    te = torch.tensor(MOBA_ECHO_TIMES)[:, None, None]
    images = torch.exp(-te / t2).to(torch.complex64)
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE), amplitude=False)

    fitted = apps.moba(_moba_kspace(images, coils), model, coils, **MOBA_SETTINGS, T2=80.0)

    assert list(fitted) == ["T2"]
    assert float(((fitted["T2"] - t2).abs() / t2).max()) < 0.01


def test_moba_estimating_the_coils_returns_the_maps_alone_unless_asked():
    from bartorch import nlop

    images, coils, *_ = _moba_phantom()
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE))
    kspace = _moba_kspace(images, coils)

    alone = apps.moba(kspace, model, **MOBA_SETTINGS, T2=80.0)
    with_coils, _ = apps.moba(kspace, model, return_sensitivities=True, **MOBA_SETTINGS, T2=80.0)

    assert isinstance(alone, dict)
    for name, value in with_coils.items():
        assert torch.equal(alone[name], value)


def test_moba_regularizes_towards_the_start_unless_given_a_reference():
    from bartorch import nlop

    images, coils, *_ = _moba_phantom()
    kspace = _moba_kspace(images, coils)
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE))

    default = apps.moba(kspace, model, coils, **MOBA_SETTINGS, T2=80.0)
    stated = apps.moba(kspace, model, coils, **MOBA_SETTINGS, T2=80.0, reference={"T2": 80.0})

    for name, value in default.items():
        assert torch.equal(stated[name], value)


@pytest.mark.parametrize("sensitivities", [True, False], ids=["given", "estimated"])
def test_a_heavily_weighted_reference_pulls_the_decay_towards_it(sensitivities):
    """With a weight that does not decay, the fit is drawn to the reference.

    A reference of no decay at all -- an R2 of zero, a T2 past the upper bound,
    clamped inside it -- draws every voxel above the T2 a reference of a fast
    decay draws it to, and above the start.
    """
    from bartorch import nlop

    images, coils, _, support, _ = _moba_phantom()
    kspace = _moba_kspace(images, coils)
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE))
    settings = {**MOBA_SETTINGS, "alpha": 1e3, "alpha_min": 1e3, "redu": 1.0, "T2": 80.0}
    given = (coils,) if sensitivities else ()

    slow = apps.moba(kspace, model, *given, **settings, reference={"T2": float("inf")})
    fast = apps.moba(kspace, model, *given, **settings, reference={"T2": 20.0})

    assert bool((slow["T2"][support] > fast["T2"][support]).all())
    assert bool((slow["T2"][support] > 80.0).all())
    assert bool((fast["T2"][support] < 80.0).all())


def test_moba_refuses_kspace_that_is_not_the_encodings_samples():
    from bartorch import nlop

    images, coils, *_ = _moba_phantom()
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE))
    with pytest.raises(ValueError, match=r"kspace is \(contrasts, coils, \*samples\)"):
        apps.moba(_moba_kspace(images, coils)[..., : MOBA_SIZE // 2], model, coils)


def test_moba_does_not_depend_on_the_scale_of_the_data():
    """The data is scaled before the solve and the amplitude after it, so a
    k-space a thousand times larger fits the same T2 and a thousand times the
    amplitude."""
    from bartorch import nlop

    images, coils, _, support, _ = _moba_phantom()
    kspace = _moba_kspace(images, coils)
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE))

    one = apps.moba(kspace, model, coils, **MOBA_SETTINGS, T2=80.0)
    other = apps.moba(1000.0 * kspace, model, coils, **MOBA_SETTINGS, T2=80.0)

    assert torch.allclose(other["T2"][support], one["T2"][support], rtol=1e-3)
    assert torch.allclose(
        other["amplitude"][support], 1000.0 * one["amplitude"][support], rtol=1e-3
    )


def test_moba_fits_the_decay_along_a_trajectory_of_its_own_per_echo():
    """Radial spokes, a different set per echo, through the encoding the app
    fits with; the known quantity is still the T2."""
    from bartorch import nlop

    size, spokes = 12, 8
    images, coils, t2, support, _ = _moba_phantom(size)
    traj = bt.traj(readout=2 * size, spokes=spokes * MOBA_ECHOES, radial=True, golden=True)
    traj = traj.reshape(MOBA_ECHOES, spokes, *traj.shape[-2:])
    encoding = linop.NoncartesianSense(coils, (MOBA_ECHOES, size, size), traj=traj)
    # The encoding puts a contrast axis the trajectory indexes behind the
    # coils; the app takes the contrasts first.
    kspace = encoding(images).transpose(0, 1)

    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (size, size))
    fitted = apps.moba(kspace, model, coils, traj=traj, **MOBA_SETTINGS, T2=80.0)

    error = _relative(fitted["T2"], t2, support)
    assert float(error.median()) < 0.02
    assert float(error.max()) < 0.1


@pytest.mark.parametrize(
    "arguments, match",
    [
        (["sensitivities", "return_sensitivities"], "estimated sensitivities"),
        (["traj"], "off the grid"),
        (["inner"], "whole state"),
    ],
)
def test_moba_refuses_what_it_cannot_estimate(arguments, match):
    from bartorch import nlop, optim

    images, coils, *_ = _moba_phantom()
    model = nlop.MultiEcho(MOBA_ECHO_TIMES, (MOBA_SIZE, MOBA_SIZE))
    given = {
        "sensitivities": coils,
        "return_sensitivities": True,
        "traj": bt.traj(readout=MOBA_SIZE, spokes=4),
        "inner": optim.CG(),
    }
    with pytest.raises(ValueError, match=match):
        apps.moba(_moba_kspace(images, coils), model, **{name: given[name] for name in arguments})


# --- pocsense --------------------------------------------------------------


def _centred(x: torch.Tensor, axes: tuple[int, ...], inverse: bool = False) -> torch.Tensor:
    """numpy's centred unitary transform, which is what the app's samples are in."""
    shifted = torch.fft.ifftshift(x, dim=axes)
    moved = (torch.fft.ifftn if inverse else torch.fft.fftn)(shifted, dim=axes, norm="ortho")
    return torch.fft.fftshift(moved, dim=axes)


def _range(samples: torch.Tensor, maps: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    """``E E^H`` written out with torch: combine the coils and spread them again."""
    image = (maps.conj() * _centred(samples, axes, inverse=True)).sum(0, keepdim=True)
    return _centred(maps * image, axes)


def _in_range(shape: int | list[int]):
    """Coil samples ``E x`` of a phantom, fully sampled, and the maps they were made with."""
    kspace = bt.phantom(shape, coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    axes = tuple(axis for axis in range(-kspace.ndim + 1, 0) if kspace.shape[axis] > 1)
    return _range(kspace, maps, axes), maps, axes


@pytest.mark.parametrize("shape", [SIZE, [16, 16, 16]], ids=["plane", "volume"])
def test_samples_the_coils_could_have_made_are_a_fixed_point(shape):
    """Every sample measured and every sample in the range of the maps: both
    projections leave it where it is."""
    samples, maps, _ = _in_range(shape)
    made = apps.pocsense(samples, maps, maxiter=5)
    scale = float(samples.abs().max())
    assert float((made - samples).abs().max()) / scale < 1e-4


@pytest.mark.parametrize(
    "arguments",
    [{}, {"alpha": 0.1}, {"robust": 0.05}, {"maxiter": 1}],
    ids=["plain", "l2", "robust", "one sweep"],
)
def test_the_answer_lies_in_the_range_of_the_coils(arguments):
    """Without a sparsity projection the last one in a sweep is onto the
    coils, or a scaling of it, so the answer is its own projection."""
    kspace, maps = _cartesian()
    made = apps.pocsense(kspace, maps, **{"maxiter": 10, **arguments})
    scale = float(made.abs().max())
    assert float((_range(made, maps, (-2, -1)) - made).abs().max()) / scale < 1e-4


def test_the_sweep_is_the_projections_in_order():
    """``pocs`` applies every projection in turn and keeps nothing else, so a
    sweep is the composition and a run is that composition repeated."""
    from bartorch import optim

    first = torch.zeros(4, dtype=torch.complex64)
    steps: list[str] = []

    def one(x):
        steps.append("one")
        return x + 1.0

    def half(x):
        steps.append("half")
        return x * 0.5

    answer = optim.POCS([one, half], maxiter=3)(first)

    assert steps == ["one", "half", "one", "half", "one", "half"]
    # x -> (x + 1) / 2 from zero: 1/2, 3/4, 7/8.
    assert torch.allclose(answer, torch.full((4,), 0.875, dtype=torch.complex64))


def test_a_sweep_starts_where_it_is_told():
    from bartorch import optim

    block = optim.POCSBlock([lambda x: x * 2.0])
    y = torch.ones(3, dtype=torch.complex64)

    assert torch.equal(block.start(y).x, torch.zeros(3, dtype=torch.complex64))
    assert torch.equal(block.start(y, None, y).x, y)
    assert torch.equal(block.output(block(block.start(y, None, y))), 2.0 * y)


def test_a_term_is_taken_as_its_projection():
    """``pocs`` calls a projection with ``mu = 1``, so a term enters as its
    proximal operator at that step size."""
    from bartorch import optim, priors

    x = torch.tensor([4.0, -2.0, 0.5], dtype=torch.complex64)
    term = priors.L2(0.25)

    swept = optim.POCS([term], maxiter=1)(x, None, x)

    assert torch.allclose(swept, x / 1.25)


def test_a_sweep_of_nothing_is_refused():
    from bartorch import optim

    with pytest.raises(ValueError, match="no projections"):
        optim.POCS([], maxiter=1)(torch.zeros(2, dtype=torch.complex64))


def test_a_projection_has_to_be_one():
    from bartorch import optim

    with pytest.raises(TypeError, match="callable on a tensor"):
        optim.POCSBlock([3])
