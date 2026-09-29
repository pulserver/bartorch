"""``bartorch.cli`` against the command line it serves.

A script that calls ``bart`` has to run against ``bartorch`` unchanged, so the
question a test here answers is whether the app route writes what the command
route writes, over the same argv and the same files: the same bits for
``pics``, and for ``mobafit`` and ``moba``, whose apps fit TorchSim's models,
the same files to a stated tolerance, both held to the rates the data was
made from.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import bartorch.tools as bt
from bartorch._dispatch import run_command
from bartorch.cli import _argv, main, route
from bartorch.io import readcfl, writecfl
from bartorch.priors.terms import ImageNIHT

SIZE, COILS, ACCEL = 24, 4, 2


@pytest.fixture
def _dataset(tmp_path, monkeypatch):
    """An under-sampled k-space and its sensitivities, as CFL pairs."""
    monkeypatch.chdir(tmp_path)
    kspace = bt.phantom(SIZE, coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    mask = torch.zeros(SIZE, dtype=torch.complex64)
    mask[::ACCEL] = 1
    mask[SIZE // 2 - 2 : SIZE // 2 + 2] = 1
    writecfl("ksp", (kspace * mask.reshape(SIZE, 1)).numpy().T)
    writecfl("maps", maps.numpy().T)
    return tmp_path


#: Command lines a script would really contain, each run both ways.
_LINES = [
    ["pics", "-l1", "-r", "0.01", "-i", "20"],
    ["pics", "-R", "W:7:0:0.01", "-i", "20"],
    ["pics", "-R", "T:6:0:0.005", "-i", "20"],
    ["pics", "-r", "0.1", "-i", "20"],
    ["pics", "-l2", "-r", "0.05", "-i", "20"],
    ["pics", "-R", "W:3:0:0.01", "-i", "20"],
]


@pytest.mark.parametrize("line", _LINES, ids=[" ".join(line) for line in _LINES])
def test_the_app_route_answers_what_the_command_answers(line, _dataset):
    where, _ = route(line[0], [*line[1:], "ksp", "maps", "x"])
    assert where == "app", f"{' '.join(line)} did not reach the app"

    assert main([*line, "ksp", "maps", "out_app"]) == 0
    code, _, failure = run_command([*line, "ksp", "maps", "out_cmd"])
    assert code == 0, failure

    assert np.array_equal(readcfl("out_app"), readcfl("out_cmd"))


def test_an_argument_the_reader_cannot_express_goes_to_the_command(_dataset):
    """``pics -c`` is the real-value constraint, which the app does not take;
    the command does, so it runs and the answer is still BART's."""
    where, _ = route("pics", ["-c", "-i", "10", "ksp", "maps", "x"])
    assert where == "command"

    assert main(["pics", "-c", "-i", "10", "ksp", "maps", "out"]) == 0
    assert readcfl("out").shape == (SIZE, SIZE)


# --- mobafit ---------------------------------------------------------------
#
# Series written out in closed form, in BART's layout: images `(x, y, 1, 1, 1,
# contrasts)` and times in seconds along TE_DIM.  Each fit is held to the
# command's coefficients and to the coefficients the series was made from.

FIT = 12
ECHO_TIMES = 0.0125 * (np.arange(8) + 1)
INVERSION_TIMES = np.array([0.05, 0.15, 0.4, 0.9, 1.5, 2.5, 4.0])
PHASE = np.exp(0.3j)


def _halves(left: float, right: float, size: int = FIT) -> np.ndarray:
    values = np.full((size, size), left)
    values[size // 2 :] = right
    return values


def _series(times: np.ndarray, series: np.ndarray) -> None:
    """``series`` of shape ``(x, y, contrasts)`` and its times, as ``t`` and ``y``."""
    writecfl("t", times.astype(np.complex64).reshape((1,) * 5 + (-1,)))
    writecfl("y", series.astype(np.complex64).reshape(FIT, FIT, 1, 1, 1, -1))


T2 = _halves(0.06, 0.11)
T1 = _halves(0.8, 1.4)
M0, MSS = 2.0 * PHASE, 0.8 * PHASE

#: flags, times, the series, and BART's coefficients for it.
_FITS = {
    "T": (
        ["-T"],
        ECHO_TIMES,
        M0 * np.exp(-ECHO_TIMES / T2[..., None]),
        [np.full(T2.shape, M0), 1 / T2],
    ),
    "I": (
        ["-I", "--init", "1:1:1"],
        INVERSION_TIMES,
        M0 * (1 - 2 * np.exp(-INVERSION_TIMES / T1[..., None])),
        [np.full(T1.shape, M0), 1 / T1, np.full(T1.shape, np.log(2))],
    ),
    "L": (
        ["-L", "--init", "0.6:1:0.8"],
        INVERSION_TIMES,
        MSS - (MSS + M0 / 2) * np.exp(-INVERSION_TIMES / T1[..., None]),
        [np.full(T1.shape, MSS), np.full(T1.shape, M0 / 2), 1 / T1],
    ),
}


def _relative(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


@pytest.mark.parametrize("model", sorted(_FITS))
def test_mobafit_writes_the_commands_coefficients_to_the_tolerance_of_a_fit(
    model, tmp_path, monkeypatch
):
    """``(M0, R2)``, ``(M0, R1, c)`` and ``(Mss, M0, R1s)`` in 1/s along
    COEFF_DIM, from a TorchSim fit in milliseconds.  Measured: within 1e-05 of
    each coefficient's peak against the command and against the series' own."""
    monkeypatch.chdir(tmp_path)
    flags, times, series, truth = _FITS[model]
    _series(times, series)
    assert route("mobafit", [*flags, "t", "y", "x"])[0] == "app"

    assert main(["mobafit", *flags, "t", "y", "app"]) == 0
    code, _, failure = run_command(["mobafit", *flags, "t", "y", "cmd"])
    assert code == 0, failure

    ours, theirs = readcfl("app"), readcfl("cmd")
    assert ours.shape == theirs.shape == (FIT, FIT, 1, 1, 1, 1, len(truth))
    for coefficient, expected in enumerate(truth):
        assert (
            _relative(ours[..., coefficient].squeeze(), theirs[..., coefficient].squeeze()) < 1e-4
        )
        assert _relative(ours[..., coefficient].squeeze(), expected) < 1e-4


@pytest.mark.parametrize(
    "line",
    [
        ["mobafit", "t", "y", "x"],
        ["mobafit", "-G", "t", "y", "x"],
        ["mobafit", "-T", "-i", "5", "t", "y", "x"],
        ["mobafit", "-T", "--init", "1:0", "t", "y", "x"],
        ["mobafit", "-T", "--scale", "1:10", "t", "y", "x"],
        ["mobafit", "-T", "t", "y", "x", "covariance"],
    ],
    ids=["default MGRE", "-G", "-i", "R2 start of zero", "--scale", "covariance"],
)
def test_a_mobafit_the_app_cannot_express_goes_to_the_command(line, tmp_path, monkeypatch):
    """The multi-echo gradient-echo models, BART's step count over its own
    coefficients, a start the bounded model has no image of, a preconditioning,
    and a covariance the app does not compute."""
    monkeypatch.chdir(tmp_path)
    flags, times, series, _ = _FITS["T"]
    _series(times, series)
    assert route(line[0], line[1:]) == ("command", None)


def test_a_mobafit_that_goes_to_the_command_answers_its_bits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, times, series, _ = _FITS["T"]
    _series(times, series)
    line = ["mobafit", "-T", "-i", "5", "t", "y"]
    assert main([*line, "ours"]) == 0
    code, _, failure = run_command([*line, "theirs"])
    assert code == 0, failure
    assert np.array_equal(readcfl("ours"), readcfl("theirs"))


# --- moba ------------------------------------------------------------------
#
# A disc of two relaxation halves behind four smooth coils, each contrast
# keeping its own random half of the phase encodes and the centre, as BART's
# k-space `(x, y, 1, coils, 1, contrasts)`: the coil images' centred unitary
# FFT, written with numpy.  With the coils estimated, the amplitude and the
# sensitivities share a scale the data does not fix, so what is compared is
# their product.

MOBA, MOBA_COILS = 16, 4
MOBA_T2 = _halves(0.06, 0.11, MOBA)
MOBA_T1 = _halves(0.8, 1.4, MOBA)


def _disc_and_coils():
    x, y = np.meshgrid(np.linspace(-1, 1, MOBA), np.linspace(-1, 1, MOBA), indexing="ij")
    support = x**2 + y**2 < 0.8
    amplitude = support * np.exp(0.5j * y)
    angles = np.arange(MOBA_COILS) * 2 * np.pi / MOBA_COILS
    coils = np.stack(
        [
            np.exp(-((x - 1.5 * np.cos(a)) ** 2 + (y - 1.5 * np.sin(a)) ** 2) / 18) * np.exp(1j * a)
            for a in angles
        ],
        axis=-1,
    )
    return support, amplitude, coils


def _moba_kspace(times: np.ndarray, images: np.ndarray, coils: np.ndarray) -> None:
    """``images`` of shape ``(x, y, contrasts)``, as ``k`` and ``t``."""
    coil_images = images[:, :, None, :] * coils[..., None]
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(coil_images, axes=(0, 1)), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )
    generator = np.random.default_rng(0)
    for contrast in range(len(times)):
        lines = np.zeros(MOBA)
        lines[generator.permutation(MOBA)[: MOBA // 2]] = 1
        lines[MOBA // 2 - 3 : MOBA // 2 + 3] = 1
        kspace[..., contrast] *= lines[None, :, None]
    writecfl("k", kspace.astype(np.complex64).reshape(MOBA, MOBA, 1, MOBA_COILS, 1, -1))
    writecfl("t", times.astype(np.complex64).reshape((1,) * 5 + (-1,)))


def test_moba_writes_the_commands_maps_to_the_tolerance_of_a_fit(tmp_path, monkeypatch):
    """``moba -T -l2``: ``(M0, R2)`` in the command's data scaling, and the
    sensitivities.  Measured on this phantom: R2 5.4e-03 from the command's at
    the median and 9.3e-03 at most, which is the command's own distance from
    the truth, and 1.4e-05 and 6.6e-05 from the truth; the product of
    amplitude and sensitivities 3.3e-03 from the command's."""
    monkeypatch.chdir(tmp_path)
    support, amplitude, coils = _disc_and_coils()
    images = amplitude[..., None] * np.exp(-ECHO_TIMES[:6] / MOBA_T2[..., None])
    _moba_kspace(ECHO_TIMES[:6], images, coils)
    flags = ["-T", "-l2", "--scale_data=100", "--normalize_scaling"]
    assert route("moba", [*flags, "k", "t", "x", "s"])[0] == "app"

    assert main(["moba", *flags, "k", "t", "app", "app_sens"]) == 0
    code, _, failure = run_command(["moba", *flags, "k", "t", "cmd", "cmd_sens"])
    assert code == 0, failure

    ours, theirs = readcfl("app"), readcfl("cmd")
    assert ours.shape == theirs.shape == (MOBA, MOBA, 1, 1, 1, 1, 2)
    assert readcfl("app_sens").shape == readcfl("cmd_sens").shape == (MOBA, MOBA, 1, MOBA_COILS)

    rate = ours[..., 1].squeeze().real
    command = theirs[..., 1].squeeze().real
    against_command = np.abs(rate - command)[support] / command[support]
    against_truth = np.abs(rate - 1 / MOBA_T2)[support] * MOBA_T2[support]
    assert np.median(against_command) < 0.02 and against_command.max() < 0.03
    assert np.median(against_truth) < 1e-3 and against_truth.max() < 5e-3

    def product(maps, sensitivities):
        return (maps[..., 0].squeeze()[..., None] * sensitivities.squeeze())[support]

    ours_product = product(ours, readcfl("app_sens"))
    theirs_product = product(theirs, readcfl("cmd_sens"))
    assert np.linalg.norm(ours_product - theirs_product) < 0.01 * np.linalg.norm(theirs_product)


def test_moba_look_locker_fits_the_recovery_it_was_made_from(tmp_path, monkeypatch):
    """``moba -L -l2``: ``(Mss, M0, R1s)`` in the command's layout.  The command
    itself does not converge on this phantom without the parameter scaling of
    ``--other pscale``, which the app does not take, so the fit is held to
    the recovery alone.  Measured: R1s 1.4e-03 from the truth at the median and
    1.0e-02 at most, and M0 / Mss 1.2502 against 1.25."""
    monkeypatch.chdir(tmp_path)
    support, amplitude, coils = _disc_and_coils()
    times = np.array([0.05, 0.2, 0.5, 1.0, 1.8, 3.0])
    images = amplitude[..., None] * (0.8 - 1.8 * np.exp(-times / MOBA_T1[..., None]))
    _moba_kspace(times, images, coils)
    flags = ["-L", "-l2", "--scale_data=100", "--normalize_scaling"]
    assert route("moba", [*flags, "k", "t", "x"])[0] == "app"

    assert main(["moba", *flags, "k", "t", "app"]) == 0
    code, _, failure = run_command(["moba", *flags, "k", "t", "cmd"])
    assert code == 0, failure
    ours = readcfl("app")
    assert ours.shape == readcfl("cmd").shape == (MOBA, MOBA, 1, 1, 1, 1, 3)

    rate = ours[..., 2].squeeze().real
    error = np.abs(rate - 1 / MOBA_T1)[support] * MOBA_T1[support]
    assert np.median(error) < 0.01 and error.max() < 0.05
    ratio = (ours[..., 1] / ours[..., 0]).squeeze()[support]
    assert abs(np.median(ratio.real) - 1.25) < 0.01


@pytest.mark.parametrize(
    "flags",
    [["-T"], ["-T", "-l1"], ["-T", "-l2", "-i", "8"], ["-T", "-l2", "-C", "50"], ["-G", "-l2"]],
    ids=["default l1", "-l1", "-i", "-C", "-G"],
)
def test_a_moba_the_app_cannot_express_goes_to_the_command(flags, tmp_path, monkeypatch):
    """The wavelet term on the maps, BART's step counts over its own
    coefficients and its inner iteration, and the gradient-echo models."""
    monkeypatch.chdir(tmp_path)
    support, amplitude, coils = _disc_and_coils()
    images = amplitude[..., None] * np.exp(-ECHO_TIMES[:6] / MOBA_T2[..., None])
    _moba_kspace(ECHO_TIMES[:6], images, coils)
    assert route("moba", [*flags, "k", "t", "x"]) == ("command", None)


def test_a_command_with_no_app_runs_as_itself(_dataset):
    assert route("fft", ["-i", "6", "ksp", "img"]) == ("command", None)
    assert main(["fft", "-i", "6", "ksp", "img"]) == 0
    assert readcfl("img").shape == (SIZE, SIZE, 1, COILS)


def test_an_input_that_is_not_there_is_named_before_bart_is_asked(_dataset, capsys):
    """BART would report it, and would be the one to ask -- except that a
    command which fails while loading its arguments leaves the library unable
    to serve the next call in the same process."""
    assert main(["pics", "nosuchfile", "maps", "out"]) == 1
    assert "no such input: nosuchfile" in capsys.readouterr().err

    assert main(["pics", "-t", "nosuchtraj", "ksp", "maps", "out"]) == 1
    assert "no such input: nosuchtraj" in capsys.readouterr().err

    # And the library still answers, which is the whole point.
    assert main(["fft", "-i", "6", "ksp", "img"]) == 0


def test_an_unknown_command_is_refused(capsys):
    assert main(["nosuchcommand"]) == 1
    assert "no command called" in capsys.readouterr().err


def test_help_comes_from_the_catalogue(capsys):
    """BART answers its own help by calling ``exit``, which in this process
    would end the interpreter."""
    assert main(["pics", "--help"]) == 0
    printed = capsys.readouterr().out
    assert "Parallel-imaging compressed-sensing reconstruction" in printed
    assert "max. number of iterations" in printed


def test_the_listing_names_every_command(capsys):
    from bartorch._catalogue import COMMANDS

    assert main([]) == 0
    printed = capsys.readouterr().out
    for name in ("pics", "ecalib", "nufft", "traj"):
        assert f"    {name}" in printed
    assert "bart " not in printed.split("Commands")[0].replace("`bart <command>`", "")
    assert len(COMMANDS) > 50


# --- the reader ------------------------------------------------------------


def test_short_flags_are_read_the_way_getopt_reads_them():
    """A value stuck to its letter, and value-less letters written together."""
    options, files = _argv.parse("pics", ["-i30", "-eS", "-r", "0.01", "k", "m", "o"])
    assert options == {"i": [30], "e": [True], "S": [True], "r": [0.01]}
    assert files == ["k", "m", "o"]


def test_long_flags_take_their_value_either_way():
    joined, _ = _argv.parse("pics", ["--wavelet=haar", "k", "m", "o"])
    apart, _ = _argv.parse("pics", ["--wavelet", "haar", "k", "m", "o"])
    assert joined == apart == {"wavelet": ["haar"]}


def test_a_selector_says_which_arm_was_given():
    options, _ = _argv.parse("pics", ["--fista", "k", "m", "o"])
    assert options == {"fista": ["--fista"]}
    letter, _ = _argv.parse("pics", ["-I", "k", "m", "o"])
    assert letter == {"ist": ["-I"]}


def test_an_option_the_command_does_not_have_is_unsupported():
    with pytest.raises(_argv.Unsupported, match="not an option"):
        _argv.parse("pics", ["--nosuchoption", "k", "m", "o"])


def test_a_regularizer_string_is_the_term_it_names():
    from bartorch import priors

    term = _argv.regularizer("W:7:0:0.01", ndim=2)
    assert isinstance(term, priors.Wavelet)
    assert term.weight == 0.01

    assert isinstance(_argv.regularizer("Q:0.1", ndim=2), priors.L2)
    assert isinstance(_argv.regularizer("S", ndim=2), priors.NonNegative)
    assert isinstance(_argv.regularizer("N:3:0:12", ndim=2), ImageNIHT)


def test_a_bitmask_names_the_axes_the_image_has():
    """BART carries a two-dimensional image on three axes and filters the third
    out with ``md_nontriv_dims``; the image here does not have it at all, so
    ``-R W:7`` and ``-R W:3`` are the same term on a slice."""
    assert _argv.axes(7, 3) == (-1, -2, -3)
    assert _argv.axes(7, 2) == (-1, -2)
    assert _argv.axes(3, 2) == (-1, -2)
    assert _argv.axes(6, 3) == (-2, -3)


def test_a_term_this_package_does_not_offer_is_unsupported():
    with pytest.raises(_argv.Unsupported, match="not a term"):
        _argv.regularizer("TF:{graph}:0.1", ndim=2)
