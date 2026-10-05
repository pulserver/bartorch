"""``bartorch.cli`` against the command line it serves.

A script that calls ``bart`` has to run against ``bartorch`` unchanged, so the
question a test here answers is whether the app route writes what the command
route writes, over the same argv and the same files: the same bits for
``pics``, and for ``mobafit`` and ``moba``, whose apps fit BlochSim's models,
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
from bartorch.priors._terms import ImageNIHT

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


#: Multi-gradient-echo series.  BART's fat is Hamilton et al.'s six peaks at
#: 3 T, with the echo times in seconds.
GRADIENT_ECHO_TIMES = 1.1e-3 * (np.arange(8) + 1)
_HAMILTON = [(-3.80, 0.086), (-3.40, 0.537), (-2.60, 0.165), (-1.94, 0.046), (-0.39, 0.052),
             (0.60, 0.114)]  # fmt: skip
_FAT = sum(
    a * np.exp(2j * np.pi * 42.57747892e6 * 3.0 * p * 1e-6 * GRADIENT_ECHO_TIMES)
    for p, a in _HAMILTON
)
WATER, FAT = _halves(0.9, 0.4) * PHASE, _halves(0.1, 0.5) * PHASE * np.exp(0.5j)
R2S, FAT_R2S, FB0 = _halves(30.0, 50.0), _halves(60.0, 80.0), _halves(20.0, -10.0)


def _gradient_echo(water, fat, water_rate, fat_rate, frequency):
    t = GRADIENT_ECHO_TIMES
    return (
        water[..., None] * np.exp(-t * water_rate[..., None])
        + fat[..., None] * _FAT * np.exp(-t * fat_rate[..., None])
    ) * np.exp(2j * np.pi * frequency[..., None] * t)


_NONE = np.zeros((FIT, FIT))
_FITS |= {
    "G0": (
        ["-G", "-m", "0"],
        GRADIENT_ECHO_TIMES,
        _gradient_echo(WATER, FAT, _NONE, _NONE, FB0),
        [WATER, FAT, FB0],
    ),
    "G1": (
        ["-G"],
        GRADIENT_ECHO_TIMES,
        _gradient_echo(WATER, FAT, R2S, R2S, FB0),
        [WATER, FAT, R2S, FB0],
    ),
    "G3": (
        ["-G", "-m", "3"],
        GRADIENT_ECHO_TIMES,
        _gradient_echo(WATER, _NONE, R2S, R2S, FB0),
        [WATER, R2S, FB0],
    ),
    "G4": (
        ["-G", "-m", "4"],
        GRADIENT_ECHO_TIMES,
        _gradient_echo(WATER, _NONE, _NONE, _NONE, FB0),
        [WATER, FB0],
    ),
    "G0 from a start": (
        ["-G", "-m", "0", "--init", "1:0.3:0"],
        GRADIENT_ECHO_TIMES,
        _gradient_echo(WATER, FAT, _NONE, _NONE, FB0),
        [WATER, FAT, FB0],
    ),
    "G1 from a start": (
        ["-G", "--init", "1:0.3:30:0"],
        GRADIENT_ECHO_TIMES,
        _gradient_echo(WATER, FAT, R2S, R2S, FB0),
        [WATER, FAT, R2S, FB0],
    ),
    "G3 from a start": (
        ["-G", "-m", "3", "--init", "1:30:0"],
        GRADIENT_ECHO_TIMES,
        _gradient_echo(WATER, _NONE, R2S, R2S, FB0),
        [WATER, R2S, FB0],
    ),
}

#: Diffusion: the encoding is -b in s/mm^2, D in mm^2/s.
B_VALUES = np.array([0.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0])
DIFFUSIVITY = _halves(0.8e-3, 1.6e-3)
#: One Lorentzian pool over offsets in ppm.
OFFSETS = np.linspace(-5.0, 5.0, 21)
DEPTH, WIDTH, SHIFT = _halves(0.8, 0.6), _halves(1.5, 2.5), _halves(0.1, -0.2)
_FITS |= {
    "D": (
        ["-D"],
        -B_VALUES,
        M0 * np.exp(-B_VALUES * DIFFUSIVITY[..., None]),
        [np.full(DIFFUSIVITY.shape, M0), DIFFUSIVITY],
    ),
    "M": (
        ["-M", "1", "--init", "1:0.7:2:0"],
        OFFSETS,
        M0
        * (
            1
            - DEPTH[..., None]
            * (WIDTH[..., None] / 2) ** 2
            / ((WIDTH[..., None] / 2) ** 2 + (OFFSETS - SHIFT[..., None]) ** 2)
        ),
        [np.full(DEPTH.shape, M0), DEPTH, WIDTH, SHIFT],
    ),
}


def _relative(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


@pytest.mark.parametrize("model", sorted(_FITS))
def test_mobafit_writes_the_commands_coefficients_to_the_tolerance_of_a_fit(
    model, tmp_path, monkeypatch
):
    """``(M0, R2)``, ``(M0, R1, c)`` and ``(Mss, M0, R1s)`` in 1/s along
    COEFF_DIM, from a BlochSim fit in milliseconds.  Measured: within 1e-05 of
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
        ["mobafit", "-P", "t", "y", "x"],
        ["mobafit", "-G", "--fB0-init", "t", "y", "x"],
        ["mobafit", "-G", "-m", "2", "t", "y", "x"],
        ["mobafit", "-G", "-m", "5", "t", "y", "x"],
        ["mobafit", "-G", "--init", "1:0:30:0", "t", "y", "x"],
        ["mobafit", "-M", "1", "t", "y", "x"],
        ["mobafit", "-T", "-i", "5", "t", "y", "x"],
        ["mobafit", "-T", "--init", "1:0", "t", "y", "x"],
        ["mobafit", "-T", "--scale", "1:10", "t", "y", "x"],
        ["mobafit", "-T", "t", "y", "x", "covariance"],
    ],
    ids=[
        "-P",
        "--fB0-init",
        "-m 2",
        "-m 5",
        "water and no fat",
        "Z-spectrum with no start",
        "-i",
        "R2 start of zero",
        "--scale",
        "covariance",
    ],
)
def test_a_mobafit_the_app_cannot_express_goes_to_the_command(line, tmp_path, monkeypatch):
    """The multi-echo gradient-echo models, BART's step count over its own
    coefficients, a start the bounded model has no image of, a preconditioning,
    and a covariance the app does not compute."""
    monkeypatch.chdir(tmp_path)
    flags, times, series, _ = _FITS["T"]
    _series(times, series)
    assert route(line[0], line[1:]) == ("command", None)


def test_a_magnitude_mobafit_writes_the_commands_coefficients(tmp_path, monkeypatch):
    """``-a`` fits the modulus of the model to the modulus of the data, so the
    phased series answers ``|M0|`` and ``R2``.  The command needs a start off
    zero, where the modulus has no gradient."""
    monkeypatch.chdir(tmp_path)
    _, times, series, (_, rate) = _FITS["T"]
    _series(times, series)
    line = ["mobafit", "-T", "-a", "--init", "1:10", "t", "y"]
    where, plan = route(line[0], [*line[1:], "x"])
    assert where == "app" and plan["call"].keywords["magnitude"] is True

    assert main([*line, "app"]) == 0
    code, _, failure = run_command([*line, "cmd"])
    assert code == 0, failure

    ours, theirs = readcfl("app"), readcfl("cmd")
    for coefficient, expected in enumerate([np.full(T2.shape, abs(M0)), rate]):
        assert _relative(np.abs(ours[..., coefficient].squeeze()), expected) < 1e-4
        assert _relative(np.abs(theirs[..., coefficient].squeeze()), expected) < 1e-4


def test_mobafits_conjugate_gradient_count_is_the_apps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, times, series, _ = _FITS["T"]
    _series(times, series)
    where, plan = route("mobafit", ["-T", "--liniter", "7", "t", "y", "x"])
    assert where == "app" and plan["call"].keywords["cg_maxiter"] == 7


def _written(name: str, array: np.ndarray) -> str:
    writecfl(name, array.astype(np.complex64))
    return name


@pytest.mark.parametrize(
    ("flags", "times", "images"),
    [
        (["-L", "--init", "0:1:1"], None, None),
        (["-I", "--init", "1:1:5"], None, None),
        (["-T"], np.ones((1, 1, 1, 1, 1, 8, 2)), None),
        (["-T"], (1 + 1j) * ECHO_TIMES.reshape((1,) * 5 + (-1,)), None),
        (["-T"], None, np.ones((FIT, FIT, 1, 1, 1, 1, 8))),
    ],
    ids=[
        "L start with no steady state",
        "I start past the efficiency bounds",
        "times along two axes",
        "complex times",
        "contrasts along COEFF_DIM",
    ],
)
def test_a_mobafit_whose_files_the_app_cannot_read_goes_to_the_command(
    flags, times, images, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _, echo_times, series, _ = _FITS["T"]
    _series(echo_times, series)
    t = "t" if times is None else _written("t_other", times)
    y = "y" if images is None else _written("y_other", images)
    assert route("mobafit", [*flags, t, y, "x"]) == ("command", None)


def test_an_array_with_more_dimensions_than_bart_has_is_unsupported():
    from bartorch.cli._apps import _bart

    with pytest.raises(_argv.Unsupported, match="more dimensions than BART has"):
        _bart(torch.zeros((1,) * 17))


def test_pics_writes_one_output():
    from bartorch.cli._apps import ADAPTERS

    with pytest.raises(_argv.Unsupported, match="one output"):
        ADAPTERS["pics"]({}, [torch.ones(1, 8, 8), torch.ones(1, 8, 8)], 2)


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


def _moba_kspace(
    times: np.ndarray, images: np.ndarray, coils: np.ndarray, undersampled: bool = True
) -> None:
    """``images`` of shape ``(x, y, contrasts)``, as ``k`` and ``t``."""
    coil_images = images[:, :, None, :] * coils[..., None]
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(coil_images, axes=(0, 1)), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )
    generator = np.random.default_rng(0)
    for contrast in range(len(times) if undersampled else 0):
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


def test_moba_look_locker_with_a_smooth_flip_angle_writes_the_commands_maps(tmp_path, monkeypatch):
    """``moba -P``: ``(M0, R1, FA)`` with the flip angle's efficiency fitted as
    a smooth map, against the command with the flags BART's own tests give it.
    Measured on this phantom: R1 6.2e-04 from the command's at the median and
    4.4e-03 at most, the flip angle 4.4e-03 and 7.2e-03, both about the
    command's own distance from the truth, and the product of amplitude and
    sensitivities 1.3e-03 from the command's."""
    monkeypatch.chdir(tmp_path)
    support, amplitude, coils = _disc_and_coils()
    repetition, flip = 0.005, 8.0
    times = repetition * np.arange(0, 300, 10)
    rate = 1 / MOBA_T1
    angle = np.deg2rad(flip)
    apparent = rate - np.log(np.cos(angle)) / repetition
    steady = (rate / apparent)[..., None]
    images = (amplitude * np.sin(angle))[..., None] * (
        steady - (1 + steady) * np.exp(-times * apparent[..., None])
    )
    _moba_kspace(times, images, coils, undersampled=False)
    flags = [
        "-P",
        "-l2",
        f"--seq=TR={repetition},FA={flip}",
        "-R3",
        "-j0.001",
        "--scale_data=5000",
        "--scale_psf=1000",
        "--normalize_scaling",
        "--other=b1-sobolev-a=44,b1-sobolev-b=10",
    ]
    where, plan = route("moba", [*flags, "k", "t", "x", "s"])
    assert where == "app" and plan["call"].keywords["smooth"] == {"B1": (44.0, 10.0)}

    assert main(["moba", *flags, "k", "t", "app", "app_sens"]) == 0
    code, _, failure = run_command(["moba", *flags, "k", "t", "cmd", "cmd_sens"])
    assert code == 0, failure

    ours, theirs = readcfl("app"), readcfl("cmd")
    assert ours.shape == theirs.shape == (MOBA, MOBA, 1, 1, 1, 1, 3)
    for index, truth in ((1, rate), (2, np.full_like(rate, flip))):
        fitted = ours[..., index].squeeze().real
        command = theirs[..., index].squeeze().real
        against_command = np.abs(fitted - command)[support] / command[support]
        against_truth = np.abs(fitted - truth)[support] / truth[support]
        assert np.median(against_command) < 5e-3 and against_command.max() < 2e-2
        assert np.median(against_truth) < 1e-2 and against_truth.max() < 3e-2

    def product(maps, sensitivities):
        return (maps[..., 0].squeeze()[..., None] * sensitivities.squeeze())[support]

    ours_product = product(ours, readcfl("app_sens"))
    theirs_product = product(theirs, readcfl("cmd_sens"))
    assert np.linalg.norm(ours_product - theirs_product) < 0.01 * np.linalg.norm(theirs_product)


def test_moba_water_and_fat_fits_the_echoes_they_were_made_from(tmp_path, monkeypatch):
    """``moba -G``: ``(W, F, R2*, fB0)`` with fB0 fitted as a smooth map.  The
    command itself does not converge on this phantom without the parameter
    scaling of ``--other pscale`` and the proximal iteration of ``-r``, which
    the app does not take, so the fit is held to the echoes alone.  The
    amplitudes are in the units of the coil images the command scales to a norm
    of a hundred; combined with the coils the k-space was made with, since the
    smooth estimated coils share each voxel's scale with them.  Measured: R2*
    4.6e-05 from the truth at most, fB0 6.7e-06 Hz, F / W 1.4e-06, and the
    combined water 5.7e-03 at the median and 1.6e-02 at most."""
    monkeypatch.chdir(tmp_path)
    support, _, coils = _disc_and_coils()
    x, y = np.meshgrid(np.linspace(-1, 1, MOBA), np.linspace(-1, 1, MOBA), indexing="ij")
    times = 1.1e-3 * (np.arange(8) + 1)
    peaks = [(-3.80, 0.086), (-3.40, 0.537), (-2.60, 0.165), (-1.94, 0.046), (-0.39, 0.052)]
    peaks.append((0.60, 0.114))
    fat_signal = sum(a * np.exp(2j * np.pi * 42.57747892 * 3.0 * ppm * times) for ppm, a in peaks)
    water = support * np.where(x < 0, 0.8, 0.5)
    fat = support * np.where(y < 0, 0.2, 0.5)
    rate = np.where(x < 0, 30.0, 50.0)
    frequency = 15.0
    images = (water[..., None] + fat[..., None] * fat_signal) * np.exp(
        (2j * np.pi * frequency - rate[..., None]) * times
    )
    _moba_kspace(times, images, coils, undersampled=False)
    where, plan = route("moba", ["-G", "k", "t", "x", "s"])
    assert where == "app" and plan["call"].keywords["smooth"] == {"B0": (222.0, 32.0)}

    assert main(["moba", "-G", "k", "t", "app", "app_sens"]) == 0
    ours = readcfl("app")
    assert ours.shape == (MOBA, MOBA, 1, 1, 1, 1, 4)
    assert readcfl("app_sens").shape == (MOBA, MOBA, 1, MOBA_COILS)

    fitted_rate = ours[..., 2].squeeze().real
    assert (np.abs(fitted_rate - rate)[support] / rate[support]).max() < 1e-3
    assert np.abs(ours[..., 3].squeeze().real - frequency)[support].max() < 0.01
    fitted_water, fitted_fat = ours[..., 0].squeeze(), ours[..., 1].squeeze()
    ratio = fitted_fat[support] / fitted_water[support]
    assert np.abs(ratio - fat[support] / water[support]).max() < 1e-4

    scale = 100 / np.linalg.norm(readcfl("k"))
    product = fitted_water[..., None] * readcfl("app_sens").squeeze()
    combined = (coils.conj() * product).sum(-1) / (np.abs(coils) ** 2).sum(-1)
    error = np.abs(combined / scale - water)[support] / water[support]
    assert np.median(error) < 0.01 and error.max() < 0.03


def test_moba_inversion_recovery_echoes_fit_what_they_were_made_from(tmp_path, monkeypatch):
    """``moba -D -m 6``: ``(Ms, M0, R1*, R2*, fB0)`` from an echo train after
    each of several inversion times, the inversion times along ``TE_DIM`` and
    the echoes along ``CSHIFT_DIM``.  The command does not converge on this
    phantom without its parameter scaling, so the fit is held to the signal
    alone.  Measured: R1 4e-04 from the truth at most, R2* 2e-03, fB0 1e-03 Hz,
    and M0 / Ms 4e-04 from the inversion's 0.9."""
    monkeypatch.chdir(tmp_path)
    support, amplitude, coils = _disc_and_coils()
    x, _ = np.meshgrid(np.linspace(-1, 1, MOBA), np.linspace(-1, 1, MOBA), indexing="ij")
    inversions = np.array([0.05, 0.2, 0.5, 1.0, 1.8, 3.0])
    echoes = 1.6e-3 * (np.arange(4) + 1)
    efficiency, frequency = 0.9, 15.0
    rate = np.where(x < 0, 30.0, 50.0)
    recovery = 1 - (1 + efficiency) * np.exp(-inversions / MOBA_T1[..., None])
    train = np.exp((2j * np.pi * frequency - rate[..., None]) * echoes)
    images = amplitude[..., None, None] * recovery[..., :, None] * train[..., None, :]
    coil_images = images[:, :, None] * coils[..., None, None]
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(coil_images, axes=(0, 1)), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )
    shape = (MOBA, MOBA, 1, MOBA_COILS, 1, len(inversions), 1, 1, 1, len(echoes))
    writecfl("k", kspace.astype(np.complex64).reshape(shape))
    writecfl("ti", inversions.astype(np.complex64).reshape((1,) * 5 + (-1,)))
    writecfl("te", echoes.astype(np.complex64).reshape((1,) * 9 + (-1,)))
    flags = ["-D", "-m", "6", "-l2", "--other=echo=te"]
    assert route("moba", [*flags, "k", "ti", "x"])[0] == "app"

    assert main(["moba", *flags, "k", "ti", "app"]) == 0
    ours = readcfl("app")
    assert ours.shape == (MOBA, MOBA, 1, 1, 1, 1, 5)

    fitted = ours.squeeze().real
    assert (np.abs(fitted[..., 2] - 1 / MOBA_T1)[support] * MOBA_T1[support]).max() < 5e-3
    assert (np.abs(fitted[..., 3] - rate)[support] / rate[support]).max() < 1e-2
    assert np.abs(fitted[..., 4] - frequency)[support].max() < 0.01
    ratio = (ours[..., 1] / ours[..., 0]).squeeze()[support]
    assert np.abs(ratio - efficiency).max() < 5e-3


@pytest.mark.parametrize(
    ("flags", "echo"),
    [(["-D", "-m", "6", "-l2"], None), (["-D", "-m", "1", "-l2"], "te"), (["-D", "-m", "6"], "te")],
    ids=["no echo times", "-m 1", "default l1"],
)
def test_a_moba_inversion_recovery_the_app_cannot_express_goes_to_the_command(
    flags, echo, tmp_path, monkeypatch
):
    """Without echo times the command names what is missing; ``-m`` below six
    is a gradient-echo model on another layout."""
    monkeypatch.chdir(tmp_path)
    writecfl("k", np.ones((MOBA, MOBA, 1, MOBA_COILS, 1, 3, 1, 1, 1, 2), dtype=np.complex64))
    writecfl("ti", np.ones((1, 1, 1, 1, 1, 3), dtype=np.complex64))
    writecfl("te", np.ones((1,) * 9 + (2,), dtype=np.complex64))
    other = [] if echo is None else [f"--other=echo={echo}"]
    assert route("moba", [*flags, *other, "k", "ti", "x"]) == ("command", None)


def _run(*line: str) -> None:
    code, _, err = run_command(list(line))
    assert code == 0, err


#: BART's own test-moba-bloch-irflash-psf, with -l2 for the command's default
#: wavelet term and without the step counts the app does not take.
_IR_FLASH = [
    "--bloch",
    "--sim",
    "STM",
    "--seq",
    "IR-FLASH,TR=0.005,TE=0.003,FA=6,Trf=0.00001,BWTP=4,pinv,ipl=0,ppl=0",
    "--other",
    "pscale=1:1:1:1,pinit=3:1:1:0",
    "-l2",
    "-R3",
    "-j0.001",
    "--scale_data=5000.",
    "--scale_psf=1000.",
    "--normalize_scaling",
]


def _nrmse(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(ours - reference) / np.linalg.norm(reference))


def test_moba_bloch_writes_the_commands_maps_on_barts_own_case(tmp_path, monkeypatch):
    """``moba --bloch`` on BART's IR-FLASH test case: ``(R1, M0, R2, B1)`` from
    a Look-Locker recovery made with a flip angle of 8 degrees and fitted with
    6, so B1 is 4 / 3, fitted as a smooth map.  R2 is held at ``pinit``, as the
    command holds it for IR-FLASH, and written as it.  The data was made with
    an R2 of 0.01 1/s rather than the 1 held, so both routes stand a little off
    the truth, and by the same amount.  Measured: R1 and B1 within 2e-03 of the
    command's, M0 times the coils within 1.3e-03, and R1 and B1 1.8e-02 and
    1.0e-02 from the truth, against the command's 1.7e-02 and 9.0e-03."""
    monkeypatch.chdir(tmp_path)
    _run("phantom", "-x16", "-c", "circ")
    signal = ["-I", "-F", "-r0.005", "-f8", "-n100", "--short-TR-LL-approx"]
    _run("signal", *signal, "-1", "1.25:1.25:1", "-2", "100:100:1", "signal")
    _run("fmac", "circ", "signal", "image")
    _run("fft", "3", "image", "k")
    _run("index", "5", "100", "index")
    _run("scale", "0.005", "index", "ti")
    where, plan = route("moba", [*_IR_FLASH, "k", "ti", "x"])
    assert where == "app" and plan["call"].keywords["smooth"] == {"B1": (440.0, 20.0)}

    assert main(["moba", *_IR_FLASH, "k", "ti", "app", "app_sens"]) == 0
    _run("moba", *_IR_FLASH, "k", "ti", "cmd", "cmd_sens")
    ours, theirs = readcfl("app"), readcfl("cmd")
    assert ours.shape == theirs.shape == (16, 16, 1, 1, 1, 1, 4)
    ours, theirs = ours.squeeze(), theirs.squeeze()
    circ = readcfl("circ").squeeze().real
    inside = circ > 0.5

    assert np.all(ours[..., 2] == 1)
    for index, truth, command, against_truth in ((0, 0.8, 5e-3, 2.5e-2), (3, 4 / 3, 5e-3, 1.5e-2)):
        fitted = ours[..., index].real * circ
        assert _nrmse(fitted, theirs[..., index].real * circ) < command
        assert _nrmse(fitted, truth * circ) < against_truth
    product = (ours[..., 1, None] * readcfl("app_sens").squeeze())[inside]
    reference = (theirs[..., 1, None] * readcfl("cmd_sens").squeeze())[inside]
    assert _nrmse(product, reference) < 5e-3


@pytest.mark.parametrize(
    "other",
    [
        "pscale=0:1:1:1",
        "pscale=1:0:1:1",
        "b0map=field",
        "b1map=field",
        "pinit=3:1:1:0.1,b1-sobolev-a=0",
    ],
    ids=["R1 held", "M0 held", "a B0 map", "a B1 map beside a fitted B1", "B1 start as k-space"],
)
def test_a_moba_bloch_the_app_cannot_express_goes_to_the_command(other, tmp_path, monkeypatch):
    """A held R1 or M0; an off-resonance map, which turns the sample off the
    axis the amplitude is read along; a B1 map beside a fitted B1; and a B1
    start without its Sobolev weighting, which the command takes as k-space."""
    monkeypatch.chdir(tmp_path)
    writecfl("k", np.ones((MOBA, MOBA, 1, MOBA_COILS, 1, 10), dtype=np.complex64))
    writecfl("ti", np.ones((1, 1, 1, 1, 1, 10), dtype=np.complex64))
    writecfl("field", np.ones((MOBA, MOBA), dtype=np.complex64))
    flags = [*_IR_FLASH[:5], "--other", other, "-l2"]
    assert route("moba", [*flags, "k", "ti", "x"]) == ("command", None)


@pytest.mark.parametrize(
    "seq",
    [
        "IR-FLASH,TR=0.005,TE=0.003,FA=6,Trf=0.00001,Nspins=10",
        "IR-FLASH,TR=0.005,TE=0.003,FA=6,Trf=0.00001,av-spokes=2",
        "IR-FLASH,TR=0.005,TE=0.003,FA=6,Trf=0.00001,slice-thickness=0.002",
        "IR-FLASH,TR=0.005,TE=0.00001,FA=6,Trf=0.001",
        "IR-BSSFP,TR=0.0045,TE=0.00225,FA=45,Trf=0.00001,pinv,ipl=0,ppl=0.00225",
        "BSSFP,TR=0.0045,TE=0.00225,FA=45,Trf=0.00001",
        "TR=0.0045,TE=0.00225,FA=45,Trf=0.00001",
    ],
    ids=[
        "spins",
        "averaged spokes",
        "slice profile",
        "TE inside the pulse",
        "IR-bSSFP",
        "bSSFP",
        "the command's default train",
    ],
)
def test_a_moba_bloch_train_the_app_does_not_play_goes_to_the_command(seq, tmp_path, monkeypatch):
    """A slice profile, spins or spokes averaged, and a balanced SSFP train,
    which is the command's default: on BART's own IR-bSSFP case the app's T1
    and T2 have not settled after twenty steps from the command's start."""
    monkeypatch.chdir(tmp_path)
    writecfl("k", np.ones((MOBA, MOBA, 1, MOBA_COILS, 1, 10), dtype=np.complex64))
    writecfl("ti", np.ones((1, 1, 1, 1, 1, 10), dtype=np.complex64))
    flags = ["--bloch", "--seq", seq, "--other", "pinit=3:1:1:0", "-l2"]
    assert route("moba", [*flags, "k", "ti", "x"]) == ("command", None)


def test_a_moba_bloch_b1_map_rides_along_a_held_b1(tmp_path, monkeypatch):
    """``b1map`` is what the command's B1 multiplies, held where ``pscale`` holds
    it; the map written is the held offset alone, ``1 + pinit[3]``."""
    monkeypatch.chdir(tmp_path)
    writecfl("k", np.ones((MOBA, MOBA, 1, MOBA_COILS, 1, 10), dtype=np.complex64))
    writecfl("ti", np.ones((1, 1, 1, 1, 1, 10), dtype=np.complex64))
    writecfl("field", np.full((MOBA, MOBA), 0.8, dtype=np.complex64))
    flags = [*_IR_FLASH[:5], "--other", "pscale=1:1:1:0,pinit=3:1:1:0,b1map=field", "-l2"]
    where, plan = route("moba", [*flags, "k", "ti", "x"])
    assert where == "app" and "smooth" not in plan["call"].keywords


@pytest.mark.parametrize(
    "flags",
    [
        ["-T"],
        ["-T", "-l1"],
        ["-T", "-l2", "-i", "8"],
        ["-T", "-l2", "-C", "50"],
        ["-G", "-m", "2"],
        ["-G", "-r", "2"],
    ],
    ids=["default l1", "-l1", "-i", "-C", "-G separate decays", "-G -r"],
)
def test_a_moba_the_app_cannot_express_goes_to_the_command(flags, tmp_path, monkeypatch):
    """The wavelet term on the maps, BART's step counts over its own
    coefficients and its inner iteration, the separate water and fat decays,
    and the proximal iteration ``-r`` puts around the gradient-echo fit."""
    monkeypatch.chdir(tmp_path)
    support, amplitude, coils = _disc_and_coils()
    images = amplitude[..., None] * np.exp(-ECHO_TIMES[:6] / MOBA_T2[..., None])
    _moba_kspace(ECHO_TIMES[:6], images, coils)
    assert route("moba", [*flags, "k", "t", "x"]) == ("command", None)


def _moba_decay_with_coils():
    support, amplitude, coils = _disc_and_coils()
    images = amplitude[..., None] * np.exp(-ECHO_TIMES[:6] / MOBA_T2[..., None])
    _moba_kspace(ECHO_TIMES[:6], images, coils)
    writecfl("s", coils.astype(np.complex64).reshape(MOBA, MOBA, 1, MOBA_COILS))
    return support


@pytest.mark.parametrize(
    "scaling", [[], ["--scale_data=100", "--normalize_scaling"]], ids=["unscaled", "normalized"]
)
def test_moba_given_the_coils_fits_the_maps_they_were_made_with(scaling, tmp_path, monkeypatch):
    """``moba --sens``: the coils are the ones the k-space was made with, so the
    fit is of the maps alone.  Held to the truth: the command itself does not
    converge on this phantom with its coils given (its R2 comes back a quarter
    of the truth unscaled and zero scaled)."""
    monkeypatch.chdir(tmp_path)
    support = _moba_decay_with_coils()
    _, amplitude, _ = _disc_and_coils()
    flags = ["-T", "-l2", "--sens", "s", *scaling]
    where, plan = route("moba", [*flags, "k", "t", "x"])
    assert where == "app" and len(plan["call"].arguments) == 3

    assert main(["moba", *flags, "k", "t", "app"]) == 0

    ours = readcfl("app")
    assert ours.shape == (MOBA, MOBA, 1, 1, 1, 1, 2)
    rate = ours[..., 1].squeeze().real
    against_truth = np.abs(rate - 1 / MOBA_T2)[support] * MOBA_T2[support]
    assert np.median(against_truth) < 1e-3 and against_truth.max() < 5e-3
    if not scaling:
        fitted = ours[..., 0].squeeze()[support]
        assert np.abs(fitted - amplitude[support]).max() < 1e-3


def test_mobas_regularization_schedule_and_sobolev_weight_reach_the_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _moba_decay_with_coils()
    flags = ["-T", "-l2", "-j", "0.01", "--reduction", "3", "--sobolev_a", "100"]
    where, plan = route("moba", [*flags, "k", "t", "x"])
    assert where == "app"
    assert plan["call"].keywords["alpha_min"] == 0.01
    assert plan["call"].keywords["redu"] == 3.0
    assert plan["call"].keywords["sobolev"] == (100.0, 32.0)


@pytest.mark.parametrize(
    ("flags", "files"),
    [
        (["-T", "-l2", "--sens", "s"], ["k", "t", "x", "sens_out"]),
        (["-T", "-l2", "--sens", "s_wrong"], ["k", "t", "x"]),
        (["-T", "-l2"], ["k_more", "t", "x"]),
    ],
    ids=["coils given and asked for", "coils of another shape", "k-space beyond echoes"],
)
def test_a_moba_whose_files_the_app_cannot_take_goes_to_the_command(
    flags, files, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _moba_decay_with_coils()
    writecfl("s_wrong", np.ones((MOBA, MOBA, 1, MOBA_COILS + 1), dtype=np.complex64))
    writecfl("k_more", np.ones((MOBA, MOBA, 1, MOBA_COILS, 2, 6), dtype=np.complex64))
    assert route("moba", [*flags, *files]) == ("command", None)


def test_a_command_with_no_app_runs_as_itself(_dataset):
    assert route("fft", ["-i", "6", "ksp", "img"]) == ("command", None)
    assert main(["fft", "-i", "6", "ksp", "img"]) == 0
    assert readcfl("img").shape == (SIZE, SIZE, 1, COILS)


def test_one_process_writes_the_headers_of_many_files(tmp_path, monkeypatch):
    """A header is a few dozen formatted writes to the file's descriptor, and
    the Windows C runtime holds a fixed number of streams per process, so a
    write that opened one over the descriptor would fail a few dozen files in."""
    monkeypatch.chdir(tmp_path)
    for index in range(64):
        code, _, failure = run_command(["ones", "3", "2", "2", "2", f"x{index}"])
        assert code == 0, f"file {index}: {failure}"
    assert readcfl("x63").shape == (2, 2, 2)


def test_an_input_that_is_not_there_is_named_before_either_route_runs(_dataset, capsys):
    """The first line would reach the app, which reads its inputs in Python,
    and the second BART, which reports a missing input itself; both are named
    the same way."""
    assert main(["pics", "nosuchfile", "maps", "out"]) == 1
    assert "no such input: nosuchfile" in capsys.readouterr().err

    assert main(["pics", "-t", "nosuchtraj", "ksp", "maps", "out"]) == 1
    assert "no such input: nosuchtraj" in capsys.readouterr().err

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
