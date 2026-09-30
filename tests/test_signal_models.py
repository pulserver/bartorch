"""TorchSim's closed forms against the signal models BART computes.

Every closed-form signal model BART ships -- the ``signal`` tool's and the
nonlinear operators ``mobasig`` evaluates for ``mobafit`` and ``moba`` -- has a
TorchSim simulator, and these tests hold each pair to single-precision
round-off.  The units are each library's own: BART takes seconds, rates in
1/s and ``fB0`` in Hz with a positive phase, TorchSim milliseconds and ``B0``
with the opposite sign.  BART's Look-Locker curves are the longitudinal
magnetization; TorchSim's carry the readout's ``sin(flip)``.
"""

import numpy as np
import pytest
import torch

from bartorch import _call

torchsim = pytest.importorskip("torchsim")
simulators = pytest.importorskip("torchsim.simulators")
if not hasattr(simulators, "MultiGradientEchoSimulator"):
    pytest.skip("this TorchSim has no closed forms for BART's models", allow_module_level=True)

#: Both commands are private: what they return is a curve, not a model a fit
#: can be built on.  They are reachable for exactly this.
signal = _call.build("signal", __name__)
mobasig = _call.build("mobasig", __name__)

T1_S, T2_S = 0.9, 0.08


def _agree(ours, theirs, rtol=2e-6):
    ours = np.asarray(ours).reshape(-1)
    theirs = np.asarray(theirs).reshape(-1)
    assert np.max(np.abs(ours - theirs)) <= rtol * np.max(np.abs(theirs))


def _coefficients(*values):
    return torch.tensor(values, dtype=torch.complex64).reshape(len(values), 1, 1, 1, 1, 1, 1)


def _encoding(values):
    return torch.tensor(np.asarray(values), dtype=torch.complex64).reshape(-1, 1, 1, 1, 1, 1)


# --- the signal tool ------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [{"I": True}, {"I": True, "short_TR_LL_approx": True}, {"s": True}],
    ids=["from-equilibrium", "short-TR", "from-steady-state"],
)
def test_the_look_locker_recovery_is_barts(flags):
    tr, flip, n = 0.005, 8.0, 60
    theirs = signal(F=True, r=tr, f=flip, n=n, flag_1=(T1_S, T1_S, 1), **flags)
    sequence = simulators.LookLockerSimulator(
        flip=flip,
        TR=tr * 1e3,
        TI=tr * 1e3 * np.arange(n),
        short_TR=flags.get("short_TR_LL_approx", False),
        from_steady_state=flags.get("s", False),
    )
    ours = sequence.simulate(T1=T1_S * 1e3) / np.sin(np.deg2rad(flip))
    _agree(ours, theirs)


def test_the_molli_blocks_are_barts():
    tr, flip = 0.005, 30.0
    theirs = signal(M=True, I=True, r=tr, f=flip, n=60, b=3, t=0.5, flag_1=(T1_S, T1_S, 1))
    sequence = simulators.MOLLISimulator(
        flip=flip, TR=tr * 1e3, readouts=(20, 20, 20), recovery=500.0
    )
    _agree(sequence.simulate(T1=T1_S * 1e3) / np.sin(np.deg2rad(flip)), theirs)


def test_the_ir_bssfp_recovery_is_barts():
    tr, flip, n = 0.0045, 45.0, 60
    theirs = signal(
        B=True, I=True, r=tr, f=flip, n=n, flag_1=(T1_S, T1_S, 1), flag_2=(T2_S, T2_S, 1)
    )
    sequence = simulators.IRbSSFPSimulator(flip=flip, TI=tr * 1e3 * np.arange(n))
    _agree(sequence.simulate(T1=T1_S * 1e3, T2=T2_S * 1e3), theirs)


def test_the_spin_echo_is_barts():
    theirs = signal(S=True, r=2.0, e=0.01, n=6, flag_1=(T1_S, T1_S, 1), flag_2=(T2_S, T2_S, 1))
    sequence = simulators.SpinEchoSimulator(TE=10.0 * np.arange(6), TR=2000.0)
    _agree(sequence.simulate(T1=T1_S * 1e3, T2=T2_S * 1e3), theirs)


def test_the_inversion_recovery_spin_echo_is_barts():
    # With -I, BART steps the inversion time and holds the echo time.
    theirs = signal(
        S=True, I=True, r=5.0, e=0.02, i=0.1, n=6,
        flag_1=(T1_S, T1_S, 1), flag_2=(T2_S, T2_S, 1),
    )  # fmt: skip
    sequence = simulators.SpinEchoSimulator(TE=20.0, TR=5000.0, TI=100.0 * np.arange(6))
    _agree(sequence.simulate(T1=T1_S * 1e3, T2=T2_S * 1e3), theirs)


@pytest.mark.parametrize("refocusing", [180.0, 120.0])
def test_the_stimulated_echo_train_is_barts_generating_function(refocusing):
    # BART's first sample is the excitation itself; the echoes follow it.
    theirs = signal(
        T=True, e=0.01, n=13, f=refocusing, flag_1=(1.0, 1.0, 1), flag_2=(T2_S, T2_S, 1)
    )
    sequence = simulators.FSESimulator(flip=refocusing * torch.ones(12), ESP=10.0, TR=1e6)
    ours = sequence.simulate(T1=1000.0, T2=T2_S * 1e3)
    _agree(np.abs(np.asarray(ours)), np.abs(np.asarray(theirs).reshape(-1)[1:]), rtol=1e-4)


def test_water_and_fat_are_barts():
    te, n = 0.0012, 8
    theirs = signal(
        G=True, fat=True, e=te, n=n, d=0.25, flag_2=(0.03, 0.03, 1), flag_0=(20.0, 20.0, 1)
    )
    sequence = simulators.MultiGradientEchoSimulator(TE=te * 1e3 * np.arange(n))
    _agree(sequence.simulate(fat_fraction=0.25, T2star=30.0, B0=-20.0), theirs)


@pytest.mark.parametrize("labelling", ["continuous", "pulsed"])
def test_the_arterial_spin_labelling_curve_is_barts(labelling):
    pulsed = labelling == "pulsed"
    theirs = signal(
        A=True, pulsed=pulsed, r=0.1, n=50, flag_1=(1.4, 1.4, 1), flag_6=(60.0, 60.0, 1)
    )
    sequence = simulators.ASLSimulator(
        t=100.0 * np.arange(50), tau=800.0 if pulsed else 1800.0, labelling=labelling
    )
    _agree(sequence.simulate(CBF=60.0, ATT=1800.0, T1=1400.0), theirs, rtol=2e-6)


# --- the operators mobafit and moba fit -------------------------------------------

_W, _F = 0.7 - 0.2j, 0.3 + 0.1j
_TE_S = 1e-3 * np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])


def _water_and_fat():
    """BART's complex W and F as a complex scale, a fat fraction and a fat phase."""
    fraction = abs(_F) / (abs(_W) + abs(_F))
    scale = _W / (1 - fraction)
    phase = np.rad2deg(np.angle(_F / (scale * fraction)))
    return scale, {"fat_fraction": fraction, "fat_phase": phase}


@pytest.mark.parametrize(
    ("model", "coefficients", "properties"),
    [
        (0, (_W, _F, 20.0), {"B0": -20.0}),
        (1, (_W, _F, 30.0, 20.0), {"T2star": 1e3 / 30.0, "B0": -20.0}),
        (2, (_W, 30.0, _F, 50.0, 20.0), {"T2star": 1e3 / 30.0, "fat_T2star": 1e3 / 50.0, "B0": -20.0}),
    ],
    ids=["WF", "WFR2S", "WF2R2S"],
)  # fmt: skip
def test_the_water_fat_models_are_barts(model, coefficients, properties):
    theirs = mobasig(_coefficients(*coefficients), _encoding(_TE_S), G=True, m=model)
    scale, fat = _water_and_fat()
    sequence = simulators.MultiGradientEchoSimulator(TE=_TE_S * 1e3)
    _agree(scale * np.asarray(sequence.simulate(**properties, **fat)), theirs)


@pytest.mark.parametrize(
    ("model", "coefficients", "properties"),
    [(3, (_W, 30.0, 20.0), {"T2star": 1e3 / 30.0, "B0": -20.0}), (4, (_W, 20.0), {"B0": -20.0})],
    ids=["R2S", "PHASEDIFF"],
)
def test_the_water_only_models_are_barts(model, coefficients, properties):
    theirs = mobasig(_coefficients(*coefficients), _encoding(_TE_S), G=True, m=model)
    sequence = simulators.MultiGradientEchoSimulator(TE=_TE_S * 1e3)
    _agree(_W * np.asarray(sequence.simulate(**properties)), theirs)


def test_the_other_fat_spectrum_at_another_field_is_barts():
    theirs = mobasig(
        _coefficients(_W, _F, 30.0, 20.0), _encoding(_TE_S),
        G=True, m=1, fat_spec_0=True, field_strength=1.5,
    )  # fmt: skip
    scale, fat = _water_and_fat()
    sequence = simulators.MultiGradientEchoSimulator(
        TE=_TE_S * 1e3, fat_spectrum="middleton2009", field_strength=1.5
    )
    _agree(scale * np.asarray(sequence.simulate(T2star=1e3 / 30.0, B0=-20.0, **fat)), theirs)


def test_the_diffusion_decay_is_barts():
    # mobasig -D is M0 exp(enc . x): the encoding is -b and the coefficient D.
    b = np.array([0.0, 500.0, 1000.0, 2000.0])
    theirs = mobasig(_coefficients(2.0, 0.8e-3), _encoding(-b), D=True)
    _agree(2.0 * np.asarray(simulators.DiffusionSimulator(b=b).simulate(D=0.8)), theirs)


def test_the_lorentzian_z_spectrum_is_barts():
    offsets = np.linspace(-5.0, 5.0, 21)
    pools = ((0.8, 1.5, 0.1), (0.1, 2.0, 3.5))
    theirs = mobasig(
        _coefficients(1.2, *(value for pool in pools for value in pool)),
        _encoding(offsets),
        M=True,
    )
    named = {}
    for index, (amplitude, width, shift) in enumerate(pools, start=1):
        named[f"pool{index}_amplitude"] = amplitude
        named[f"pool{index}_width"] = width
        named[f"pool{index}_shift"] = shift
    sequence = simulators.LorentzianSimulator(len(pools), offsets=offsets)
    _agree(1.2 * np.asarray(sequence.simulate(**named)), theirs)
