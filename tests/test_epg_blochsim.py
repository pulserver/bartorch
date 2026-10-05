"""BlochSim's simulators against the sequences BART's ``epg`` plays.

BART's ``epg`` evolves the extended phase graph of six sequences, and BlochSim
plays each with one of its simulators: CPMG as ``FSESimulator``, fmSSFP as
``fmSSFPSimulator``, Hyperecho as ``HyperechoSimulator``, FLASH as
``FLASHSimulator`` with instantaneous pulses and the sample at the pulse,
bSSFP as ``TrueFISPSimulator`` with instantaneous pulses, and Spinecho as
``StimulatedEchoSimulator``.  Both evolve the same configuration states, so
the two agree to single-precision round-off, derivatives included.

The conventions differ, and are mapped here once.  BART takes seconds and an
off-resonance of the opposite sign to BlochSim's ``B0``.  BART turns a sample
forward by the receiver phase where BlochSim turns it back, so an RF-spoiled
FLASH sample of BART's is BlochSim's times ``exp(2 i phi_n)``; its balanced
trains are played about axes that make each sample ``i`` times BlochSim's.
BART's Spinecho samples the spin echo of two pulses ``TE`` apart and the
stimulated echo of a third ``TE`` after the spin echo, which BlochSim times as
an echo time of ``2 TE`` and a mixing time of ``3 TE``.  BART's derivatives
are ``T1 dS/dT1``, ``T2 dS/dT2``, ``B1^2 dS/dB1`` and ``dS/d(off)``: the B1
term is relative twice, once where the pulse matrix is differentiated with
respect to the flip angle times the flip angle (``simu/epg.c``) and once more
where ``epg.c`` scales it by B1.
"""

import math

import pytest
import torch
from blochsim import simulators

from bartorch import _dispatch

T1_S, T2_S, B1, OFF_HZ = 0.8, 0.05, 0.9, 17.0
TR_S, TE_S, FLIP = 0.005, 0.01, 30.0
SHOTS = 10
TISSUE = {
    "T1": torch.tensor(1e3 * T1_S),
    "T2": torch.tensor(1e3 * T2_S),
    "B1": torch.tensor(B1),
    "B0": torch.tensor(-OFF_HZ),
}
#: BART's derivatives in BlochSim's, per unit of BlochSim's T1, T2, B1 and B0.
DERIVATIVE_SCALE = torch.tensor([1e3 * T1_S, 1e3 * T2_S, B1 * B1, -1.0])[:, None]


def _epg(sequence, outputs=1, **options):
    """BART's signal and, with ``outputs=3``, its derivatives."""
    tissue = dict(flag_1=T1_S, flag_2=T2_S, b=B1, o=OFF_HZ)
    out = _dispatch.dispatch(
        "epg", [], None, _n_out=outputs, **{sequence: True}, **tissue, **options
    )
    if outputs == 1:
        return out.reshape(-1)
    return out[0].reshape(-1), out[2].reshape(4, -1)


def _spoiling_phase(shots, increment_deg=50.0):
    shot = torch.arange(shots, dtype=torch.float64)
    phase = math.radians(increment_deg) * shot * (shot + 1) / 2
    return torch.exp(2j * phase).to(torch.complex64)


#: BART's options, BlochSim's simulator, and what BlochSim's sample is times
#: to be BART's.
DIFFERENTIATED = {
    "CPMG": (
        "C",
        dict(n=SHOTS, e=TE_S, f=120.0),
        lambda: simulators.FSESimulator(
            flip=120.0 * torch.ones(SHOTS), ESP=1e3 * TE_S, states=2 * SHOTS + 1
        ),
        1.0,
    ),
    "fmSSFP": (
        "M",
        dict(n=SHOTS, r=TR_S, f=FLIP),
        lambda: simulators.fmSSFPSimulator(flip=FLIP, TR=1e3 * TR_S, nshots=SHOTS),
        1j,
    ),
    "FLASH": (
        "F",
        dict(n=SHOTS, r=TR_S, f=FLIP, s=0),
        lambda: simulators.FLASHSimulator(
            flip=FLIP, TR=1e3 * TR_S, TE=0.0, nshots=SHOTS, pulse_duration=0.0
        ),
        1.0,
    ),
    "RF-spoiled FLASH": (
        "F",
        dict(n=SHOTS, r=TR_S, f=FLIP, s=1),
        lambda: simulators.FLASHSimulator(
            flip=FLIP,
            TR=1e3 * TR_S,
            TE=0.0,
            nshots=SHOTS,
            pulse_duration=0.0,
            rf_spoiling=50.0,
        ),
        _spoiling_phase(SHOTS),
    ),
    "bSSFP": (
        "B",
        dict(n=SHOTS, r=TR_S, f=FLIP),
        lambda: simulators.TrueFISPSimulator(
            flip=FLIP, TR=1e3 * TR_S, nshots=SHOTS, pulse_duration=0.0
        ),
        1j,
    ),
}

#: BART answers these without derivatives.
UNDIFFERENTIATED = {
    "Hyperecho": (
        "H",
        dict(n=2 * 5 + 1, e=TE_S, f=FLIP),
        lambda: simulators.HyperechoSimulator(flip=FLIP * torch.ones(5), ESP=1e3 * TE_S, TR=1e15),
        1.0,
    ),
    "Spinecho": (
        "S",
        dict(n=2, e=TE_S, f=FLIP),
        lambda: simulators.StimulatedEchoSimulator(flip=FLIP, TE=2e3 * TE_S, TM=3e3 * TE_S),
        1.0,
    ),
}


def _close(signal, expected, peak, tolerance=1e-5):
    error = float((torch.as_tensor(signal) - expected).abs().max() / peak)
    assert error < tolerance, error


@pytest.mark.parametrize("name", [*DIFFERENTIATED, *UNDIFFERENTIATED])
def test_blochsim_plays_the_signal_bart_s_epg_plays(name):
    sequence, options, simulator, factor = {**DIFFERENTIATED, **UNDIFFERENTIATED}[name]
    expected = _epg(sequence, **options)
    signal = factor * torch.as_tensor(simulator().simulate(**TISSUE))
    _close(signal, expected, expected.abs().max())


@pytest.mark.parametrize("name", list(DIFFERENTIATED))
def test_blochsim_differentiates_the_signal_as_bart_s_epg_does(name):
    sequence, options, simulator, factor = DIFFERENTIATED[name]
    expected, derivatives = _epg(sequence, outputs=3, **options)
    _, jacobian = simulator().jacobian(["T1", "T2", "B1", "B0"], **TISSUE)
    _close(factor * jacobian * DERIVATIVE_SCALE, derivatives, expected.abs().max())
