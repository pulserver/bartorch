"""TorchSim's Bloch-simulated sequences against BART's ``sim``.

BART integrates the Bloch-McConnell equations through each pulse with an ODE
solver; TorchSim plays each pulse as one hard pulse per sample.  Both play the
same sequences -- FLASH and balanced SSFP, each optionally after a hyperbolic
secant inversion, and pulsed CEST saturation -- over the free water and up to
four pools exchanging with it, and these tests hold each pair to what the two
integrations leave between them.

The conventions differ, and are mapped here once.  BART takes seconds, rates
in 1/s and frequencies in rad/s; TorchSim milliseconds and Hz, with the
opposite sign for an off-resonance or a chemical shift, and a complex sample
that is the conjugate of BART's.  BART's water has unit M0 and a pool's M0 is
relative to it, with ``k`` the rate back to the water; TorchSim's pools are
fractions of the total, with an exchange rate that the fractions split.  BART
measures TE from the start of the pulse, TorchSim from its centre, and BART's
secant inversion carries a constant phase that TorchSim takes as
``inversion_phase``.  BART's output is per pool, and the signal is their sum.
"""

import math

import pytest
import torch
from torchsim import simulators

from bartorch import _call

#: `sim` is private: what it returns is a curve, not a model a fit can be built
#: on.  It is reachable for exactly this.
sim = _call.build("sim", __name__)

GAMMA_BAR_MHZ_PER_T = 42.57747892
#: The constant phase of BART's secant inversion, mu log(B1 peak).
SECANT_PHASE_DEG = math.degrees(4.9 * math.log(14e-6))

#: T1 in s, T2 in s, chemical shift in rad/s, M0 relative to the water's and
#: exchange rate back to the water in 1/s, for pools B to E.
POOLS = (
    (1.0, 0.02, 600.0, 0.05, 30.0),
    (1.2, 0.01, -900.0, 0.02, 60.0),
    (0.8, 0.03, 1500.0, 0.03, 20.0),
    (1.5, 0.005, -300.0, 0.01, 100.0),
)
#: The same, with the chemical shift in ppm, for CEST.
CEST_POOLS = (
    (1.0, 0.01, 3.5, 0.01, 1000.0),
    (1.0, 0.02, 2.0, 0.005, 300.0),
    (1.3, 0.005, -3.5, 0.02, 30.0),
    (1.0, 0.1, 1.0, 0.003, 2000.0),
)


def _bart(seq, pools=(), tolerance="1e-6", t1=1.0, t2=0.1, **options):
    """BART's signal, summed over the pools."""
    count = len(pools) + 1
    arguments = dict(T1=(t1, t1, 1), T2=(t2, t2, 1), ODE=True, seq=seq)
    arguments["other"] = f"ode-tol={tolerance}"
    if pools:
        columns = [
            ":".join([str(pool[i]) for pool in pools] + ["0"] * (4 - len(pools))) for i in range(5)
        ]
        arguments["BMC"] = True
        arguments["pool"] = (
            f"P={count},T1={columns[0]},T2={columns[1]},Om={columns[2]},"
            f"M0={columns[3]},k={columns[4]}"
        )
    out = sim(**arguments, **options)
    if options.get("split_dim"):
        return out.reshape(count, -1, 3).sum(0)
    return out.reshape(count, -1).sum(0)


def _tissue(pools=(), shift_hz=lambda om: om / (2 * math.pi), **tissue):
    """TorchSim's properties for BART's water and pools."""
    total = 1.0 + sum(pool[3] for pool in pools)
    properties = {"T1": 1000.0, "T2": 100.0, "M0": total, **tissue}
    for letter, (t1, t2, shift, m0, exchange) in zip("BCDE"[: len(pools)], pools, strict=True):
        properties |= {
            f"pool{letter}_fraction": m0 / total,
            f"pool{letter}_exchange": exchange * total,
            f"pool{letter}_T1": 1e3 * t1,
            f"pool{letter}_T2": 1e3 * t2,
            f"pool{letter}_shift": -shift_hz(shift),
        }
    return properties


def _agree(ours, theirs, rtol):
    ours = torch.as_tensor(ours).reshape(-1)
    theirs = torch.as_tensor(theirs).reshape(-1)
    error = float((ours - theirs).abs().max() / theirs.abs().max())
    assert error <= rtol, error


SHOTS = 30


@pytest.mark.parametrize("off", [0.0, 150.0], ids=["on-resonance", "off-resonance"])
@pytest.mark.parametrize("count", [1, 2, 5], ids=["water", "two-pools", "five-pools"])
def test_flash_agrees_with_bart(count, off):
    pools = POOLS[: count - 1]
    theirs = _bart(f"FLASH,TR=0.005,TE=0.0025,Nrep={SHOTS},Trf=0.001,FA=30,BWTP=4,off={off}", pools)
    sequence = simulators.FLASHSimulator(flip=30.0, TR=5.0, TE=2.0, nshots=SHOTS)
    ours = sequence.simulate(**_tissue(pools, B0=-off / (2 * math.pi)))
    _agree(ours.conj(), theirs, 1e-4)


@pytest.mark.parametrize("off", [0.0, 150.0], ids=["on-resonance", "off-resonance"])
@pytest.mark.parametrize("count", [1, 2, 5], ids=["water", "two-pools", "five-pools"])
def test_balanced_ssfp_agrees_with_bart(count, off):
    pools = POOLS[: count - 1]
    theirs = _bart(
        f"BSSFP,TR=0.005,TE=0.0025,Nrep={SHOTS},Trf=0.001,FA=45,BWTP=4,ppl=0.0025,off={off}",
        pools,
    )
    sequence = simulators.TrueFISPSimulator(
        flip=45.0, TR=5.0, TE=2.0, nshots=SHOTS, preparation=2.5
    )
    ours = sequence.simulate(**_tissue(pools, B0=-off / (2 * math.pi)))
    _agree(ours.conj(), theirs, 2e-4)


@pytest.mark.parametrize("count", [1, 5], ids=["water", "five-pools"])
def test_inversion_recovery_flash_agrees_with_bart(count):
    pools = POOLS[: count - 1]
    theirs = _bart(f"IR-FLASH,TR=0.005,TE=0.0025,Nrep={SHOTS},Trf=0.001,FA=8,BWTP=4", pools)
    sequence = simulators.FLASHSimulator(
        flip=8.0,
        TR=5.0,
        TE=2.0,
        nshots=SHOTS,
        inversion="adiabatic",
        inversion_phase=SECANT_PHASE_DEG,
    )
    _agree(sequence.simulate(**_tissue(pools)).conj(), theirs, 5e-4)


@pytest.mark.parametrize("count", [1, 5], ids=["water", "five-pools"])
def test_inversion_recovery_balanced_ssfp_agrees_with_bart(count):
    pools = POOLS[: count - 1]
    theirs = _bart(
        f"IR-BSSFP,TR=0.005,TE=0.0025,Nrep={SHOTS},Trf=0.001,FA=45,BWTP=4,ppl=0.0025", pools
    )
    sequence = simulators.TrueFISPSimulator(
        flip=45.0,
        TR=5.0,
        TE=2.0,
        nshots=SHOTS,
        preparation=2.5,
        inversion="adiabatic",
        inversion_phase=SECANT_PHASE_DEG,
    )
    _agree(sequence.simulate(**_tissue(pools)).conj(), theirs, 5e-4)


@pytest.mark.parametrize("count", [2, 5], ids=["two-pools", "five-pools"])
def test_the_cest_z_spectrum_agrees_with_bart(count):
    # BART's solver is held to its default tolerance here: a tighter one takes
    # minutes over a saturation train.
    pools = CEST_POOLS[: count - 1]
    offsets = 11
    theirs = _bart(
        f"CEST,Trf=0.02,Nrep={offsets}",
        pools,
        tolerance="1e-5",
        t1=1.3,
        t2=0.075,
        split_dim=True,
        CEST=(f"b1=1.0,b0=3.0,gamma={GAMMA_BAR_MHZ_PER_T},min=-5,max=5,n_p=2,t_d=0.01,t_pp=0.0065"),
    )
    sequence = simulators.CESTSimulator(
        offsets=torch.linspace(-5.0, 5.0, offsets),
        B1sat=1.0,
        npulses=2,
        pulse_duration=20.0,
        interpulse_delay=10.0,
        recovery=6.5,
    )
    ours = sequence.simulate(
        **_tissue(pools, shift_hz=lambda ppm: ppm * 3.0 * GAMMA_BAR_MHZ_PER_T, T1=1300.0, T2=75.0)
    )
    _agree(ours, theirs[..., 2].real, 3e-4)
