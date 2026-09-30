"""The commands an app answers, and what each of BART's flags means to it.

Each adapter reads a command line into one :class:`Call` of the app and says
what the command's output files hold of the app's answer.  An argument missing
from an adapter sends the whole command line back to BART rather than being
ignored.

``pics`` is held to BART's bits: its app configures BART's own iteration over
BART's own operators.  ``mobafit`` and ``moba`` are held to a tolerance: their
apps fit TorchSim's models, in TorchSim's bounded parameterisation, so they
reach the minimum the command reaches by a different path and answer to
round-off of the fit rather than to the bit.  Their adapters convert the named
maps back into BART's coefficients -- rates in 1/s from times in seconds, the
same stacking along ``COEFF_DIM`` and the same data scaling -- and route only
the models that parameterisation expresses exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from bartorch.cli._argv import Unsupported, axes, regularizer

__all__ = ["ADAPTERS", "Call"]


@dataclass
class Call:
    """One call of an app, and the arrays the command would have written.

    Attributes
    ----------
    arguments : tuple
        Positional arguments of the app.
    keywords : dict
        Keyword arguments of the app.
    written : callable
        The app's answer to the arrays of the command's output files, in the
        order the command line names them and in this package's layout.
    """

    arguments: tuple
    keywords: dict = field(default_factory=dict)
    written: Callable[[Any], list[torch.Tensor]] = lambda answer: [answer]


def _only(values: list[Any]) -> Any:
    """A flag BART takes once, as it was given."""
    return values[-1]


def _pics(options: dict[str, list[Any]], inputs: list, outputs: int) -> Call:
    """``pics``'s flags as :func:`bartorch.apps.pics` takes them."""
    from bartorch import priors
    from bartorch.apps._pics import image_shape

    ndim = len(image_shape(inputs[0], inputs[1], options.get("t", [None])[-1]))
    made: dict[str, Any] = {}
    terms = [regularizer(text, ndim) for text in options.pop("R", [])]

    # `-l1` and `-l2` are one term apiece over the three transformed axes, and
    # they take the weight `-r` carries rather than one of their own
    # (`grecon/optreg.c:260`); a `-r` left over after that is the plain
    # Tikhonov penalty below.
    weight = options["r"][-1] if "r" in options else 0.0
    for kind in options.pop("l", []):
        if str(kind) == "1":
            terms.append(priors.Wavelet(axes(7, ndim), weight))
        elif str(kind) == "2":
            terms.append(priors.L2(weight))
        else:
            raise Unsupported(f"pics -l takes 1 or 2, got {kind!r}")
        options.pop("r", None)

    for flag, keyword in (
        ("i", "maxiter"),
        ("s", "step"),
        ("u", "admm_rho"),
        ("C", "cg_maxiter"),
        ("w", "scaling"),
        ("t", "traj"),
        ("p", "pattern"),
        ("B", "basis"),
        ("W", "initial"),
    ):
        if flag in options:
            made[keyword] = _only(options.pop(flag))
    if "r" in options:
        made["l2"] = _only(options.pop("r"))
    if "e" in options:
        options.pop("e")
        made["eigen_step"] = True
    if "no_toeplitz" in options:
        options.pop("no_toeplitz")
        made["toeplitz"] = False

    for arm in ("ist", "fista", "admm", "pridu"):
        if arm in options:
            options.pop(arm)
            made["solver"] = arm
    if "eulermaruyama" in options:
        raise Unsupported("-R eulermaruyama is not one of the app's iterations")

    if terms:
        made["regularizers"] = terms
    if options:
        raise Unsupported(f"pics {sorted(options)} is not something the app takes")
    if outputs != 1:
        raise Unsupported("pics writes one output")
    return Call(tuple(inputs), made)


# --- the signal models -----------------------------------------------------
#
# BART's closed-form models, each written in the variables of one TorchSim
# model.  BART's times are in seconds and its rates in 1/s; TorchSim's times
# are in milliseconds.
#
#   T  (M0, R2)       M0 exp(-t R2)                  MultiEcho:  A exp(-t / T2)
#   I  (M0, R1, c)    M0 (1 - exp(c) exp(-t R1))     IR:         A (1 - (1 + e) exp(-t / T1))
#   L  (Mss, M0, R1s) Mss - (Mss + M0) exp(-t R1s)   IR:         the same, Mss = A, M0 = A e
#
# so R = 1000 / T, exp(c) = 1 + e and M0 / Mss = e.  The IR model's inversion
# efficiency e is bounded to (-1, 9), which covers every c below ln 10 and
# every M0 / Mss up to nine, and starts at a perfect inversion.  In `L` the two
# amplitudes share a phase, which BART leaves free.

_EFFICIENCY = (-1.0, 9.0)
_COEFFICIENTS = {"T": 2, "I": 3, "L": 3}


def _model(mode: str, times: np.ndarray, voxels: tuple[int, ...]):
    from bartorch import nlop

    milliseconds = [1e3 * float(t) for t in times]
    if mode == "T":
        return nlop.MultiEcho(milliseconds, voxels)
    return nlop.InversionRecovery(
        milliseconds,
        voxels,
        unknown=("T1", "inv_efficiency"),
        bounds={"inv_efficiency": _EFFICIENCY},
    )


def _start(mode: str, given: tuple[float, ...] | None) -> dict[str, Any]:
    """BART's starting coefficients as the app's starting values.

    Raises
    ------
    Unsupported
        A start the model's bounds do not contain.
    """
    if given is None:
        return {} if mode == "T" else {"inv_efficiency": 1.0}
    values = _padded(given, _COEFFICIENTS[mode])
    if mode == "T":
        amplitude, rate = values
        start = {"amplitude": amplitude, "T2": _time(rate, (1.0, 1000.0))}
    elif mode == "I":
        amplitude, rate, c = values
        start = {"amplitude": amplitude, "T1": _time(rate, (10.0, 5000.0))}
        start["inv_efficiency"] = math.exp(c) - 1.0
    else:
        steady, amplitude, rate = values
        if steady == 0.0:
            raise Unsupported("an L start with Mss = 0 has no inversion efficiency")
        start = {"amplitude": steady, "T1": _time(rate, (10.0, 5000.0))}
        start["inv_efficiency"] = amplitude / steady
    low, high = _EFFICIENCY
    if "inv_efficiency" in start and not low < start["inv_efficiency"] < high:
        raise Unsupported(f"a start outside the inversion efficiency's bounds {_EFFICIENCY}")
    return start


def _padded(given: tuple[float, ...], count: int) -> list[float]:
    """``--init`` as the command reads it: zeros past the values given."""
    values = [float(v) for v in given[:count]]
    return values + [0.0] * (count - len(values))


def _time(rate: float, bounds: tuple[float, float]) -> float:
    """A rate in 1/s as a time in ms, strictly inside the model's bounds."""
    if rate <= 0.0 or not bounds[0] < 1e3 / rate < bounds[1]:
        raise Unsupported(f"a start rate of {rate} 1/s is outside the model's bounds {bounds} ms")
    return 1e3 / rate


def _coefficients(mode: str, maps: dict[str, torch.Tensor], scale: float = 1.0) -> torch.Tensor:
    """The fitted maps as BART's coefficients, stacked on a leading axis."""
    amplitude = maps["amplitude"] * scale
    if mode == "T":
        made = [amplitude, 1e3 / maps["T2"]]
    elif mode == "I":
        made = [amplitude, 1e3 / maps["T1"], torch.log1p(maps["inv_efficiency"])]
    else:
        made = [amplitude, amplitude * maps["inv_efficiency"], 1e3 / maps["T1"]]
    return torch.stack([m.to(torch.complex64) for m in made])


# --- water and fat -----------------------------------------------------------
#
# `-G` fits BART's multi-echo models in the variables of TorchSim's
# MultiGradientEchoSimulator.  BART's echo times are in seconds -- its fat
# spectrum is -- so R2* comes back in 1/s and fB0 in Hz.
#
#   m  BART                        TorchSim
#   0  (W, F, fB0)                 fat_fraction, fat_phase, B0
#   1  (W, F, R2*, fB0)            ... and T2star
#   2  (W, R2*W, F, R2*F, fB0)     (goes to the command)
#   3  (rho, R2*, fB0)             T2star, B0
#   4  (rho, fB0)                  B0
#
# with W = A (1 - f), F = A f exp(i phase), R2* = 1000 / T2star and
# fB0 = -B0: the off-resonance turns the other way round in TorchSim.  A
# bounded T2* keeps each step physical; a thousand milliseconds is an R2* of
# 1/s, below anything a gradient-echo train resolves.

_WATER_FAT = {
    0: ("fat_fraction", "fat_phase", "B0"),
    1: ("fat_fraction", "fat_phase", "T2star", "B0"),
    3: ("T2star", "B0"),
    4: ("B0",),
}
_T2STAR = (1.0, 1000.0)
#: The command's five steps are over coefficients it scales to order one; four
#: bounded and periodic variables take this many to settle.
_WATER_FAT_ITERATIONS = 60


def _water_fat_model(times, voxels, which: int, field_strength: float, spectrum: str):
    from torchsim.simulators import MultiGradientEchoSimulator

    from bartorch import nlop

    unknown = _WATER_FAT[which]
    bounds = {"fat_fraction": (0.0, 1.0), "T2star": _T2STAR}
    scale = {"fat_phase": 90.0, "B0": 50.0}
    sequence = MultiGradientEchoSimulator(
        TE=[1e3 * float(t) for t in times],
        field_strength=field_strength,
        fat_spectrum=spectrum,
    )
    return nlop.Bloch(
        sequence,
        *unknown,
        shape=voxels,
        bounds={name: bounds[name] for name in unknown if name in bounds},
        contrasts=len(times),
        **{name: scale[name] for name in unknown if name in scale},
    )


def _water_fat_start(which: int, given: tuple[float, ...] | None) -> dict[str, Any]:
    """The app's start, or BART's ``--init`` in the app's variables.

    Raises
    ------
    Unsupported
        A start the model's bounds do not contain.
    """
    unknown = _WATER_FAT[which]
    if given is None:
        start = {"fat_fraction": 0.5, "fat_phase": 0.0, "T2star": 30.0}
        return {name: value for name, value in start.items() if name in unknown} | {"B0": 0.0}
    values = _padded(given, len(unknown) + (0 if which < 3 else 1))
    if which == 0:
        water, fat, frequency = values
        rates = []
    elif which == 1:
        water, fat, rate, frequency = values
        rates = [rate]
    else:
        water, fat, *rates, frequency = values[0], 0.0, *values[1:]
    start: dict[str, Any] = {"B0": -frequency}
    if which < 3:
        total = abs(water) + abs(fat)
        fraction = 0.5 if total == 0.0 else abs(fat) / total
        if not 0.0 < fraction < 1.0:
            raise Unsupported("a start with no water or no fat has no phase between them")
        start |= {
            "amplitude": water / (1 - fraction),
            "fat_fraction": fraction,
            "fat_phase": 0.0 if fat * water >= 0 else 180.0,
        }
    else:
        start["amplitude"] = water
    for rate in rates:
        start["T2star"] = _time(rate, _T2STAR)
    return start


def _water_fat_coefficients(which: int, maps: dict[str, torch.Tensor]) -> torch.Tensor:
    amplitude = maps["amplitude"]
    frequency = -maps["B0"]
    if which >= 3:
        rates = [1e3 / maps["T2star"]] if which == 3 else []
        made = [amplitude, *rates, frequency]
    else:
        fraction = maps["fat_fraction"]
        water = amplitude * (1 - fraction)
        fat = amplitude * fraction * torch.exp(1j * torch.deg2rad(maps["fat_phase"]))
        rates = [1e3 / maps["T2star"]] if which == 1 else []
        made = [water, fat, *rates, frequency]
    return torch.stack([m.to(torch.complex64) for m in made])


# --- diffusion and Z-spectra ---------------------------------------------------
#
# `-D` is M0 exp(enc x) over an encoding with one column, which is -b: TorchSim's
# DiffusionSimulator is M0 exp(-1e-3 b D), so b = -1000 enc and D is BART's
# coefficient in whatever units the encoding's reciprocal carries.  `-M` is the
# Lorentzian Z-spectrum over the offsets the encoding lists, (M0, then amplitude,
# width and shift per pool), in the same order as TorchSim's properties.


def _diffusion_model(times, voxels):
    from torchsim.simulators import DiffusionSimulator

    from bartorch import nlop

    b = [-1e3 * float(t) for t in times]
    largest = max(abs(value) for value in b) or 1.0
    return nlop.Bloch(DiffusionSimulator(b=b), "D", shape=voxels, contrasts=len(b), D=1e3 / largest)


def _lorentzian_names(pools: int) -> tuple[str, ...]:
    return tuple(
        f"pool{index}_{what}"
        for index in range(1, pools + 1)
        for what in ("amplitude", "width", "shift")
    )


def _lorentzian_model(offsets, voxels, pools: int):
    from torchsim.simulators import LorentzianSimulator

    from bartorch import nlop

    # A line's width enters squared; bounding it positive keeps the fit on
    # the side the command's start puts it.
    names = _lorentzian_names(pools)
    return nlop.Bloch(
        LorentzianSimulator(pools, offsets=[float(t) for t in offsets]),
        *names,
        shape=voxels,
        contrasts=len(offsets),
        bounds={name: (0.0, None) for name in names if name.endswith("_width")},
    )


def _mode(options: dict[str, list[Any]], name: str, modes: tuple[str, ...]) -> str:
    chosen = [mode for mode in modes if mode in options]
    if len(chosen) != 1:
        raise Unsupported(f"{name} is routed for one of -{', -'.join(modes)}")
    for mode in chosen:
        options.pop(mode)
    return chosen[0]


# --- BART's layout ---------------------------------------------------------

_DIMS = 16
_TE, _COEFF = 5, 6


def _bart(tensor: torch.Tensor) -> np.ndarray:
    """A tensor in this package's layout as a BART-order array of all its dimensions."""
    array = tensor.detach().cpu().numpy().T
    if array.ndim > _DIMS:
        raise Unsupported("more dimensions than BART has")
    return array.reshape(array.shape + (1,) * (_DIMS - array.ndim))


def _ours(array: np.ndarray) -> torch.Tensor:
    """A BART-order array in this package's layout."""
    return torch.as_tensor(np.ascontiguousarray(array.T))


def _times(enc: torch.Tensor, contrasts: int) -> np.ndarray:
    """Times in seconds, one per contrast, from a file that varies along ``TE_DIM`` alone."""
    times = _bart(enc)
    shape = list(times.shape)
    if shape[_TE] != contrasts or any(n != 1 for i, n in enumerate(shape) if i != _TE):
        raise Unsupported(f"times of shape {times.shape} are not one per contrast along TE_DIM")
    times = times.reshape(-1)
    if np.any(times.imag != 0):
        raise Unsupported("complex times")
    return times.real


# --- mobafit ---------------------------------------------------------------


def _mobafit(options: dict[str, list[Any]], inputs: list, outputs: int) -> Call:
    """``mobafit``'s flags as :func:`bartorch.apps.mobafit` takes them.

    ``-i`` counts Gauss-Newton steps over BART's own coefficients, which
    TorchSim's bounded parameterisation needs several times more of, so it is
    not something the app takes; the app's own count applies.  ``--liniter``
    counts conjugate-gradient steps in both.  Without ``--init`` the app
    starts from its own values, since the command's start of zero is no
    relaxation time; ``-M`` has no such start and is routed with one.
    """
    pools = _only(options.pop("M")) if "M" in options else 0
    # With no model named the command fits the multi-echo one.
    named = [flag for flag in ("T", "I", "L", "G", "D") if flag in options]
    mode = "M" if pools else _mode(options, "mobafit", ("T", "I", "L", "G", "D")) if named else "G"
    which = _only(options.pop("m", [1]))
    if mode == "G" and which == 2:
        # Two decays that differ only by fat's share of the signal are nearly
        # one; the Gauss-Newton weight towards the middle of each bound walks
        # the pair off where the command's unbounded rates stay.
        raise Unsupported("mobafit -m 2 does not converge in the app's bounded variables")
    if mode == "G" and which not in _WATER_FAT:
        raise Unsupported(f"mobafit -m {which} is not a multi-echo model")
    field_strength = _only(options.pop("field_strength", [3.0]))
    spectrum = "middleton2009" if options.pop("fat_spec_0", False) else "hamilton2011"
    made: dict[str, Any] = {}
    if "a" in options:
        options.pop("a")
        made["magnitude"] = True
    if "liniter" in options:
        made["cg_maxiter"] = _only(options.pop("liniter"))
    given = _only(options.pop("init")) if "init" in options else None
    if options:
        raise Unsupported(f"mobafit {sorted(options)} is not something the app takes")
    if outputs != 1:
        raise Unsupported("the app computes no covariance")

    enc, images = inputs
    y = _bart(images)
    if y.shape[_COEFF] != 1:
        raise Unsupported("contrast images along COEFF_DIM are a basis's")
    contrasts = y.shape[_TE]
    times = _times(enc, contrasts)
    rest = np.moveaxis(y, _TE, 0)
    data = torch.as_tensor(rest.reshape(contrasts, -1)).to(torch.complex64)
    empty = data.abs().amax(dim=0) == 0
    voxels = (data.shape[1],)

    if mode == "G":
        model = _water_fat_model(times, voxels, which, field_strength, spectrum)
        start = _water_fat_start(which, given)
        made.setdefault("iterations", _WATER_FAT_ITERATIONS)

        def coefficients(maps):
            return _water_fat_coefficients(which, maps)
    elif mode == "D":
        model = _diffusion_model(times, voxels)
        amplitude, rate = _padded(given, 2) if given is not None else (1.0, 0.0)
        start = {"amplitude": amplitude, "D": rate}

        def coefficients(maps):
            return torch.stack([maps["amplitude"], maps["D"].to(torch.complex64)])
    elif mode == "M":
        if given is None:
            raise Unsupported("mobafit -M is routed with --init; a Z-spectrum has no neutral start")
        model = _lorentzian_model(times, voxels, pools)
        names = _lorentzian_names(pools)
        values = _padded(given, 1 + len(names))
        start = {"amplitude": values[0], **dict(zip(names, values[1:], strict=True))}

        def coefficients(maps):
            return torch.stack(
                [maps["amplitude"], *(maps[name].to(torch.complex64) for name in names)]
            )
    else:
        model = _model(mode, times, voxels)
        start = _start(mode, given)

        def coefficients(maps):
            return _coefficients(mode, maps)

    def written(maps: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        # A voxel with no data is written as zero, which is what the command
        # writes for a slice with none.
        made = torch.where(empty, 0, coefficients(maps)).numpy()
        made = made.reshape(made.shape[:1] + rest.shape[1:])
        return [_ours(np.moveaxis(made, 0, _COEFF))]

    return Call((data, model), {**made, **start}, written)


# --- moba ------------------------------------------------------------------


def _moba(options: dict[str, list[Any]], inputs: list, outputs: int) -> Call:
    """``moba``'s flags as :func:`bartorch.apps.moba` takes them.

    Routed on a Cartesian grid with ``-l2``, which is the regularization the
    app's Gauss-Newton step carries.  ``-i`` is not taken, for the reason
    :func:`_mobafit` gives, and neither is ``-C``, which counts the command's
    FISTA iterations rather than the app's conjugate-gradient steps.  The
    command fits the data scaled by ``--scale_data``, and by the inverse of
    its norm with ``--normalize_scaling``, against a pattern scaled likewise
    by ``--scale_psf``; its amplitudes are in those units, and so are the ones
    written here.
    """
    mode = _mode(options, "moba", ("T", "L"))
    if options.pop("l", None) != [2]:
        raise Unsupported("moba is routed with -l2; the app has no wavelet term on the maps")
    made: dict[str, Any] = {}
    for flag, keyword in (("j", "alpha_min"), ("reduction", "redu"), ("R", "redu")):
        if flag in options:
            made[keyword] = _only(options.pop(flag))
    if "sobolev_a" in options or "sobolev_b" in options:
        a = _only(options.pop("sobolev_a", [880.0]))
        b = _only(options.pop("sobolev_b", [32.0]))
        made["sobolev"] = (a, b)
    data_scale = _only(options.pop("scale_data", [1.0]))
    psf_scale = _only(options.pop("scale_psf", [1.0]))
    normalized = bool(options.pop("normalize_scaling", False))
    sensitivities = _only(options.pop("sens")) if "sens" in options else None
    if options:
        raise Unsupported(f"moba {sorted(options)} is not something the app takes")
    if outputs > 1 and sensitivities is not None:
        raise Unsupported("the command writes no sensitivities it was given")

    kspace, enc = inputs
    k = _bart(kspace)
    if k.shape[4] != 1 or any(n != 1 for n in k.shape[_TE + 1 :]):
        raise Unsupported(f"k-space of shape {k.shape} has more than coils and echoes")
    contrasts = k.shape[_TE]
    coil_shape = k.shape[:4]
    times = _times(enc, contrasts)
    # (x, y, z, coils, echoes) to (echoes, coils, z, y, x), without z on a slice.
    k = torch.as_tensor(np.ascontiguousarray(k.reshape(*k.shape[:4], contrasts).T))
    grid = tuple(k.shape[2:]) if k.shape[2] > 1 else tuple(k.shape[3:])
    k = k.reshape(contrasts, k.shape[1], *grid).to(torch.complex64)

    if sensitivities is not None:
        s = _bart(sensitivities)
        if s.shape[4:] != (1,) * (_DIMS - 4) or s.shape[:4] != coil_shape:
            raise Unsupported(f"sensitivities of shape {s.shape} are not the k-space's coils")
        sensitivities = torch.as_tensor(np.ascontiguousarray(s.reshape(s.shape[:4]).T))
        sensitivities = sensitivities.reshape(k.shape[1], *grid).to(torch.complex64)

    # What the command's amplitudes are in: its data is scaled by `s` and its
    # pattern by `q`, and its model applies the pattern's normal, `q^2 P`.
    pattern = (k != 0).any(dim=1)
    if normalized:
        image = _centred_inverse(k, len(grid))
        if sensitivities is not None:
            image = (sensitivities.conj() * image).sum(dim=1)
        data_scale /= float(image.norm())
        psf_scale /= math.sqrt(float(pattern.sum()))
    scale = data_scale / psf_scale**2

    def written(answer) -> list[torch.Tensor]:
        maps, coils = answer if isinstance(answer, tuple) else (answer, None)
        coefficients = _coefficients(mode, maps, scale).reshape(-1, *_spatial(grid))
        # (coefficients, z, y, x) is BART's (x, y, z, 1, 1, 1, coefficients).
        arrays = [coefficients.reshape(-1, 1, 1, 1, *_spatial(grid))]
        if coils is not None:
            arrays.append(coils.reshape(-1, *_spatial(grid)))
        return arrays

    model = _model(mode, times, grid)
    if sensitivities is not None:
        return Call((k, model, sensitivities), {**made, **_start(mode, None)}, written)
    if outputs > 1:
        made["return_sensitivities"] = True
    return Call((k, model), {**made, **_start(mode, None)}, written)


def _spatial(grid: tuple[int, ...]) -> tuple[int, ...]:
    return grid if len(grid) == 3 else (1, *grid)


def _centred_inverse(k: torch.Tensor, ndim: int) -> torch.Tensor:
    """BART's ``ifftuc`` over the trailing ``ndim`` axes."""
    dims = tuple(range(-ndim, 0))
    shifted = torch.fft.ifftshift(k, dim=dims)
    return torch.fft.fftshift(torch.fft.ifftn(shifted, dim=dims, norm="ortho"), dim=dims)


#: The commands an app answers, and the reader that makes its call.  A command
#: absent from here is BART's own.
ADAPTERS = {
    "pics": _pics,
    "mobafit": _mobafit,
    "moba": _moba,
}
