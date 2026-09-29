"""FINUFFT under BART's ``nufft_create``: the substitution's switches and counters.

FINUFFT, and cuFINUFFT in a CUDA build, are compiled into ``libbartorch``.  The
substitution installs itself on first use (:func:`install_once`).  Tests and
``scripts/check_device.py`` read the counters.
"""

from __future__ import annotations

import contextlib
import logging

import torch

Shape = tuple[int, ...]


def available() -> bool:
    """Whether ``libbartorch`` carries FINUFFT, which every build does."""
    from bartorch._lib import library

    return bool(library().bartorch_finufft_built_on(0))


def cuda_available() -> bool:
    """Whether ``libbartorch`` carries cuFINUFFT, which a CUDA build does."""
    from bartorch._lib import library

    return bool(library().bartorch_finufft_built_on(1))


def version() -> str:
    """The FINUFFT release the compiled-in sources declare."""
    from bartorch._lib import library

    return library().bartorch_finufft_version().decode()


def used_on_device() -> bool:
    """Whether a transform BART runs on a device would be computed by cuFINUFFT."""
    from bartorch._lib import library

    return bool(library().bartorch_finufft_usable_on(1))


def use_in_tools(
    enable: bool = True,
    tolerance: float = 1e-3,
    upsampling: float = 1.25,
) -> bool:
    """Install FINUFFT as the NUFFT behind BART's tools and operators.

    Parameters
    ----------
    enable : bool
        False restores BART's own gridder.
    tolerance : float
        Tolerance FINUFFT plans are made with.
    upsampling : float
        Oversampling of FINUFFT's fine grid; 0 lets FINUFFT choose per problem.
        BART's ``-o`` overrides it wherever it is not BART's default of 2.

    Returns
    -------
    bool
        Whether the substitution is in place.

    Raises
    ------
    RuntimeError
        FINUFFT disagrees with BART's gridder on a test transform.
    """
    from bartorch._lib import library

    lib = library()
    if not enable:
        lib.bartorch_finufft_use_in_tools(0)
        lib.bartorch_nufft_allow_fallback(1)
        return False

    lib.bartorch_finufft_set_tolerance(float(tolerance))
    lib.bartorch_finufft_set_upsampling(float(upsampling))
    lib.bartorch_nufft_allow_fallback(0)
    lib.bartorch_finufft_use_in_tools(1)

    if not _tools_agree_with_bart():
        lib.bartorch_finufft_use_in_tools(0)
        raise RuntimeError(
            "FINUFFT's NUFFT does not agree with BART's own; the substitution has been left off"
        )

    return bool(lib.bartorch_finufft_usable())


def used_in_tools() -> bool:
    """Whether BART's tools are computing their NUFFT with FINUFFT."""
    from bartorch._lib import library

    return bool(library().bartorch_finufft_usable())


def serves(device: bool = False) -> bool:
    """Whether the substitution answers ``nufft_create`` on this side.

    The condition ``nufft_finufft.c`` applies before it takes an operator at
    all: the host table, and the cuFINUFFT one as well for a transform BART
    would run on a card.  What a caller needs it for is a transform only the
    substitution can compute, so that the answer where it declines is a route
    BART's own gridder does have rather than an error.

    The substitution is installed first if nothing has needed one yet, so this
    answers what would happen and not merely what has happened.
    """
    from bartorch._lib import library

    install_once()

    lib = library()
    if not lib.bartorch_finufft_usable_on(0):
        return False
    return (not device) or bool(lib.bartorch_finufft_usable_on(1))


def stream_psf(enable: bool = True) -> None:
    """Stream the Toeplitz point spread function to the card one set of frequencies at a time.

    Uses BART's low-memory normal: less device memory, more time.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_stream_psf(int(bool(enable)))


def streaming_psf() -> bool:
    """Whether the function is being kept off the card."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_stream_psf())


def compress_psf(enable: bool = True) -> None:
    """Store a Toeplitz function only where the samples reach, beside an index of those points.

    Decided for each function when it is built: kept for a subspace function whose
    trajectory leaves enough of the grid unreached, never for a scalar function.
    The function is small but not zero elsewhere, so compression is approximate.
    :func:`functions_compressed` counts compressed functions.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_compress_psf(int(bool(enable)))


def compressing_psf() -> bool:
    """Whether only the places the samples reach are kept."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_compress_psf())


def overlap_psf(enable: bool = True) -> None:
    """Transfer the next set of frequencies to the card while the current one is convolved.

    Needs a second device slot of one set's size and a page-locked host copy.  Off
    by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_overlap_psf(int(bool(enable)))


def overlapping_psf() -> bool:
    """Whether a set crosses while the one before it is convolved."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_overlap_psf())


def _contraction_kernel(enable: bool = True) -> None:
    """Choose bartorch's or BART's kernel for the real upper-triangular contraction."""
    from bartorch._lib import library

    library().bartorch_nufft_set_contraction_kernel(int(bool(enable)))


def release_transforms(enable: bool = True) -> None:
    """Free the device's FINUFFT plans at the first normal application.

    With a Toeplitz function built, a normal reads neither the plans nor the sample
    positions; a later forward or adjoint plans again from the host trajectory.  On
    by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_release_transforms(int(bool(enable)))


def releasing_transforms() -> bool:
    """Whether the device's transform pair is let go at the first normal."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_release_transforms())


def fft_callbacks(enable: bool = True) -> None:
    """Run the per-volume phase, sensitivity, gather and scatter passes as cuFFT LTO callbacks.

    Applies to streamed, compressed sets.  Where cuFFT's LTO callbacks or nvJitLink
    are unavailable the passes run separately.  On by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_fft_callbacks(int(bool(enable)))


def pair_sets(enable: bool = True) -> None:
    """Convolve Toeplitz sets that differ only along x in pairs, sharing their z and y transforms.

    Uses the cuFFTDx pair kernels, compiled for fixed grid sizes.  Applies to a
    compressed real function kept as its upper triangle, with four coefficients,
    on a cubic grid of a compiled size; other functions are convolved a set at a
    time.  Read when a function is streamed, so it affects operators built
    afterwards.  On by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_paired(int(bool(enable)))


def pairing_sets() -> bool:
    """Whether the sets of a Toeplitz function are convolved in pairs where they can be."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_paired())


def bfloat16_function(enable: bool = True) -> None:
    """Store paired Toeplitz functions in bfloat16.

    Halves the host copy, the transfer and the device slot.  Values are rounded to
    a relative 2^-9 (float32: 2^-24) with float32's exponent range.  Read when a
    function is streamed.  On by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_bf16(int(bool(enable)))


def storing_bfloat16() -> bool:
    """Whether a Toeplitz function whose sets are paired is kept in bfloat16."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_bf16())


def paired_built() -> bool:
    """Whether the library was built with the pair kernels (``BARTORCH_MATHDX_DIR``)."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_paired_built())


def functions_bfloat16() -> int:
    """Toeplitz functions kept in bfloat16 since the counters were reset."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(6))


def using_fft_callbacks() -> bool:
    """Whether the passes around a volume's transform run inside it where they can."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_fft_callbacks())


def live_plans() -> int:
    """FINUFFT plans made and not yet destroyed; zero once every owner of one has been freed."""
    from bartorch._lib import library

    return int(library().bartorch_finufft_live_plans())


def decline_reason() -> str:
    """Why the last operator was BART's rather than FINUFFT's; empty if it was FINUFFT's."""
    from bartorch._lib import library

    return library().bartorch_nufft_decline_text().decode()


def operators_built() -> tuple[int, int]:
    """NUFFT operators built since the last reset, by FINUFFT and by BART."""
    from bartorch._lib import library

    lib = library()
    return int(lib.bartorch_nufft_counter(0)), int(lib.bartorch_nufft_counter(1))


def normals_built() -> tuple[int, int]:
    """Normal operators built since the last reset, by point spread function and by transform pair.

    ``pics --no-toeplitz`` and ``nufft -t`` decide which.
    """
    from bartorch._lib import library

    lib = library()
    return int(lib.bartorch_toeplitz_counter(0)), int(lib.bartorch_toeplitz_counter(1))


def functions_compressed() -> int:
    """Toeplitz functions built compressed since the last reset; decided at build time."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(2))


def functions_real() -> int:
    """Toeplitz functions stored as floats since the last reset; decided from the basis."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(4))


def pairs_convolved() -> int:
    """Pairs of sets convolved together by the pair kernels since the counters were reset."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(5))


def sets_through_callbacks() -> int:
    """Sets convolved with the passes inside cuFFT's transforms since the counters were reset."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(3))


def reset_counters() -> None:
    """Reset the NUFFT and Toeplitz counters."""
    from bartorch._lib import library

    lib = library()
    lib.bartorch_nufft_reset_counters()
    lib.bartorch_toeplitz_reset_counters()


def tolerance() -> float:
    """The tolerance FINUFFT plans are made with."""
    from bartorch._lib import library

    return float(library().bartorch_finufft_tolerance())


def upsampling() -> float:
    """Oversampling of FINUFFT's fine grid; 0 lets FINUFFT choose.

    BART's ``-o`` overrides it wherever it is not BART's default of 2.
    """
    from bartorch._lib import library

    return float(library().bartorch_finufft_upsampling())


def set_threads(n: int) -> None:
    """Set the threads a host transform takes; 0 lets FINUFFT take one per physical core.

    :func:`bartorch.set_num_threads` sets this together with BART's own count.
    Ignored on a card.
    """
    from bartorch._lib import library

    library().bartorch_finufft_set_threads(int(n))


def threads() -> int:
    """Threads a transform on the host takes; zero means FINUFFT chooses."""
    from bartorch._lib import library

    return int(library().bartorch_finufft_threads())


def simd_built() -> tuple[str, ...]:
    """The x86-64 levels FINUFFT's CPU transform is built for, the compiled-in baseline first.

    ``("default",)`` where FINUFFT is built for the target's own baseline alone.
    """
    from bartorch._lib import library

    return tuple(library().bartorch_finufft_simd_built().decode().split(","))


def simd() -> str:
    """The level host plans are made at.

    The newest built level this processor runs, unless the ``BARTORCH_FINUFFT_SIMD``
    environment variable, read when the level is first needed, or :func:`use_simd` names
    another.
    """
    from bartorch._lib import library

    return library().bartorch_finufft_simd().decode()


def use_simd(level: str | None = None) -> None:
    """Make host plans at ``level`` from now on, or at the newest available for None.

    Plans already made keep the level they were made at.

    Raises
    ------
    ValueError
        ``level`` is not built, or this processor does not run it.
    """
    from bartorch._lib import library

    if library().bartorch_finufft_set_simd(None if level is None else level.encode()) != 0:
        raise ValueError(
            f"FINUFFT at {level} is not available: built for {', '.join(simd_built())}"
        )


def fft_built() -> tuple[str, ...]:
    """The FFTs FINUFFT's CPU transform is built on, the compiled-in one first."""
    from bartorch._lib import library

    return tuple(library().bartorch_finufft_fft_built().decode().split(","))


def fft() -> str:
    """The FFT host plans are made on.

    ``"mkl"`` where the library carries a FINUFFT built on oneMKL at the level in
    force and the process has oneMKL (the ``mkl`` extra), unless :func:`use_fft`
    names another; the compiled-in one otherwise.
    """
    from bartorch._dispatch import _ensure_ready
    from bartorch._lib import library

    _ensure_ready()
    return library().bartorch_finufft_fft().decode()


def use_fft(name: str | None = None) -> None:
    """Make host plans on the FFT ``name`` from now on, or on the default for None.

    Plans already made keep the FFT they were made on.

    Raises
    ------
    ValueError
        ``name`` is not built at the level in force, or the process has no oneMKL
        for ``"mkl"``.
    """
    from bartorch._dispatch import _ensure_ready
    from bartorch._lib import library

    _ensure_ready()
    if library().bartorch_finufft_set_fft(None if name is None else name.encode()) != 0:
        raise ValueError(
            f"FINUFFT on {name} is not available at {simd()}: built on {', '.join(fft_built())}"
        )


def fallback_allowed() -> bool:
    """Whether BART's own operator may answer what FINUFFT will not."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_fallback_allowed())


def _tools_agree_with_bart(tolerance: float = 1e-2) -> bool:
    """Whether ``bart nufft`` answers the same with FINUFFT and with BART's gridder.

    Relative to BART's peak; ``tolerance`` allows for BART's gridding error.
    """
    import bartorch
    import bartorch.tools as bt
    from bartorch._lib import library

    lib = library()
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    image = bt.phantom([n, n]).reshape(1, n, n)

    lib.bartorch_finufft_use_in_tools(1)
    fast = bartorch.nufft(image, traj)

    with barts_own_gridder():
        reference = bartorch.nufft(image, traj)

    lib.bartorch_finufft_use_in_tools(1)

    scale = reference.abs().max()
    if scale == 0:
        return False
    return bool(((fast - reference).abs().max() / scale).item() < tolerance)


def three_components(traj: torch.Tensor) -> torch.Tensor:
    """``traj`` with ``kz`` written out as zero where it carries ``kx, ky`` only.

    BART's operators take three components per sample, and the FINUFFT
    substitution reads them in that layout.  The FINUFFT plan's dimension
    comes from the image, so a two-dimensional image is transformed by a 2D
    plan either way.  The padding is one copy of the trajectory, made when an
    operator is built; a three-component trajectory is returned as it is.
    """
    d = int(traj.shape[-1])
    if d == 3:
        return traj
    if d != 2:
        raise ValueError(f"a trajectory carries 2 or 3 components per sample, not {d}")
    return torch.cat((traj, torch.zeros_like(traj[..., :1])), dim=-1).contiguous()


def spatial_ndim(traj: torch.Tensor) -> int:
    """2 or 3: whether the trajectory's third component is used (BART always carries three)."""
    if traj.shape[-1] < 3:
        return 2
    return 3 if bool(torch.any(traj[..., 2].real != 0)) else 2


@contextlib.contextmanager
def barts_own_gridder():
    """Context in which BART's Kaiser-Bessel gridder serves the NUFFT instead of FINUFFT.

    For tests and the install-time agreement check only; nothing public reaches it.
    """
    from bartorch._lib import library

    lib = library()
    allowed = lib.bartorch_nufft_fallback_allowed()
    was_in_tools = lib.bartorch_finufft_usable()

    lib.bartorch_nufft_allow_fallback(1)
    lib.bartorch_finufft_use_in_tools(0)
    try:
        yield
    finally:
        lib.bartorch_finufft_use_in_tools(1 if was_in_tools else 0)
        lib.bartorch_nufft_allow_fallback(allowed)


_installed = False


def install_once() -> None:
    """Install the substitution on first use; a failure is logged once at warning level, not raised.

    Most commands need no NUFFT.  A NUFFT asked for after a failed install is
    refused, with the reason.
    """
    global _installed
    if _installed:
        return
    _installed = True
    try:
        use_in_tools(True)
    except RuntimeError as exc:
        logging.getLogger("bartorch._finufft").warning(
            "FINUFFT was not put in BART's place, so every non-uniform transform will be "
            "refused: %s",
            exc,
        )
