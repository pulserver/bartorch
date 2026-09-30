"""BART's reconstruction commands, held privately as references.

Each is answered publicly by an assembly -- :func:`bartorch.apps.pics`,
:func:`bartorch.apps.mobafit`, :func:`bartorch.apps.moba`, or an encoding from
:mod:`bartorch.linop` under a solver from :mod:`bartorch.optim` -- and the
tests hold the assembly to the command run here.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from bartorch import _call
from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch.priors._base import Regularizer, _as_terms, _command_line

__all__ = ["pics"]

#: The solvers ``pics`` chooses between; BART writes each as a flag of its own.
SOLVERS = {
    "ist": "ist",
    "fista": "fista",
    "admm": "admm",
    "pridu": "pridu",
    "eulermaruyama": "eulermaruyama",
}

#: ``pics`` options the terms set, and where each is set instead.
_SET_BY_TERMS = {
    "R": "the regularizers argument",
    "n": "the terms' randshift",
    "N": "LocallyLowRank(overlapping=True)",
    "b": "LocallyLowRank(block=...)",
    "wavelet": "the terms' family",
    "alpha": "the terms' alpha",
    "gamma": "the terms' gamma",
}


@curated("pics")
def pics(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    *,
    regularizers: Regularizer | Iterable[Regularizer] | None = None,
    l2: float | None = None,
    solver: str | None = None,
    maxiter: int | None = None,
    step: float | None = None,
    admm_rho: float | None = None,
    cg_maxiter: int | None = None,
    traj: torch.Tensor | None = None,
    pattern: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    initial: torch.Tensor | None = None,
    psf: torch.Tensor | None = None,
    toeplitz: bool | None = None,
    lowmem: bool = False,
    real: bool = False,
    eigen_step: bool = False,
    **extra,
) -> torch.Tensor:
    """Parallel-imaging compressed-sensing reconstruction.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space, C order.
    sensitivities : torch.Tensor
        Coil sensitivities, as :func:`bartorch.tools.ecalib` or
        :func:`bartorch.tools.caldir` produce them.
    regularizers : Regularizer or iterable of Regularizer, default=None
        :mod:`bartorch.priors` terms (``-R``).  Their axes index ``kspace``'s
        shape, negative ones counting from the last axis.  A setting ``pics``
        takes once for every term -- ``randshift``, ``family``, a
        :class:`~bartorch.priors.LocallyLowRank` ``block`` -- has to agree
        across the terms.
    l2 : float, default=None
        Plain Tikhonov weight (``-r``).
    solver : {'ist', 'fista', 'admm', 'pridu', 'eulermaruyama'}, default=None
        ``None`` lets ``pics`` choose from the regularizers.  ``pics`` passes
        IST and FISTA no transform, and chooses FISTA when the first term is
        :class:`~bartorch.priors.FourierL1` or
        :class:`~bartorch.priors.Laplace`; the proximal operator is then
        applied to the image itself.  :func:`bartorch.apps.pics` and the
        solvers of :mod:`bartorch.optim` refuse that combination.
    maxiter : int, default=None
        Iterations (``-i``).
    step : float, default=None
        Step size (``-s``).
    admm_rho : float, default=None
        ADMM penalty (``-u``); setting it selects ADMM unless ``solver`` says
        otherwise.
    cg_maxiter : int, default=None
        Inner conjugate-gradient steps for ADMM (``-C``).
    traj : tensor, default=None
        Non-Cartesian trajectory (``-t``), in grid units.
    pattern : tensor, default=None
        Sampling pattern or weights (``-p``).
    basis : tensor, default=None
        Subspace basis over frames and coefficients (``-B``).
    initial : tensor, default=None
        Warm start (``-W``).
    psf : tensor, default=None
        A point spread function computed elsewhere (``--psf_import``), the
        route by which a normal operator built outside BART is supplied.
    toeplitz : bool, default=None
        ``False`` passes ``--no-toeplitz``; ``None`` leaves BART's default.
    lowmem : bool, default=False
        Hold one set of frequencies of the point spread function at a time
        (``--lowmem``).
    real : bool, default=False
        Constrain the image to be real (``-c``).
    eigen_step : bool, default=False
        Scale the step size by the largest eigenvalue (``-e``).
    **extra
        Further BART ``pics`` options, by name.  One that picks dimensions
        (``L``, ``shared_img_dims``) takes axes of ``kspace``.

    Returns
    -------
    torch.Tensor
        The reconstructed image.

    Examples
    --------
    >>> image = pics(kspace, maps, l2=0.01, maxiter=50)
    >>> image = pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005), solver="fista")
    >>> image = pics(kspace, maps, traj=trajectory, basis=subspace)
    """
    for keyword in sorted(_SET_BY_TERMS.keys() & extra.keys()):
        raise TypeError(f"pics takes {keyword} from {_SET_BY_TERMS[keyword]}")
    flags: dict = _call.translate("pics", dict(extra), [kspace, sensitivities])
    if regularizers is not None:
        arguments, shared = _command_line(_as_terms(regularizers), kspace.ndim, "pics")
        flags["R"] = arguments
        if "randshift" in shared:
            flags["n"] = True
        if shared.get("overlapping"):
            flags["N"] = True
        for setting, keyword in (
            ("block", "b"),
            ("family", "wavelet"),
            ("alpha", "alpha"),
            ("gamma", "gamma"),
        ):
            if setting in shared:
                flags[keyword] = shared[setting]
    if l2 is not None:
        flags["r"] = l2
    if solver is not None:
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {sorted(SOLVERS)}, not {solver!r}")
        flags[SOLVERS[solver]] = True
    if maxiter is not None:
        flags["i"] = maxiter
    if step is not None:
        flags["s"] = step
    if admm_rho is not None:
        flags["u"] = admm_rho
    if cg_maxiter is not None:
        flags["C"] = cg_maxiter
    for keyword, value in (
        ("t", traj),
        ("p", pattern),
        ("B", basis),
        ("W", initial),
        ("psf_import", psf),
    ):
        if value is not None:
            flags[keyword] = value
    if toeplitz is False:
        flags["no_toeplitz"] = True
    if lowmem:
        flags["lowmem"] = True
    if real:
        flags["c"] = True
    if eigen_step:
        flags["e"] = True
    return dispatch("pics", [kspace, sensitivities], None, **flags)


#: Commands run through a wrapper built from the catalogue.
_DERIVED = ("itsense", "looklocker", "moba", "mobafit", "wave", "wshfl")

for _name in _DERIVED:
    globals()[_name] = _call.build(_name, __name__)
del _name

__all__ = [*__all__, *_DERIVED]
