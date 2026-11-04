"""iGROG: non-Cartesian samples moved onto the Cartesian grid by closed-form GRAPPA kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = ["igrog"]


@dataclass
class _Correlation:
    """Coil cross-correlation ``Q`` on displacements ``-reach..reach``, ``1 / upsample`` apart.

    ``table`` is ``(C, C, *[2 reach upsample + 1] * d)``, displacement axes in trajectory order.
    """

    table: torch.Tensor
    reach: int
    upsample: int


@dataclass
class _Gridded:
    values: torch.Tensor
    index: torch.Tensor
    shot: torch.Tensor
    sample: torch.Tensor
    shape: tuple
    noise: torch.Tensor
    also: list = field(default_factory=list)

    def weights(self) -> torch.Tensor:
        """``1 / sqrt(noise)`` at unit mean square: the weights that whiten the gridded samples."""
        w = self.noise.clamp(min=1e-12).rsqrt()
        return w / w.square().mean().sqrt()


def _correlation(calib: torch.Tensor, reach: int, upsample: int) -> _Correlation:
    """Tabulate ``Q`` from Cartesian calibration k-space ``(C, [z,] y, x)``, centre ``n // 2``."""
    calib = calib.to(torch.complex128)
    cal = calib.shape[1:]
    d = len(cal)
    dims = tuple(range(1, d + 1))
    img = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(calib, dim=dims), dim=dims, norm="forward"), dim=dims
    )
    q = img[:, None] * img[None].conj()
    m = torch.arange(-reach * upsample, reach * upsample + 1, dtype=torch.float64) / upsample
    for ax in range(d):  # consumes the leading image axis, appends its displacement axis
        n = cal[ax]
        r = torch.arange(n, dtype=torch.float64) - n // 2
        q = torch.tensordot(q, torch.exp(-2j * math.pi * torch.outer(m, r) / n), dims=([2], [1]))
    q = q.permute(0, 1, *reversed(range(2, 2 + d))).contiguous()  # image order -> trajectory order
    return _Correlation(q.to(torch.complex64), reach, upsample)


def _lagrange4(t: torch.Tensor) -> torch.Tensor:
    """Cubic Lagrange weights of the nodes -1, 0, 1, 2 at ``t``."""
    return torch.stack(
        [
            -t * (t - 1) * (t - 2) / 6,
            (t + 1) * (t - 1) * (t - 2) / 2,
            -(t + 1) * t * (t - 2) / 2,
            (t + 1) * t * (t - 1) / 6,
        ],
        -1,
    )


def _interpolate(
    rows: torch.Tensor, size: int, cc: _Correlation, disp: torch.Tensor
) -> torch.Tensor:
    """``Q`` at displacements ``(..., d)`` as ``(..., C C)``, from ``(nodes, C C)`` rows."""
    d, U, R = disp.shape[-1], cc.upsample, cc.reach
    u = (disp.double() + R) * U
    base = torch.floor(u).long()
    w = _lagrange4(u - base).to(rows.real.dtype)  # (..., d, 4)
    shape = disp.shape[:-1]
    out = torch.zeros(*shape, rows.shape[1], dtype=rows.dtype, device=disp.device)
    # Node index and weight of each tap along each axis; a tap is one node per axis.
    strides = [size ** (d - 1 - a) for a in range(d)]
    node = [
        (base[..., a, None] + torch.arange(-1, 3, device=disp.device)) * strides[a]
        for a in range(d)
    ]
    for taps in torch.cartesian_prod(*[torch.arange(4)] * d).reshape(-1, d).tolist():
        flat = node[0][..., taps[0]]
        wt = w[..., 0, taps[0]]
        for a in range(1, d):
            flat = flat + node[a][..., taps[a]]
            wt = wt * w[..., a, taps[a]]
        out.add_(rows.index_select(0, flat.reshape(-1)).view(*shape, -1) * wt[..., None])
    return out


def _kernels(
    cc: _Correlation, rows: torch.Tensor, size: int, offsets: torch.Tensor, lam: float
) -> torch.Tensor:
    """Wiener kernels ``(M, C, D C)`` of sources at ``offsets`` ``(M, D, d)`` from their targets.

    ``Q(-δ) = Q(δ)^H`` and ``Q(0)`` is one node, so only the source pairs ``i < j``
    and the source-to-target displacements are interpolated.
    """
    M, D, d = offsets.shape
    C = int(round(rows.shape[1] ** 0.5))
    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
    disp = torch.stack(
        [offsets[:, i] - offsets[:, j] for i, j in pairs] + [-offsets[:, k] for k in range(D)], 1
    )
    q = _interpolate(rows, size, cc, disp).view(M, len(pairs) + D, C, C)
    zero = _interpolate(rows, size, cc, torch.zeros(1, d, device=offsets.device)).view(C, C)
    Rss = torch.empty(M, D, D, C, C, dtype=rows.dtype, device=offsets.device)
    for k in range(D):
        Rss[:, k, k] = zero
    for p, (i, j) in enumerate(pairs):
        Rss[:, i, j] = q[:, p]
        Rss[:, j, i] = q[:, p].mH
    Rss = Rss.permute(0, 1, 3, 2, 4).reshape(M, D * C, D * C)
    Rss = 0.5 * (Rss + Rss.mH)
    Rts = q[:, len(pairs) :].permute(0, 2, 1, 3).reshape(M, C, D * C)
    tr = torch.diagonal(Rss, dim1=-2, dim2=-1).real.mean(-1)
    Rss = Rss + (lam * tr)[:, None, None] * torch.eye(D * C, dtype=Rss.dtype, device=Rss.device)
    return torch.cholesky_solve(Rts.mH, torch.linalg.cholesky(Rss)).mH


def _grog_grid():
    """The library's host gridder, ``bartorch_grog_grid``."""
    from bartorch._lib import library

    return library().bartorch_grog_grid


_GROG_ERRORS = {
    1: "a missing argument",
    2: "a bad shape (sources x coils is at most 256)",
    3: "out of memory",
    4: "an index out of range",
    5: "a kernel system that is not positive definite",
    6: "a displacement beyond the correlation table",
}


def _native(cc, lam, tr, sets, shot, sample, src, outs, noise):
    """Grid the targets ``(shot, sample)`` of every set in ``sets`` with the host loop.

    ``sets`` are ``(C, S, P)`` complex64 on the host; ``outs`` the ``(C, M)``
    results and ``noise`` ``(M,)``, written in place.
    """
    d = tr.shape[-1]
    C, D = cc.table.shape[0], src.shape[1]
    table = cc.table.movedim((0, 1), (-2, -1)).contiguous()  # (nodes..., C, C)
    traj = tr.float().contiguous()
    shot, sample = shot.contiguous(), sample.contiguous()
    src = src.contiguous()
    from bartorch._marshal import float_pointer, int64_pointer, pointers

    err = _grog_grid()(
        d,
        C,
        D,
        cc.reach,
        cc.upsample,
        float(lam),
        table.data_ptr(),
        traj.shape[0],
        traj.shape[1],
        float_pointer(traj.data_ptr()),
        len(sets),
        pointers([a.data_ptr() for a in sets]),
        shot.numel(),
        int64_pointer(shot.data_ptr()),
        int64_pointer(sample.data_ptr()),
        int64_pointer(src.data_ptr()),
        pointers([o.data_ptr() for o in outs]),
        float_pointer(noise.data_ptr()),
    )
    if err:
        raise RuntimeError(f"igrog: the host gridder stopped on {_GROG_ERRORS.get(err, err)}")


def _card_grid(cc, lam, tr, sets, shot, sample, src, card, batch):
    """The card gridder over ``batch`` targets at a time; ``None`` when this build has none.

    ``sets`` are ``(C, S, P)`` complex64 on ``card``; returns the ``(C, M)``
    results of each set and ``(M,)`` noise gains, on the host.
    """
    from bartorch._lib import library
    from bartorch._marshal import float_pointer, int64_pointer

    lib = library()
    C, D = cc.table.shape[0], src.shape[1]
    if not lib.bartorch_grog_cuda_built() or C * D > 48:
        return None
    d = tr.shape[-1]
    table = cc.table.movedim((0, 1), (-2, -1)).contiguous().to(card)
    traj = tr.float().contiguous().to(card)
    data = torch.tensor([a.data_ptr() for a in sets], dtype=torch.int64, device=card)
    M = shot.numel()
    outs = [torch.empty(a.shape[0], M, dtype=torch.complex64) for a in sets]
    noise = torch.empty(M, dtype=torch.float32)
    status = torch.zeros(1, dtype=torch.int32, device=card)
    with torch.cuda.device(card):
        stream = torch.cuda.current_stream(card).cuda_stream
        for b in range(0, M, batch):
            e = min(M, b + batch)
            sh, sa = shot[b:e].to(card), sample[b:e].to(card)
            sr = src[b:e].contiguous().to(card)
            res = [torch.empty(a.shape[0], e - b, dtype=torch.complex64, device=card) for a in sets]
            nb = torch.empty(e - b, dtype=torch.float32, device=card)
            ptrs = torch.tensor([o.data_ptr() for o in res], dtype=torch.int64, device=card)
            err = lib.bartorch_grog_grid_cuda(
                d,
                C,
                D,
                cc.reach,
                cc.upsample,
                float(lam),
                table.data_ptr(),
                traj.shape[0],
                traj.shape[1],
                float_pointer(traj.data_ptr()),
                len(sets),
                data.data_ptr(),
                e - b,
                int64_pointer(sh.data_ptr()),
                int64_pointer(sa.data_ptr()),
                int64_pointer(sr.data_ptr()),
                ptrs.data_ptr(),
                float_pointer(nb.data_ptr()),
                status.data_ptr(),
                stream,
            )
            if err == -2:
                return None
            if err:
                raise RuntimeError(f"igrog: the card gridder failed with CUDA error {err}")
            for o, r in zip(outs, res):
                o[:, b:e] = r.cpu()
            noise[b:e] = nb.cpu()
    code = int(status.item())
    if code:
        raise RuntimeError(f"igrog: the card gridder stopped on {_GROG_ERRORS.get(code, code)}")
    return outs, noise


def _sources(traj: torch.Tensor, shot: torch.Tensor, sample: torch.Tensor, sources: int, spacing):
    """Sample indices ``(M, sources)`` of each target's sources along its readout."""
    S, P, _ = traj.shape
    win = torch.clamp(sample - sources // 2, 0, P - sources)[:, None] + torch.arange(sources)
    if spacing is None:
        return win
    step = torch.linalg.vector_norm(traj[:, 1:] - traj[:, :-1], dim=-1).double()
    arc = torch.cat([torch.zeros(S, 1, dtype=step.dtype), step.cumsum(1)], 1)
    big = float(arc[:, -1].max()) + 2 * spacing * sources + 1
    flat = (arc + big * torch.arange(S, dtype=step.dtype)[:, None]).ravel()
    o = (torch.arange(sources, dtype=step.dtype) - (sources - 1) / 2) * spacing
    want = arc[shot, sample][:, None] + o + big * shot[:, None].to(step.dtype)
    hi = torch.searchsorted(flat, want.contiguous()).clamp(max=S * P - 1)
    lo = (hi - 1).clamp(min=0)
    pick = torch.where((flat[hi] - want).abs() < (flat[lo] - want).abs(), hi, lo)
    first = shot[:, None] * P
    pick = torch.minimum(torch.maximum(pick, first), first + P - 1) - first
    srt = pick.sort(1).values
    return torch.where((srt[:, 1:] == srt[:, :-1]).any(1)[:, None], win, pick)


def igrog(
    kspace: torch.Tensor,
    traj: torch.Tensor,
    shape: tuple,
    calib,
    *,
    group: torch.Tensor | None = None,
    sources: int = 3,
    lam: float = 1e-3,
    spacing: float | None = 1.0,
    reach: int = 4,
    upsample: int = 8,
    batch: int = 1 << 20,
    device=None,
    double: bool = False,
    also=(),
):
    r"""Non-Cartesian samples moved to their nearest grid point by closed-form GRAPPA (iGROG).

    Each sample on the grid's extent is moved to its nearest Cartesian point by
    a kernel over ``sources`` samples of its readout and every coil.  iGROG
    (Abraham et al. [1]_) learns that kernel from the source offsets; here it
    is the Wiener (linear minimum mean square error) interpolator under the
    coil cross-correlation of the calibration k-space,

    .. math::

        W = R_{ts} (R_{ss} + \lambda \bar r I)^{-1}, \qquad
        Q_{cc'}(\delta) = \sum_r m_c(r) \overline{m_{c'}(r)} e^{-2\pi i \delta \cdot r / n},

    with :math:`R_{ss}` and :math:`R_{ts}` the correlation :math:`Q` between
    the sources and from them to the target, :math:`m_c` the calibration's
    coil images and :math:`\bar r` the mean signal power.  :math:`Q` is
    tabulated once on a fine grid of displacements and interpolated, so there
    is no fit and no seed.  The gridded samples can then be encoded by
    :func:`bartorch.linop.CartesianSense` with ``positions=``, whose normal is
    FFTs on the image grid: no NUFFT and no oversampled grid.

    Parameters
    ----------
    kspace : torch.Tensor
        ``(coils, *shots, samples)`` samples.
    traj : torch.Tensor
        ``(*shots, samples, ndim)`` in grid units, ``kx, ky[, kz]``, as for
        :class:`bartorch.linop.NUFFT`.
    shape : tuple
        The image grid, ``([z,] y, x)``.
    calib : torch.Tensor or sequence of torch.Tensor
        Cartesian calibration k-space ``(coils, [z,] y, x)`` centred at
        ``n // 2``, on the grid's k-space spacing; a sequence of them with
        ``group`` when the contrast changes over the scan.
    group : torch.Tensor, default=None
        ``(*shots,)`` index into ``calib`` of each shot's calibration.
    sources : int, default=3
        Readout samples per kernel.
    lam : float, default=1e-3
        Tikhonov weight relative to the mean signal power.
    spacing : float, default=1.0
        Arc length between sources along the readout, in grid units; ``None``
        takes consecutive samples.
    reach : int, default=4
        Displacements the correlation is tabulated to, in cells.
    upsample : int, default=8
        Table steps per cell.
    batch : int, default=1048576
        Targets per pass on a card: what it holds besides the samples.
    device : torch.device, default=None
        Where the kernels are solved; the result is on the host.  ``None``
        uses the samples' device.  Each target's kernel is solved in single
        precision by a compiled loop: one target per thread on a card, the
        targets spread over the cores on the host.  The card holds the samples
        and ``batch`` targets' indices and results, nothing else.
    double : bool, default=False
        Solve the kernels in double precision, by torch (slow, for checks).
    also : sequence of torch.Tensor, default=()
        More samples shaped as ``kspace``, such as a draw of pure noise, moved
        by the same kernels without solving them again.

    Returns
    -------
    gridded
        With ``values`` ``(coils, M)`` complex64, their flat grid ``index``
        ``(M,)``, the ``shot`` (flat index of ``*shots``) and ``sample`` each
        came from, ``noise`` ``(M,)`` (each value's noise variance per coil for
        unit white noise on the samples) and ``weights()``, the weights of unit
        mean square that whiten the values; ``also`` holds the values of the
        extra samples, in their order.

    References
    ----------
    .. [1] Abraham D, Blumenthal M, Uecker M, Ong F, et al. Implicit GROG for
       non-Cartesian MRI (iGROG). arXiv:2310.10823.

    Examples
    --------
    >>> g = igrog(kspace, traj, (256, 256, 256), calib)
    >>> values = g.values * g.weights()
    """
    d = traj.shape[-1]
    P = traj.shape[-2]
    tr = traj.reshape(-1, P, d).cpu()
    y = kspace.reshape(kspace.shape[0], -1, P)
    card = torch.device(device) if device is not None else y.device
    if sources > P:
        raise ValueError("more sources than samples per readout")
    calibs = [calib] if isinstance(calib, torch.Tensor) else list(calib)
    if len(calibs) > 1 and group is None:
        raise ValueError("several calibrations need the group of each shot")
    tables = [_correlation(c.cpu(), reach, upsample) for c in calibs]

    # Targets: every sample on the grid, with its nearest grid point.
    S = tr.shape[0]
    t = torch.round(tr)
    n = torch.tensor(list(reversed(shape)), dtype=tr.dtype)
    inside = ((t >= -(n // 2)) & (t <= n - n // 2 - 1)).all(-1)
    lin = torch.zeros(S, P, dtype=torch.long)
    for a in reversed(range(d)):  # flat index in image order ([z,] y, x)
        lin = lin * int(n[a]) + (t[..., a].long() + int(n[a]) // 2)
    shot = torch.arange(S)[:, None].expand(S, P)[inside]
    sample = torch.arange(P)[None].expand(S, P)[inside]
    lin = lin[inside]

    src = _sources(tr, shot, sample, sources, spacing)
    off = tr[shot[:, None], src] - torch.round(tr[shot, sample])[:, None]
    far = max(float((off[:, :, None] - off[:, None]).abs().amax()), float(off.abs().amax()))
    lim = reach - 2.0 / upsample
    if far > lim:
        raise ValueError(f"sources reach {far:.2f} cells apart; the table reaches {lim:.2f}")

    out = torch.empty(y.shape[0], shot.numel(), dtype=torch.complex64)
    extra = [a.reshape(a.shape[0], -1, P) for a in also]
    extra_out = [torch.empty(a.shape[0], shot.numel(), dtype=torch.complex64) for a in extra]
    noise = torch.empty(shot.numel(), dtype=torch.float32)
    which = torch.zeros_like(shot) if group is None else group.reshape(-1).cpu()[shot]

    def store(idx, values, gains):
        for o, r in zip([out, *extra_out], values):
            o[:, idx] = r
        noise[idx] = gains

    groups = [(cc, torch.nonzero(which == gi).squeeze(1)) for gi, cc in enumerate(tables)]
    groups = [(cc, idx) for cc, idx in groups if idx.numel()]
    if not double and card.type == "cuda":
        sets = [a.to(card, torch.complex64).contiguous() for a in [y, *extra]]
        got = []
        for cc, idx in groups:
            got.append(_card_grid(cc, lam, tr, sets, shot[idx], sample[idx], src[idx], card, batch))
            if got[-1] is None:  # no card gridder in this build, or a shape it does not take
                break
        del sets
        if all(g is not None for g in got):
            for (_, idx), g in zip(groups, got):
                store(idx, *g)
            return _Gridded(out, lin, shot, sample, tuple(shape), noise, extra_out)
    elif not double:
        sets = [a.to("cpu", torch.complex64).contiguous() for a in [y, *extra]]
        for cc, idx in groups:
            outs = [torch.empty(a.shape[0], idx.numel(), dtype=torch.complex64) for a in sets]
            gains = torch.empty(idx.numel(), dtype=torch.float32)
            _native(cc, lam, tr, sets, shot[idx], sample[idx], src[idx], outs, gains)
            store(idx, outs, gains)
        return _Gridded(out, lin, shot, sample, tuple(shape), noise, extra_out)
    # Torch: double precision, or a card this build has no gridder for.
    step = min(batch, 65536)  # its (B, D C, D C) systems are what bound the batch
    for gi, cc in enumerate(tables):
        # The table as (nodes, C C) rows, so each tap gathers contiguous rows.
        dtype = torch.complex128 if double else torch.complex64
        size = cc.table.shape[2]
        rows = cc.table.reshape(cc.table.shape[0] ** 2, -1).T.contiguous().to(card, dtype)
        idx = torch.nonzero(which == gi).squeeze(1)
        for b in range(0, idx.numel(), step):
            ib = idx[b : b + step]
            s, sb = shot[ib], src[ib]
            W = _kernels(cc, rows, size, off[ib].to(card), lam)  # (B, C, D C)
            v = y[:, s[:, None], sb].permute(1, 2, 0).reshape(len(s), -1)  # (B, D C), source-major
            out[:, ib] = torch.einsum("bcj,bj->cb", W, v.to(card, W.dtype)).to(
                "cpu", torch.complex64
            )
            noise[ib] = (W.abs().square().sum((1, 2)) / W.shape[1]).float().cpu()
            for a, o in zip(extra, extra_out):
                v = a[:, s[:, None], sb].permute(1, 2, 0).reshape(len(s), -1)
                o[:, ib] = torch.einsum("bcj,bj->cb", W, v.to(card, W.dtype)).to(
                    "cpu", torch.complex64
                )
    return _Gridded(out, lin, shot, sample, tuple(shape), noise, extra_out)
