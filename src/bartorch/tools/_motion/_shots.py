"""Rigid motion between the shots of a segmented 3D acquisition: its encoding and its pose fit."""

from __future__ import annotations

import math

import torch

__all__ = ["estimate_rigid_motion", "rigid_encoding"]


def _lines(masks: torch.Tensor):
    if masks.ndim != 3:
        raise ValueError(f"masks must be (shots, z, y), got shape {tuple(masks.shape)}")
    if int(masks.sum(dim=0).max()) > 1:
        raise ValueError("a line is placed by more than one shot")
    shot, partition, line = torch.nonzero(masks, as_tuple=True)
    return shot, partition, line


def _poses(count: int, rotations, shifts) -> tuple[torch.Tensor, torch.Tensor]:
    turn = torch.eye(3, dtype=torch.float64).expand(count, 3, 3).clone()
    move = torch.zeros((count, 3), dtype=torch.float64)
    if rotations is not None:
        turn = torch.as_tensor(rotations, dtype=torch.float64).reshape(count, 3, 3).clone()
    if shifts is not None:
        move = torch.as_tensor(shifts, dtype=torch.float64).reshape(count, 3).clone()
    return turn, move


def _wavenumbers(partition, line, shape, voxel):
    """Scanner-frame wavenumbers ``(lines, x, 3)``, ``(x, y, z)``, per unit of ``voxel``."""
    nz, ny, nx = shape
    vz, vy, vx = voxel
    kx = (torch.arange(nx, dtype=torch.float64) - nx // 2) / (nx * vx)
    ky = (line.double() - ny // 2) / (ny * vy)
    kz = (partition.double() - nz // 2) / (nz * vz)
    count = line.numel()
    return torch.stack(
        [kx.expand(count, nx), ky[:, None].expand(count, nx), kz[:, None].expand(count, nx)],
        dim=-1,
    )


def _sampling(wavenumbers, shot, turn, move, shape, voxel):
    """Trajectory in grid units and the shift phase of each sample under the shots' poses."""
    nz, ny, nx = shape
    vz, vy, vx = voxel
    rotated = torch.einsum("lji,lsj->lsi", turn[shot], wavenumbers)
    extent = torch.tensor([nx * vx, ny * vy, nz * vz], dtype=torch.float64)
    traj = (rotated * extent).float()
    phase = torch.exp(-2j * math.pi * (wavenumbers * move[shot][:, None]).sum(-1))
    return traj, rotated, phase.to(torch.complex64)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], -1),
            torch.stack([z, zero, -x], -1),
            torch.stack([-y, x, zero], -1),
        ],
        -2,
    )


def rigid_encoding(
    maps: torch.Tensor,
    masks: torch.Tensor,
    voxel_size: tuple[float, float, float],
    rotations: torch.Tensor | None = None,
    shifts: torch.Tensor | None = None,
    *,
    device=None,
):
    r"""SENSE encoding of a segmented 3D acquisition whose shots each read the object in a pose.

    Shot :math:`s` reads, on its lines, the object turned by :math:`R_s`
    about the centre of the grid and then shifted by :math:`d_s`:
    :math:`y_s(k) = e^{-2\pi i\, k \cdot d_s}\, \mathcal{F}[S x](R_s^{T} k)`,
    with :math:`k` the wavenumber of the sample in the scanner frame. The
    coil sensitivities turn and shift with the object, which holds for the
    small turns of a head between shots. The samples a turn moves off the
    grid are read by :class:`~bartorch.linop.NoncartesianSense`; at rest the
    operator is :func:`~bartorch.linop.CartesianSense` over the placed lines,
    to the accuracy of the non-uniform transform.

    Parameters
    ----------
    maps : torch.Tensor
        ``(coils, z, y, x)`` coil sensitivities.
    masks : torch.Tensor
        ``(shots, z, y)`` booleans, the lines each shot placed; a line belongs
        to one shot at most.
    voxel_size : tuple of float
        ``(z, y, x)`` voxel size, in the length unit of ``shifts``.
    rotations : torch.Tensor, default=None
        ``(shots, 3, 3)`` rotation matrices acting on ``(x, y, z)`` column
        vectors; ``None`` is the identity for every shot.
    shifts : torch.Tensor, default=None
        ``(shots, 3)`` displacements ``(x, y, z)``; ``None`` is zero.
    device : torch.device or str, default=None
        Where the transform runs.

    Returns
    -------
    bartorch.linop.LinearOperator
        From the ``(z, y, x)`` image to ``(coils, z, y, x)`` k-space, centred,
        zero off the placed lines.

    Raises
    ------
    ValueError
        If ``masks`` is not ``(shots, z, y)`` or two shots place one line.
    """
    from bartorch import linop

    coils, *grid = maps.shape
    grid = tuple(int(n) for n in grid)
    if len(grid) != 3:
        raise ValueError(f"maps must be (coils, z, y, x), got shape {tuple(maps.shape)}")
    shot, partition, line = _lines(masks)
    turn, move = _poses(int(masks.shape[0]), rotations, shifts)
    wavenumbers = _wavenumbers(partition, line, grid, voxel_size)
    traj, _, phase = _sampling(wavenumbers, shot, turn, move, grid, voxel_size)
    on = torch.device(device) if device is not None else maps.device
    reads = linop.NoncartesianSense(
        maps.to(torch.complex64), grid, traj=traj.to(on), weights=phase.to(on), device=device
    )
    count, nx = line.numel(), grid[-1]
    where = (partition.to(on), line.to(on))

    def place(samples):
        kspace = torch.zeros((coils, *grid), dtype=samples.dtype, device=samples.device)
        kspace[:, where[0].to(samples.device), where[1].to(samples.device)] = samples
        return kspace

    def take(kspace):
        return kspace[:, where[0].to(kspace.device), where[1].to(kspace.device)].contiguous()

    scatter = linop.LinearOperator.from_callbacks((coils, *grid), (coils, count, nx), place, take)
    return scatter @ reads


def _central(kspace, maps, masks, image, voxel_size, fraction):
    """Restrict the acquisition to the central ``fraction`` of k-space, coarser grid, same FOV."""
    from bartorch._fourier import fft

    full = tuple(int(n) for n in maps.shape[1:])
    kept = tuple(max(2, int(round(n * fraction))) if n > 1 else 1 for n in full)
    cut = tuple(
        slice(n // 2 - m // 2, n // 2 - m // 2 + m) for n, m in zip(full, kept, strict=True)
    )
    axes = (-3, -2, -1)
    gain = math.sqrt(math.prod(full) / math.prod(kept))

    def coarse(values):
        spectrum = fft(values, axes, unitary=True).reshape(values.shape)[(..., *cut)]
        return fft(spectrum.contiguous(), axes, inverse=True, unitary=True).reshape(spectrum.shape)

    voxel = tuple(v * n / m for v, n, m in zip(voxel_size, full, kept, strict=True))
    return (
        kspace[(..., *cut)].contiguous(),
        coarse(maps.to(torch.complex64)) * gain,
        masks[:, cut[0], cut[1]].contiguous(),
        coarse(image.reshape(full).to(torch.complex64)),
        voxel,
    )


def estimate_rigid_motion(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    masks: torch.Tensor,
    image: torch.Tensor,
    voxel_size: tuple[float, float, float],
    rotations: torch.Tensor | None = None,
    shifts: torch.Tensor | None = None,
    *,
    steps: int = 3,
    iterations: int = 40,
    lambda_: float = 1e-5,
    centre: float = 1.0,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    r"""Rigid pose of each shot of a segmented 3D acquisition relative to the first, and its image.

    Each of ``steps`` Gauss-Newton steps linearises the encoding of
    :func:`rigid_encoding` about the current poses and image and solves for
    the image and the pose steps together, by ``iterations`` conjugate
    gradients on :math:`\| A x + J \delta - y \|^2 + \lambda \| (x, \delta) \|^2`
    with :math:`J` scaled to unit columns, from the current image. Solved
    jointly, the image cannot absorb the inconsistency a pose leaves in the
    data, as it does when the two are fitted in turn. :math:`J` is in closed
    form: a turn :math:`\delta` composed after :math:`R_s` moves the rotated
    wavenumber :math:`k'` by :math:`-\delta \times k'`, so a sample's
    derivative along it is :math:`2\pi i\, k' \times \mathcal{F}[S x\, r]`,
    which one transform of the image weighted by its position :math:`r` gives
    for every shot at once; a shift multiplies a sample by
    :math:`-2\pi i\, k`. The first shot stays at rest. Data are scaled to a
    peak of one for the solve. With ``centre`` below one, the fit runs on the
    central fraction of k-space along each axis, on the coarser grid of the
    same field of view, the maps and the starting image resampled onto it.

    Parameters
    ----------
    kspace, maps : torch.Tensor
        ``(coils, z, y, x)`` centred k-space and coil maps.
    masks : torch.Tensor
        ``(shots, z, y)`` booleans, the lines each shot placed.
    image : torch.Tensor
        ``(z, y, x)`` complex starting image, in the first shot's pose.
    voxel_size : tuple of float
        ``(z, y, x)`` voxel size, in the length unit of the shifts.
    rotations, shifts : torch.Tensor, default=None
        ``(shots, 3, 3)`` and ``(shots, 3)`` starting poses, as
        :func:`rigid_encoding` takes them; ``None`` starts at rest.
    steps : int, default=3
        Gauss-Newton steps.
    iterations : int, default=40
        Conjugate-gradient iterations of each step.
    lambda_ : float, default=1e-5
        Tikhonov weight of the joint solve, on the image and on the scaled
        pose steps alike; a larger weight damps the pose steps enough to
        stall them.
    centre : float, default=1.0
        Fraction of k-space, along each axis, the fit reads.
    device : torch.device or str, default=None
        Where the transforms run.

    Returns
    -------
    rotations : torch.Tensor
        ``(shots, 3, 3)`` float64 rotation matrices.
    shifts : torch.Tensor
        ``(shots, 3)`` float64 displacements.
    image : torch.Tensor
        ``(z, y, x)`` image of the last step, on the grid the fit ran on, on
        the device of ``image``.
    residual : float
        ``||A x - M y|| / ||M y||`` of that image under the returned poses.
    """
    from bartorch import linop, optim

    if centre < 1.0:
        kspace, maps, masks, image, voxel_size = _central(
            kspace, maps, masks, image, voxel_size, centre
        )
    grid = tuple(int(n) for n in maps.shape[1:])
    shot, partition, line = _lines(masks)
    count = int(masks.shape[0])
    turn, move = _poses(count, rotations, shifts)
    wavenumbers = _wavenumbers(partition, line, grid, voxel_size)
    on = torch.device(device) if device is not None else maps.device
    home = image.device
    maps = maps.to(on, torch.complex64)
    data = kspace.to(on)[:, partition.to(on), line.to(on)].to(torch.complex64)
    scale = float(data.abs().max()) or 1.0
    data = data / scale
    image = image.reshape(grid).to(on, torch.complex64) / scale
    nz, ny, nx = grid
    vz, vy, vx = voxel_size
    axes = [
        (torch.arange(n, dtype=torch.float32, device=on) - n // 2) * v
        for n, v in ((nx, vx), (ny, vy), (nz, vz))
    ]
    owner = shot.to(on)
    lab = wavenumbers.to(on, torch.float32)
    size, unknowns = math.prod(grid), 6 * (count - 1)

    def reads(batch):
        traj, rotated, phase = _sampling(wavenumbers, shot, turn, move, grid, voxel_size)
        operator = linop.NoncartesianSense(
            maps,
            grid if batch is None else (batch, *grid),
            traj=traj.to(on),
            weights=phase.to(on),
            device=device,
        )
        return operator, rotated.to(on, torch.float32)

    for _ in range(steps):
        transform, rotated = reads(4)
        weighted = torch.stack(
            [
                image,
                image * axes[0][None, None, :],
                image * axes[1][None, :, None],
                image * axes[2][:, None, None],
            ]
        )
        read = transform(weighted)
        signal, moments = read[0], read[1:].permute(1, 2, 3, 0)
        turned = rotated[None].expand_as(moments.real)
        cross = torch.linalg.cross(turned, moments.real) + 1j * torch.linalg.cross(
            turned, moments.imag
        )
        jacobian = torch.cat(
            [2j * math.pi * cross, -2j * math.pi * lab[None] * signal[..., None]], dim=-1
        )
        norms = torch.zeros((count, 6), dtype=torch.float32, device=on)
        norms.index_add_(0, owner, jacobian.abs().square().sum(dim=(0, 2)))
        norms = norms.sqrt().clamp_min(1e-30)
        jacobian = jacobian / norms[owner][None, :, None, :]
        coefficient_shape = (count, 6)

        def along(steps_, jacobian=jacobian):
            full = torch.zeros(coefficient_shape, dtype=steps_.dtype, device=steps_.device)
            full[1:] = steps_.reshape(count - 1, 6)
            return (jacobian * full[owner.to(steps_.device)][None, :, None, :]).sum(-1)

        def against(samples, jacobian=jacobian):
            per_line = (jacobian.conj() * samples[..., None]).sum(dim=(0, 2))
            full = torch.zeros(coefficient_shape, dtype=samples.dtype, device=samples.device)
            full.index_add_(0, owner.to(samples.device), per_line)
            return full[1:].reshape(-1)

        poses = linop.LinearOperator.from_callbacks(tuple(data.shape), (unknowns,), along, against)
        single, _ = reads(None)
        joint = linop.hstack(
            [single @ linop.Reshape(grid, (size,)), poses @ linop.Real((unknowns,))]
        )
        start = torch.cat([image.reshape(-1), torch.zeros(unknowns, dtype=image.dtype, device=on)])
        solved = optim.CG(lambda_=lambda_, maxiter=iterations)(data, joint, x0=start)
        image = solved[:size].reshape(grid).to(on)
        step = (solved[size:].real.reshape(count - 1, 6) / norms[1:]).double().cpu()
        turn[1:] = turn[1:] @ torch.linalg.matrix_exp(_skew(step[:, :3]))
        move[1:] = move[1:] + step[:, 3:]
    single, _ = reads(None)
    error = single(image) - data
    residual = float(error.norm() / data.norm().clamp_min(1e-30))
    return turn, move, (image * scale).to(home), residual
