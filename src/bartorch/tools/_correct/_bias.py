"""N4 receive-field (bias-field) correction of magnitude images."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["bias_field_correct"]


def _simpleitk():
    try:
        import SimpleITK
    except ImportError as error:
        raise ImportError(
            "bias field correction requires SimpleITK: pip install 'bartorch[correct]'"
        ) from error
    return SimpleITK


def bias_field_correct(
    image: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    shrink_factor: int = 4,
    iterations: tuple[int, ...] = (50, 50, 50, 50),
    fitting_levels: int | None = None,
    return_field: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Divide a magnitude image by its N4-estimated multiplicative bias field.

    The field is SimpleITK's N4 estimate [1]_, a B-spline fitted on the image
    shrunk by ``shrink_factor`` and evaluated on the full grid.  Voxels where
    the field is zero are set to zero.

    Parameters
    ----------
    image : torch.Tensor
        Real 2D or 3D magnitude image, typically after coil combination.
    mask : torch.Tensor, default=None
        Voxels the field is estimated over, shaped like ``image``.  ``None``
        uses Otsu's threshold of the image.
    shrink_factor : int, default=4
        Downsampling factor of the grid the field is fitted on.
    iterations : tuple of int, default=(50, 50, 50, 50)
        Maximum iterations per fitting level, coarsest first.
    fitting_levels : int, default=None
        Number of fitting levels; ``len(iterations)`` if ``None``.
    return_field : bool, default=False
        Also return the field.

    Returns
    -------
    corrected : torch.Tensor
        ``image`` divided by the field, with the dtype and device of ``image``.
    field : torch.Tensor
        The field, returned only with ``return_field``.

    Raises
    ------
    ImportError
        If SimpleITK is not installed.
    ValueError
        If ``image`` is not 2D or 3D, or ``shrink_factor`` is not positive.

    References
    ----------
    .. [1] Tustison NJ, Avants BB, Cook PA, et al. N4ITK: improved N3 bias
       correction. IEEE Trans Med Imaging 2010;29:1310-1320.
    """
    sitk = _simpleitk()
    if shrink_factor < 1:
        raise ValueError(f"shrink_factor must be positive, got {shrink_factor}")
    image = torch.as_tensor(image)
    host = image.detach().cpu().numpy()
    if host.ndim not in {2, 3}:
        raise ValueError(f"image must be 2D or 3D, got shape {host.shape}")

    volume = sitk.GetImageFromArray(host.astype(np.float32))
    if mask is None:
        mask_volume = sitk.OtsuThreshold(volume, 0, 1, 200)
    else:
        host_mask = torch.as_tensor(mask).detach().cpu().numpy()
        mask_volume = sitk.GetImageFromArray(host_mask.astype(np.uint8))

    levels = len(iterations) if fitting_levels is None else int(fitting_levels)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(count) for count in iterations][:levels])
    if shrink_factor > 1:
        factors = [shrink_factor] * volume.GetDimension()
        corrector.Execute(sitk.Shrink(volume, factors), sitk.Shrink(mask_volume, factors))
    else:
        corrector.Execute(volume, mask_volume)

    log_field = corrector.GetLogBiasFieldAsImage(volume)
    field = sitk.GetArrayFromImage(sitk.Exp(log_field)).astype(np.float64)
    corrected = np.divide(
        host.astype(np.float64), field, out=np.zeros_like(field), where=field != 0
    )
    corrected = torch.as_tensor(corrected, device=image.device).to(image.dtype)
    if not return_field:
        return corrected
    return corrected, torch.as_tensor(field, device=image.device).to(image.dtype)
