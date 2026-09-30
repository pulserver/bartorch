"""File formats: BART's CFL, ISMRMRD raw data, DICOM and NIfTI images.

Images and k-space are tensors in bartorch's layout; geometry is a ``(4, 4)``
affine from voxel indices ``(x, y, z)`` to RAS millimetres, which every reader
returns and every image writer takes.  ISMRMRD, DICOM and NIfTI need the
``io`` extra.
"""

from ._cfl import readcfl, writecfl
from ._dicom import read_dicom, to_dicom, write_dicom
from ._mrd import read_mrd
from ._nifti import read_nifti, write_nifti

__all__ = [
    "read_dicom",
    "read_mrd",
    "read_nifti",
    "readcfl",
    "to_dicom",
    "write_dicom",
    "write_nifti",
    "writecfl",
]
