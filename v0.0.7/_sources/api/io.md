# File I/O

`bartorch.io` reads and writes the files a reconstruction starts from and
ends in: BART's CFL arrays, ISMRMRD raw data, and DICOM and NIfTI images.
Every reader returns tensors in bartorch's layout
({doc}`../explanation/data-layout`) and plain containers beside them: the
ISMRMRD header as `ismrmrd` parses it, a dictionary of per-readout or
per-contrast values, and the image geometry as one `(4, 4)` affine from voxel
indices `(x, y, z)` -- the last three axes of an image tensor, reversed -- to
RAS coordinates in millimetres, the convention of NIfTI.  The image writers
take the same affine, so an image read from one format is written to the other
in the same place.  ISMRMRD, DICOM and NIfTI need the `io` extra:
`pip install 'bartorch[io]'`.

CFL arrays are NumPy arrays in BART's dimension order, the reverse of a
C-order tensor shape, so `array.T` converts between the two.

```{eval-rst}
.. currentmodule:: bartorch.io
```

| Object | Description |
| --- | --- |
| {obj}`~bartorch.io.read_mrd` | Read one encoding space of an ISMRMRD file into k-space placed by its counters, with the mask, trajectory, noise readouts and affine |
| {obj}`~bartorch.io.read_dicom` | Read a DICOM MR series, or several as one, into images sorted by contrast and slice, with their timings and affine |
| {obj}`~bartorch.io.to_dicom` | Convert a real image and its affine to the datasets of one DICOM MR series |
| {obj}`~bartorch.io.write_dicom` | Write a real image and its affine as a DICOM MR series |
| {obj}`~bartorch.io.read_nifti` | Read NIfTI files and their BIDS sidecars into images, timings and affine |
| {obj}`~bartorch.io.write_nifti` | Write an image and its affine as NIfTI, and its timings as a BIDS sidecar |
| {obj}`~bartorch.io.readcfl` | Read `name.hdr` and `name.cfl` into a NumPy array in BART's dimension order |
| {obj}`~bartorch.io.writecfl` | Write a NumPy array in BART's dimension order as `name.hdr` and `name.cfl` |

{doc}`../auto_examples/08-workflows/03-maps-from-scanner-images` reads a
multi-echo series from DICOM, fits a $T_2$ map to it and writes the map back.
