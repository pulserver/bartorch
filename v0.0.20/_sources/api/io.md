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
in the same place.

CFL arrays are NumPy arrays in BART's dimension order, the reverse of a
C-order tensor shape, so `array.T` converts between the two.

```{eval-rst}
.. currentmodule:: bartorch.io
```

| Object | Description |
| --- | --- |
| {obj}`~bartorch.io.read_mrd` | Read one encoding space of an ISMRMRD file into k-space placed by its counters, with the mask, trajectory, noise readouts and affine |
| {obj}`~bartorch.io.read_mrd_stream` | Read one encoding space of an MRD stream -- the XML header and the acquisitions a scanner sends -- as `read_mrd` reads a file's |
| {obj}`~bartorch.io.read_dicom` | Read a DICOM MR series, or several as one, into images sorted by contrast and slice, with their timings and affine |
| {obj}`~bartorch.io.to_dicom` | Convert a real image and its affine to the datasets of one DICOM MR series |
| {obj}`~bartorch.io.write_dicom` | Write a real image and its affine as a DICOM MR series |
| {obj}`~bartorch.io.read_nifti` | Read NIfTI files and their BIDS sidecars into images, timings and affine |
| {obj}`~bartorch.io.write_nifti` | Write an image and its affine as NIfTI, and its timings as a BIDS sidecar |
| {obj}`~bartorch.io.read_series` | Read a DICOM or NIfTI series with its magnitude and phase, or real and imaginary, images paired into one complex image, phase in radians, and its acquisition |
| {obj}`~bartorch.io.Series` | What `read_series` returns: the image, its affine, its acquisition and a header to write from |
| {obj}`~bartorch.io.Acquisition` | Timings per contrast, voxel size, main-field direction in voxel axes, field strength, vendor, body part and the phase scaling used |
| {obj}`~bartorch.io.readcfl` | Read `name.hdr` and `name.cfl` into a NumPy array in BART's dimension order |
| {obj}`~bartorch.io.writecfl` | Write a NumPy array in BART's dimension order as `name.hdr` and `name.cfl` |

How a series stores the parts of a complex image differs by vendor: Siemens
writes magnitude and phase as two series, GE and Philips write them into one,
and an enhanced (multi-frame) dataset tags each frame.  `read_series` finds
each image's part from the standard and private tags, and scales phase from
the span of stored values that stands for one turn.  A reconstruction of one's
own may store phase in any span, so `phase_range=` states it, and a warning
says when the values do not fill the span used.

{doc}`../auto_examples/08-workflows/03-maps-from-scanner-images` reads a
multi-echo series from DICOM, fits a $T_2$ map to it and writes the map back.
