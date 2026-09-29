Tours
-----

Corrections applied to the data before reconstruction or to the image after
it, each shown on its own and independent of the course: removal of readout
oversampling and apodization, EPI Nyquist-ghost correction and regridding of
ramp-sampled readouts, receive bias-field correction, correction of the
geometric distortion caused by gradient nonlinearity, off-resonance deblurring
of spiral images, and rigid head-motion tracking from navigators.  Each tour
simulates the artefact from a known ground truth, so that the correction is
evaluated against it.

The section additionally requires::

    pip install SimpleITK
