Parallel imaging
----------------

Parallel imaging recovers an image from k-space undersampled along the
phase-encoding directions by exploiting the spatial sensitivities of a receive
array.  A SENSE reconstruction is only as accurate as its coil sensitivity
maps.  This section treats their estimation from the autocalibration (ACS)
region, by direct division and by ESPIRiT, and the joint estimation of image
and sensitivities by nonlinear inversion when the ACS region is too small for a
separate calibration.  The prewhitening of correlated channel noise, a
correction applied before this estimation, is the tour
:doc:`/auto_examples/07-tours/08-noise-prewhitening`.
:doc:`/explanation/encoding` derives the SENSE model and
:doc:`/explanation/nonlinear` the joint estimation.
