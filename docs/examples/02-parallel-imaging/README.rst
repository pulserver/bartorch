Parallel imaging
----------------

Parallel imaging recovers an image from k-space undersampled along the
phase-encoding directions by exploiting the spatial sensitivities of a receive
array.  A SENSE reconstruction is only as accurate as its coil sensitivity
maps, and it propagates the thermal noise of the channels, amplified by the
g-factor.  This section treats both: sensitivity estimation from the
autocalibration (ACS) region, by direct division and by ESPIRiT; joint
estimation of image and sensitivities by nonlinear inversion when the ACS
region is too small for a separate calibration; and prewhitening of correlated
channel noise, evaluated by the SNR of the reconstruction.
:doc:`/explanation/encoding` derives the SENSE model and
:doc:`/explanation/nonlinear` the joint estimation.
