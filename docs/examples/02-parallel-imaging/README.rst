Parallel imaging
----------------

A SENSE reconstruction is only as good as its coil sensitivities. In lesson 3
you estimate them from the autocalibration region, by direct division and by
ESPIRiT, and see what a poor estimate does to the image. In lesson 4 the
calibration region is too small for that, and you estimate the image and the
sensitivities together by nonlinear inversion. If your channels carry
correlated noise, whiten them first: the tour
:doc:`/auto_examples/07-tours/08-noise-prewhitening` shows how.
:doc:`/explanation/encoding` and :doc:`/explanation/nonlinear` give the
background.
