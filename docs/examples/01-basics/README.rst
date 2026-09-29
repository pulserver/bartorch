Basics
------

The course starts from the data a scanner delivers, multichannel k-space, and
reconstructs an image from it.  The first lesson relates PyTorch tensors to
BART's arrays and commands, simulates an acquisition with BART's analytical
phantom and verifies the centring and scaling of BART's FFT against NumPy.  The
second reconstructs an undersampled Cartesian acquisition through the three
steps every later section builds on: coil compression, calibration of the coil
sensitivity maps by ESPIRiT, and a regularized SENSE reconstruction.  The
signal model is derived in :doc:`/explanation/encoding`, and the array
conventions are stated in :doc:`/explanation/data-layout`.
