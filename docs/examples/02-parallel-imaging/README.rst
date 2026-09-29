Parallel imaging
----------------

The coil sensitivities a SENSE reconstruction depends on, and the channel
noise it propagates.

The first lesson compares three sensitivity estimates from the same
acquisition, :func:`bartorch.tools.caldir`, ESPIRiT and nonlinear inversion.
The second estimates the image and the sensitivities jointly where the
calibration region is too small for a separate calibration. The third measures
the effect of correlated channel noise on the reconstruction and removes it by
prewhitening.
