Tours
-----

Standalone examples of the corrections and the rigid-motion tracking in
:mod:`bartorch.tools`, each simulated from a known ground truth so that the
correction can be measured against it. They do not depend on one another or
on the course.

The first tour removes readout oversampling and compares Fermi and Hann
apodization by their Gibbs ringing and resolution. The second corrects the
Nyquist ghost of an EPI train from a three-line navigator and resamples a
ramp-sampled readout. The third estimates the receive bias field of a head
array with N4. The fourth corrects the geometric distortion and intensity
error of gradient nonlinearity from the coil's spherical-harmonic
coefficients. The fifth deblurs a spiral image off resonance by
multifrequency interpolation and by a time-segmented reconstruction. The
sixth measures rigid head motion from three orthogonal navigator planes and
filters it across a scan. The seventh corrects the susceptibility distortion
of an EPI image from a reversed phase-encoding pair.

The section additionally requires::

    pip install SimpleITK PyHySCO
