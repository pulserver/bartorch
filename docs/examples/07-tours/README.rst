Tours: acquisition and corrections
----------------------------------

Corrections applied to the data before reconstruction or to the image after
it, each shown on its own and independent of the course: removal of readout
oversampling and apodization, EPI Nyquist-ghost correction and regridding of
ramp-sampled readouts, receive bias-field correction, correction of the
geometric distortion caused by gradient nonlinearity, off-resonance deblurring
of spiral images, rigid head-motion tracking from navigators, correction of
susceptibility distortion in EPI, and prewhitening of correlated channel
noise.  The first six tours and the last simulate the artefact or the noise
from a known ground truth, so that the correction is evaluated against it; the
seventh corrects measured data and evaluates the correction against an
independent field map.

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
of a 3 T EPI pair with reversed phase encoding, downloaded from OpenNeuro. The
eighth measures the signal-to-noise ratio gained by prewhitening a SENSE
reconstruction with a noise-only acquisition.

The section additionally requires the following packages, and the seventh tour
downloads about 2 MB of data into ``~/.cache/bartorch-examples``::

    pip install SimpleITK PyHySCO
