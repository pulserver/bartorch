Tours: acquisition and corrections
----------------------------------

Corrections you apply to the data before the reconstruction, or to the image
after it, each on its own. Most of them simulate the artefact from a known
truth, so you can see how much the correction recovers; the EPI
susceptibility tour corrects measured data instead, and checks the result
against an independent field map.

- Remove readout oversampling, and compare Fermi and Hann apodization.
- Correct the Nyquist ghost of an EPI train, and regrid a ramp-sampled readout.
- Estimate the receive bias field of a head array with N4.
- Undo the distortion of gradient nonlinearity from the coil's
  spherical-harmonic coefficients.
- Deblur a spiral image off resonance, by multifrequency interpolation and by
  a time-segmented reconstruction.
- Track rigid head motion from three orthogonal navigator planes.
- Correct the susceptibility distortion of a 3 T EPI pair with reversed phase
  encoding, downloaded from OpenNeuro.
- Prewhiten correlated channel noise, and measure the SNR it buys.

These tours also need the following packages, and the susceptibility tour
downloads about 2 MB of data into ``~/.cache/bartorch-examples``::

    pip install SimpleITK PyHySCO
