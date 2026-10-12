# Tours

Each tour stands on its own: it opens with what it shows and the lessons of
the {doc}`course` it assumes, so you can read any of them once you have the
Course behind you.

## Acquisition and corrections

Corrections you apply to the data before the reconstruction, or to the image
after it, each on its own. Most of them simulate the artefact from a known
truth, so you can see how much the correction recovers; the EPI
susceptibility tour corrects measured data instead, and checks the result
against an independent field map. These tours also need the following
packages, and the susceptibility tour downloads about 2 MB of data into
`~/.cache/bartorch-examples`:

```bash
pip install SimpleITK PyHySCO
```

| Tour | What it establishes |
| --- | --- |
| {doc}`/auto_examples/07-tours/01-readout-oversampling-and-apodization` | Remove readout oversampling, and compare Fermi and Hann apodization. |
| {doc}`/auto_examples/07-tours/02-epi-ghost-and-ramp-sampling` | Correct the Nyquist ghost of an EPI train, and regrid a ramp-sampled readout. |
| {doc}`/auto_examples/07-tours/03-bias-field` | Estimate the receive bias field of a head array with N4. |
| {doc}`/auto_examples/07-tours/04-gradient-nonlinearity` | Undo the distortion of gradient nonlinearity from the coil's spherical-harmonic coefficients. |
| {doc}`/auto_examples/07-tours/05-spiral-deblurring` | Deblur a spiral image off resonance, by multifrequency interpolation and by a time-segmented reconstruction. |
| {doc}`/auto_examples/07-tours/06-navigator-motion` | Track rigid head motion from three orthogonal navigator planes. |
| {doc}`/auto_examples/07-tours/07-epi-susceptibility-distortion` | Correct the susceptibility distortion of a 3 T EPI pair with reversed phase encoding, downloaded from OpenNeuro. |
| {doc}`/auto_examples/07-tours/08-noise-prewhitening` | Prewhiten correlated channel noise, and measure the SNR it buys. |

## Dynamic and quantitative workflows

The Course's encodings applied to a time series and to quantitative mapping.
{doc}`/explanation/non-cartesian` and {doc}`/explanation/nonlinear` give the
background.

| Tour | What it establishes |
| --- | --- |
| {doc}`/auto_examples/08-workflows/01-dynamic-golden-angle` | Reconstruct a dynamic contrast-enhanced series from one continuous golden-angle radial scan, with total variation along time. |
| {doc}`/auto_examples/08-workflows/02-subspace-t1-mapping` | Map $T_1$ from one continuous inversion-recovery scan, by reconstructing the coefficients of a subspace of the signal curves. |
| {doc}`/auto_examples/08-workflows/03-maps-from-scanner-images` | Fit a $T_2$ decay to magnitude images read from DICOM, as the scanner exports them, and write the map back. |

## Learning workflows

What comes after lessons 10 and 11, when you train and use a learned
regularizer on real problems. {doc}`/explanation/learned-reconstruction`
explains where a network enters a reconstruction. These tours also need:

```bash
pip install lightning torchio monai deepinv
```

| Tour | What it establishes |
| --- | --- |
| {doc}`/auto_examples/09-learning-workflows/01-networks-for-complex-volumes` | Networks for complex multi-contrast volumes, applied patch by patch. |
| {doc}`/auto_examples/09-learning-workflows/02-staged-training` | Training an unrolled network in stages, within the memory of one iteration. |
| {doc}`/auto_examples/09-learning-workflows/03-self-supervised-training` | Self-supervised training from undersampled data alone. |
| {doc}`/auto_examples/09-learning-workflows/04-annealed-plug-and-play` | Plug-and-play with a noise level that anneals over the iterations. |
| {doc}`/auto_examples/09-learning-workflows/05-uncertainty` | Voxel-wise uncertainty, calibrated. |

```{toctree}
:hidden:

/auto_examples/07-tours/01-readout-oversampling-and-apodization
/auto_examples/07-tours/02-epi-ghost-and-ramp-sampling
/auto_examples/07-tours/03-bias-field
/auto_examples/07-tours/04-gradient-nonlinearity
/auto_examples/07-tours/05-spiral-deblurring
/auto_examples/07-tours/06-navigator-motion
/auto_examples/07-tours/07-epi-susceptibility-distortion
/auto_examples/07-tours/08-noise-prewhitening
/auto_examples/08-workflows/01-dynamic-golden-angle
/auto_examples/08-workflows/02-subspace-t1-mapping
/auto_examples/08-workflows/03-maps-from-scanner-images
/auto_examples/09-learning-workflows/01-networks-for-complex-volumes
/auto_examples/09-learning-workflows/02-staged-training
/auto_examples/09-learning-workflows/03-self-supervised-training
/auto_examples/09-learning-workflows/04-annealed-plug-and-play
/auto_examples/09-learning-workflows/05-uncertainty
```
