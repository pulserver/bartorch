# Tours

Standalone examples beyond the {doc}`course`. Each names the lessons it assumes.

## Acquisition and corrections

Corrections before and after reconstruction. Most are tested on a simulated
artefact with a known truth; the EPI susceptibility tour uses measured data and
an independent field map. These tours also need the following
packages, and the susceptibility tour downloads about 2 MB of data into
`~/.cache/bartorch-examples`:

```bash
pip install PyHySCO
```

| Tour | Contents |
| --- | --- |
| {doc}`/auto_examples/07-tours/01-readout-oversampling-and-apodization` | Readout oversampling removal; Fermi and Hann apodization |
| {doc}`/auto_examples/07-tours/02-epi-ghost-and-ramp-sampling` | EPI Nyquist ghost correction; ramp-sampling regridding |
| {doc}`/auto_examples/07-tours/03-bias-field` | Receive bias field estimation with N4 |
| {doc}`/auto_examples/07-tours/04-gradient-nonlinearity` | Gradient nonlinearity correction from spherical-harmonic coefficients |
| {doc}`/auto_examples/07-tours/05-spiral-deblurring` | Spiral off-resonance deblurring: multifrequency interpolation, time-segmented reconstruction |
| {doc}`/auto_examples/07-tours/06-navigator-motion` | Rigid head motion from orthogonal navigator planes |
| {doc}`/auto_examples/07-tours/07-epi-susceptibility-distortion` | EPI susceptibility distortion correction from reversed phase encoding |
| {doc}`/auto_examples/07-tours/08-noise-prewhitening` | Noise prewhitening and SNR |

## Dynamic and quantitative workflows

The Course's encodings applied to a time series and to quantitative mapping.
{doc}`/explanation/non-cartesian` and {doc}`/explanation/nonlinear` give the
background.

| Tour | Contents |
| --- | --- |
| {doc}`/auto_examples/08-workflows/01-dynamic-golden-angle` | Dynamic golden-angle radial reconstruction with temporal total variation |
| {doc}`/auto_examples/08-workflows/02-subspace-t1-mapping` | Subspace $T_1$ mapping from a continuous inversion-recovery scan |
| {doc}`/auto_examples/08-workflows/03-maps-from-scanner-images` | $T_2$ mapping from DICOM magnitude images |

## Learning workflows

Training and use of learned regularizers beyond lessons 10 and 11. {doc}`/explanation/learned-reconstruction`
explains where a network enters a reconstruction. These tours also need:

```bash
pip install lightning torchio monai deepinv
```

| Tour | Contents |
| --- | --- |
| {doc}`/auto_examples/09-learning-workflows/01-networks-for-complex-volumes` | Networks for complex multi-contrast volumes; patchwise execution |
| {doc}`/auto_examples/09-learning-workflows/02-staged-training` | Staged training of unrolled networks |
| {doc}`/auto_examples/09-learning-workflows/03-self-supervised-training` | Self-supervised training from undersampled data |
| {doc}`/auto_examples/09-learning-workflows/04-annealed-plug-and-play` | Plug-and-play with an annealed noise level |
| {doc}`/auto_examples/09-learning-workflows/05-uncertainty` | Calibrated voxel-wise uncertainty |

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
