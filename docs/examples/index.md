# Examples

Reconstructions executed when the documentation is built, so every figure and
printed number on these pages is produced by the code shown.  The course is
read in order: each lesson states its aim and builds on the lessons before it.
The tours are standalone.  Every page can be downloaded as a Python script or
a notebook, or opened in Colab.  The concepts are in
{doc}`../explanation/index`, and the interfaces in {doc}`../api/index`.

The scripts need `pip install 'bartorch[io]' brainweb-dl matplotlib cmap`;
`brainweb-dl` downloads the BrainWeb phantoms several examples build their
images from.

## Basics

```{eval-rst}
.. include:: 01-basics/README.rst
   :start-line: 3
```

| Lesson | Covers |
| --- | --- |
| {doc}`../auto_examples/01-basics/01-tensors-and-commands` | Tensor shapes and BART's dimensions, the analytical phantom, BART's FFT against NumPy, the command line |
| {doc}`../auto_examples/01-basics/02-from-kspace-to-image` | Undersampled Cartesian k-space to an image: coil compression, ESPIRiT, {func}`bartorch.apps.pics` |

```{toctree}
:hidden:
:caption: Basics

../auto_examples/01-basics/01-tensors-and-commands
../auto_examples/01-basics/02-from-kspace-to-image
```

## Parallel imaging

```{eval-rst}
.. include:: 02-parallel-imaging/README.rst
   :start-line: 3
```

| Lesson | Covers |
| --- | --- |
| {doc}`../auto_examples/02-parallel-imaging/01-coil-calibration` | Coil sensitivity maps by direct division, ESPIRiT and nonlinear inversion, and the SENSE reconstructions they give |
| {doc}`../auto_examples/02-parallel-imaging/02-nonlinear-inversion` | Image and coil sensitivities estimated jointly by iteratively regularized Gauss-Newton |
| {doc}`../auto_examples/02-parallel-imaging/03-noise-prewhitening` | Correlated channel noise, prewhitening, and the SNR by the pseudo-replica method |

```{toctree}
:hidden:
:caption: Parallel imaging

../auto_examples/02-parallel-imaging/01-coil-calibration
../auto_examples/02-parallel-imaging/02-nonlinear-inversion
../auto_examples/02-parallel-imaging/03-noise-prewhitening
```

## Regularization

```{eval-rst}
.. include:: 03-regularization/README.rst
   :start-line: 3
```

| Lesson | Covers |
| --- | --- |
| {doc}`../auto_examples/03-regularization/01-regularized-reconstruction` | Tikhonov, wavelet and total-variation penalties, and the choice of the regularization weight |
| {doc}`../auto_examples/03-regularization/02-operators-and-solvers` | The same reconstruction as an encoding operator and an iterative solver |

```{toctree}
:hidden:
:caption: Regularization

../auto_examples/03-regularization/01-regularized-reconstruction
../auto_examples/03-regularization/02-operators-and-solvers
```

## Non-Cartesian imaging

```{eval-rst}
.. include:: 04-non-cartesian/README.rst
   :start-line: 3
```

| Lesson | Covers |
| --- | --- |
| {doc}`../auto_examples/04-non-cartesian/01-trajectories-and-transforms` | Trajectories, the NUFFT, density compensation, the point spread function |
| {doc}`../auto_examples/04-non-cartesian/02-radial-sense` | Radial SENSE with coil sensitivities estimated from the radial data |
| {doc}`../auto_examples/04-non-cartesian/03-dynamic-golden-angle` | A golden-angle radial acquisition reconstructed as a time series |

```{toctree}
:hidden:
:caption: Non-Cartesian imaging

../auto_examples/04-non-cartesian/01-trajectories-and-transforms
../auto_examples/04-non-cartesian/02-radial-sense
../auto_examples/04-non-cartesian/03-dynamic-golden-angle
```

## Model-based reconstruction

```{eval-rst}
.. include:: 05-model-based/README.rst
   :start-line: 3
```

| Lesson | Covers |
| --- | --- |
| {doc}`../auto_examples/05-model-based/01-subspace-t1-mapping` | Subspace-constrained inversion recovery and $T_1$ mapping |
| {doc}`../auto_examples/05-model-based/02-quantitative-models` | $T_2$ maps from multi-echo k-space through a signal model |
| {doc}`../auto_examples/05-model-based/03-maps-from-scanner-images` | A $T_2$ map from DICOM magnitude images, written back as DICOM and NIfTI |

```{toctree}
:hidden:
:caption: Model-based reconstruction

../auto_examples/05-model-based/01-subspace-t1-mapping
../auto_examples/05-model-based/02-quantitative-models
../auto_examples/05-model-based/03-maps-from-scanner-images
```

## Learned regularization

```{eval-rst}
.. include:: 06-learning/README.rst
   :start-line: 3
```

| Lesson | Covers |
| --- | --- |
| {doc}`../auto_examples/06-learning/01-plug-and-play` | A pretrained denoiser as the proximal operator of ADMM and FISTA |
| {doc}`../auto_examples/06-learning/02-modl-with-admm` | MoDL unrolled on BART's ADMM and trained end to end |
| {doc}`../auto_examples/06-learning/03-networks-for-complex-volumes` | A U-Net for complex multi-contrast volumes, trained on patches and applied patch by patch |
| {doc}`../auto_examples/06-learning/04-staged-training` | An unrolled network trained in stages: denoiser, per iteration, end to end |
| {doc}`../auto_examples/06-learning/05-self-supervised-training` | The same network trained from undersampled k-space alone |
| {doc}`../auto_examples/06-learning/06-annealed-plug-and-play` | A noise-conditioned denoiser in ADMM with an annealed noise level |
| {doc}`../auto_examples/06-learning/07-uncertainty` | Voxel-wise error bars from dropout and k-space subsets, calibrated to a coverage |

```{toctree}
:hidden:
:caption: Learned regularization

../auto_examples/06-learning/01-plug-and-play
../auto_examples/06-learning/02-modl-with-admm
../auto_examples/06-learning/03-networks-for-complex-volumes
../auto_examples/06-learning/04-staged-training
../auto_examples/06-learning/05-self-supervised-training
../auto_examples/06-learning/06-annealed-plug-and-play
../auto_examples/06-learning/07-uncertainty
```

## Tours

```{eval-rst}
.. include:: 07-tours/README.rst
   :start-line: 3
```

| Tour | Covers |
| --- | --- |
| {doc}`../auto_examples/07-tours/01-readout-oversampling-and-apodization` | Removal of readout oversampling; Gibbs ringing and resolution with Fermi and Hann apodization |
| {doc}`../auto_examples/07-tours/02-epi-ghost-and-ramp-sampling` | Nyquist ghost correction from a three-line navigator; resampling of ramp-sampled readouts |
| {doc}`../auto_examples/07-tours/03-bias-field` | N4 correction of the receive bias field of a head array |
| {doc}`../auto_examples/07-tours/04-gradient-nonlinearity` | Geometric and intensity correction of gradient nonlinearity from spherical-harmonic coefficients |
| {doc}`../auto_examples/07-tours/05-spiral-deblurring` | Spiral off-resonance deblurring by multifrequency interpolation and time-segmented reconstruction |
| {doc}`../auto_examples/07-tours/06-navigator-motion` | Rigid head motion from three orthogonal navigator planes, filtered across a scan |
| {doc}`../auto_examples/07-tours/07-epi-susceptibility-distortion` | EPI susceptibility distortion corrected from a reversed phase-encoding pair |

```{toctree}
:hidden:
:caption: Tours

../auto_examples/07-tours/01-readout-oversampling-and-apodization
../auto_examples/07-tours/02-epi-ghost-and-ramp-sampling
../auto_examples/07-tours/03-bias-field
../auto_examples/07-tours/04-gradient-nonlinearity
../auto_examples/07-tours/05-spiral-deblurring
../auto_examples/07-tours/06-navigator-motion
../auto_examples/07-tours/07-epi-susceptibility-distortion
```
