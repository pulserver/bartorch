# Examples

Reconstructions executed when the documentation is built, so every figure and
printed number is produced by the code shown.  The course is read in order:
each lesson states its aim and learning objectives and builds on the lessons
before it.  The tours are standalone.  Concepts are in
{doc}`../explanation/index`, interfaces in {doc}`../api/index`.

## Course

| Section | Lesson | Covers |
| --- | --- | --- |
| {doc}`Basics <../auto_examples/01-basics/index>` | {doc}`../auto_examples/01-basics/01-tensors-and-commands` | Tensor shapes and BART's dimensions, the analytical phantom, BART's FFT against NumPy, the command line |
| | {doc}`../auto_examples/01-basics/02-from-kspace-to-image` | Undersampled Cartesian k-space to an image: channel compression, ESPIRiT, {func}`bartorch.apps.pics` |
| {doc}`Parallel imaging <../auto_examples/02-parallel-imaging/index>` | {doc}`../auto_examples/02-parallel-imaging/01-coil-calibration` | Sensitivities by `caldir`, ESPIRiT and nonlinear inversion, and the SENSE reconstructions they give |
| | {doc}`../auto_examples/02-parallel-imaging/02-nonlinear-inversion` | Image and sensitivities estimated jointly by iteratively regularized Gauss-Newton |
| | {doc}`../auto_examples/02-parallel-imaging/03-noise-prewhitening` | Correlated channel noise, prewhitening, and the pseudo-replica SNR |
| {doc}`Regularization <../auto_examples/03-regularization/index>` | {doc}`../auto_examples/03-regularization/01-regularized-reconstruction` | Tikhonov, wavelet and total-variation terms, and the choice of their weight |
| | {doc}`../auto_examples/03-regularization/02-operators-and-solvers` | The same reconstruction as an encoding operator and a solver |
| {doc}`Non-Cartesian imaging <../auto_examples/04-non-cartesian/index>` | {doc}`../auto_examples/04-non-cartesian/01-trajectories-and-transforms` | Trajectories, the non-uniform transform, density compensation, the point spread function |
| | {doc}`../auto_examples/04-non-cartesian/02-radial-sense` | Radial SENSE with sensitivities estimated from the radial data |
| | {doc}`../auto_examples/04-non-cartesian/03-dynamic-golden-angle` | A golden-angle acquisition reconstructed as a time series |
| {doc}`Model-based reconstruction <../auto_examples/05-model-based/index>` | {doc}`../auto_examples/05-model-based/01-subspace-t1-mapping` | Subspace-constrained inversion recovery and $T_1$ mapping |
| | {doc}`../auto_examples/05-model-based/02-quantitative-models` | $T_2$ maps from multi-echo k-space through a signal model |
| {doc}`Learned regularization <../auto_examples/06-learning/index>` | {doc}`../auto_examples/06-learning/01-plug-and-play` | A pretrained denoiser as the proximal step of ADMM and FISTA |
| | {doc}`../auto_examples/06-learning/02-modl-with-admm` | MoDL unrolled on BART's ADMM and trained end to end |
| | {doc}`../auto_examples/06-learning/03-networks-for-complex-volumes` | A U-Net for complex multi-contrast volumes, trained on patches and applied patch by patch |
| | {doc}`../auto_examples/06-learning/04-staged-training` | An unrolled network trained in stages: denoiser, per iteration, end to end |
| | {doc}`../auto_examples/06-learning/05-self-supervised-training` | The same network trained from undersampled k-space alone |
| | {doc}`../auto_examples/06-learning/06-annealed-plug-and-play` | A noise-conditioned denoiser in ADMM with an annealed noise level |
| | {doc}`../auto_examples/06-learning/07-uncertainty` | Voxel-wise error bars from dropout and k-space subsets, calibrated to a coverage |

## Tours

| Tour | Covers |
| --- | --- |
| {doc}`../auto_examples/07-tours/01-readout-oversampling-and-apodization` | Removal of readout oversampling; Gibbs ringing and resolution with Fermi and Hann apodization |
| {doc}`../auto_examples/07-tours/02-epi-ghost-and-ramp-sampling` | Nyquist ghost correction from a three-line navigator; resampling of ramp-sampled readouts |
| {doc}`../auto_examples/07-tours/03-bias-field` | N4 correction of the receive bias field of a head array |
| {doc}`../auto_examples/07-tours/04-gradient-nonlinearity` | Geometric and intensity correction of gradient nonlinearity from spherical-harmonic coefficients |
| {doc}`../auto_examples/07-tours/05-spiral-deblurring` | Spiral off-resonance deblurring by multifrequency interpolation and time-segmented reconstruction |
| {doc}`../auto_examples/07-tours/06-navigator-motion` | Rigid head motion from three orthogonal navigator planes, filtered across a scan |
| {doc}`../auto_examples/07-tours/07-epi-susceptibility-distortion` | EPI susceptibility distortion corrected from a reversed phase-encoding pair |

The scripts need `pip install bartorch brainweb-dl matplotlib cmap`; the
learned-regularization section also `lightning torchio monai deepinv`, and the
tours `SimpleITK` and `PyHySCO`.  Each page can be downloaded as a script or a
notebook.

```{toctree}
:hidden:

../auto_examples/01-basics/index
../auto_examples/02-parallel-imaging/index
../auto_examples/03-regularization/index
../auto_examples/04-non-cartesian/index
../auto_examples/05-model-based/index
../auto_examples/06-learning/index
../auto_examples/07-tours/index
```
