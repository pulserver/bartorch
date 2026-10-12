# Examples

Every page here runs when the documentation is built, so every figure and
every number on it comes from the code you see. Each page has an *Open in
Colab* button, so you need nothing but a browser to run it.
The ideas behind them are in {doc}`/explanation/index`, and every function they
call is in {doc}`/api/index`.

## Course

Eleven lessons from a k-space tensor to a learned reconstruction, read in order
({doc}`course`).

| Lesson | Topics |
| --- | --- |
| {doc}`Your data as tensors </auto_examples/01-basics/01-tensors-and-commands>` | Array layout, BART commands on tensors |
| {doc}`From k-space to an image </auto_examples/01-basics/02-from-kspace-to-image>` | Cartesian SENSE with wavelet regularization |
| {doc}`Coil sensitivities </auto_examples/02-parallel-imaging/01-coil-calibration>` | ESPIRiT calibration, map quality |
| {doc}`Joint reconstruction of image and sensitivity maps </auto_examples/02-parallel-imaging/02-nonlinear-inversion>` | Nonlinear inversion with a small calibration region |
| {doc}`Choosing a regularizer </auto_examples/03-regularization/01-regularized-reconstruction>` | Tikhonov, wavelet and TV penalties, regularization weight |
| {doc}`Custom encoding and solver </auto_examples/03-regularization/02-operators-and-solvers>` | Encoding operators and solvers, `pics` assembled by hand |
| {doc}`Basics of non-Cartesian imaging </auto_examples/04-non-cartesian/01-trajectories-and-transforms>` | Trajectories, NUFFT, density compensation, point spread function |
| {doc}`Radial SENSE </auto_examples/04-non-cartesian/02-radial-sense>` | Undersampled golden-angle radial SENSE |
| {doc}`Nonlinear inversion for T2 mapping </auto_examples/05-model-based/01-quantitative-models>` | Model-based T2 mapping from raw data |
| {doc}`Plug-and-play reconstruction </auto_examples/06-learning/01-plug-and-play>` | Pretrained denoiser inside ADMM and FISTA |
| {doc}`Training an unrolled network </auto_examples/06-learning/02-modl-with-admm>` | Unrolled network trained through ADMM (MoDL) |

## Tours

Corrections, dynamic and quantitative workflows, and learning workflows, each
standalone ({doc}`tours`).

| Tour | Topics |
| --- | --- |
| {doc}`/auto_examples/07-tours/01-readout-oversampling-and-apodization` | Readout oversampling removal, Fermi and Hann apodization |
| {doc}`/auto_examples/07-tours/02-epi-ghost-and-ramp-sampling` | EPI Nyquist ghost correction, ramp-sampling regridding |
| {doc}`/auto_examples/07-tours/03-bias-field` | Receive bias field estimation (N4) |
| {doc}`/auto_examples/07-tours/04-gradient-nonlinearity` | Gradient nonlinearity correction from spherical-harmonic coefficients |
| {doc}`/auto_examples/07-tours/05-spiral-deblurring` | Spiral off-resonance deblurring: multifrequency interpolation, time segmentation |
| {doc}`/auto_examples/07-tours/06-navigator-motion` | Rigid head motion from orthogonal navigator planes |
| {doc}`/auto_examples/07-tours/07-epi-susceptibility-distortion` | EPI susceptibility distortion correction from reversed phase encoding |
| {doc}`/auto_examples/07-tours/08-noise-prewhitening` | Channel noise prewhitening and SNR |
| {doc}`/auto_examples/08-workflows/01-dynamic-golden-angle` | Dynamic golden-angle radial reconstruction with temporal TV |
| {doc}`/auto_examples/08-workflows/02-subspace-t1-mapping` | Subspace-constrained $T_1$ mapping from inversion recovery |
| {doc}`/auto_examples/08-workflows/03-maps-from-scanner-images` | $T_2$ fitting of DICOM magnitude images |
| {doc}`/auto_examples/09-learning-workflows/01-networks-for-complex-volumes` | Networks for complex multi-contrast volumes, patchwise execution |
| {doc}`/auto_examples/09-learning-workflows/02-staged-training` | Staged training of an unrolled network |
| {doc}`/auto_examples/09-learning-workflows/03-self-supervised-training` | Self-supervised training from undersampled data |
| {doc}`/auto_examples/09-learning-workflows/04-annealed-plug-and-play` | Plug-and-play with an annealed noise level |
| {doc}`/auto_examples/09-learning-workflows/05-uncertainty` | Calibrated voxel-wise uncertainty |

## Requirements

The scripts need a built `bartorch` with its `io` extra, `brainweb-dl`, which
downloads the BrainWeb phantoms several examples build their images from, and
`matplotlib` and `cmap` for the figures:

```bash
pip install 'bartorch[io]' brainweb-dl matplotlib cmap
```

The learned-regularization lessons and tours additionally require `lightning`,
`torchio`, `monai` and `deepinv`, and the corrections tours `SimpleITK`
and `PyHySCO`; the {doc}`course` and {doc}`tours` pages name what each needs.

```{toctree}
:hidden:

course
tours
```
