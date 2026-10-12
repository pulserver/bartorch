# Examples

Every page here runs when the documentation is built, so every figure and
every number on it comes from the code you see. Each page has an *Open in
Colab* button, so you need nothing but a browser to run it.
The ideas behind them are in {doc}`/explanation/index`, and every function they
call is in {doc}`/api/index`.

## Course

Eleven lessons from a k-space tensor to a learned reconstruction, read in order
({doc}`course`).

| Lesson | The question it answers |
| --- | --- |
| {doc}`Your data as tensors </auto_examples/01-basics/01-tensors-and-commands>` | How does BART's data look in Python, and how do I call a BART command on it? |
| {doc}`From k-space to an image </auto_examples/01-basics/02-from-kspace-to-image>` | How do I reconstruct an undersampled Cartesian scan, step by step? |
| {doc}`Coil sensitivities </auto_examples/02-parallel-imaging/01-coil-calibration>` | Where do the coil maps come from, and how much does their quality matter? |
| {doc}`Joint reconstruction of image and sensitivity maps </auto_examples/02-parallel-imaging/02-nonlinear-inversion>` | What do I do when the calibration region is too small to estimate the maps? |
| {doc}`Choosing a regularizer </auto_examples/03-regularization/01-regularized-reconstruction>` | Tikhonov, wavelets or total variation, and how strong? |
| {doc}`Custom encoding and solver </auto_examples/03-regularization/02-operators-and-solvers>` | How do I write the reconstruction myself, as an operator and a solver, and get exactly `pics`? |
| {doc}`Basics of non-Cartesian imaging </auto_examples/04-non-cartesian/01-trajectories-and-transforms>` | What are a trajectory, the NUFFT, density compensation and the point spread function? |
| {doc}`Radial SENSE </auto_examples/04-non-cartesian/02-radial-sense>` | How do I reconstruct an undersampled golden-angle radial scan? |
| {doc}`Nonlinear inversion for T2 mapping </auto_examples/05-model-based/01-quantitative-models>` | Why fit a signal model to the raw data instead of to the images? |
| {doc}`Plug-and-play reconstruction </auto_examples/06-learning/01-plug-and-play>` | How do I plug a pretrained network into BART's iterations, with no training? |
| {doc}`Training an unrolled network </auto_examples/06-learning/02-modl-with-admm>` | How do I train a network through BART's ADMM? |

## Tours

Corrections, dynamic and quantitative workflows, and learning workflows, each
standalone ({doc}`tours`).

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
| {doc}`/auto_examples/08-workflows/01-dynamic-golden-angle` | Reconstruct a dynamic contrast-enhanced series from one continuous golden-angle radial scan, with total variation along time. |
| {doc}`/auto_examples/08-workflows/02-subspace-t1-mapping` | Map $T_1$ from one continuous inversion-recovery scan, by reconstructing the coefficients of a subspace of the signal curves. |
| {doc}`/auto_examples/08-workflows/03-maps-from-scanner-images` | Fit a $T_2$ decay to magnitude images read from DICOM, as the scanner exports them, and write the map back. |
| {doc}`/auto_examples/09-learning-workflows/01-networks-for-complex-volumes` | Networks for complex multi-contrast volumes, applied patch by patch. |
| {doc}`/auto_examples/09-learning-workflows/02-staged-training` | Training an unrolled network in stages, within the memory of one iteration. |
| {doc}`/auto_examples/09-learning-workflows/03-self-supervised-training` | Self-supervised training from undersampled data alone. |
| {doc}`/auto_examples/09-learning-workflows/04-annealed-plug-and-play` | Plug-and-play with a noise level that anneals over the iterations. |
| {doc}`/auto_examples/09-learning-workflows/05-uncertainty` | Voxel-wise uncertainty, calibrated. |

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
