# Course

Eleven lessons that take you from a k-space tensor to an image, then through
the reconstructions BART is known for, and finally to writing and training
your own. Read them in order: each one starts from what the previous one built.

| Lesson | The question it answers |
| --- | --- |
| {doc}`1. Your data as tensors </auto_examples/01-basics/01-tensors-and-commands>` | How does BART's data look in Python, and how do I call a BART command on it? |
| {doc}`2. From k-space to an image </auto_examples/01-basics/02-from-kspace-to-image>` | How do I reconstruct an undersampled Cartesian scan, step by step? |
| {doc}`3. Coil sensitivities </auto_examples/02-parallel-imaging/01-coil-calibration>` | Where do the coil maps come from, and how much does their quality matter? |
| {doc}`4. Joint reconstruction of image and sensitivity maps </auto_examples/02-parallel-imaging/02-nonlinear-inversion>` | What do I do when the calibration region is too small to estimate the maps? |
| {doc}`5. Choosing a regularizer </auto_examples/03-regularization/01-regularized-reconstruction>` | Tikhonov, wavelets or total variation, and how strong? |
| {doc}`6. Custom encoding and solver </auto_examples/03-regularization/02-operators-and-solvers>` | How do I write the reconstruction myself, as an operator and a solver, and get exactly `pics`? |
| {doc}`7. Basics of non-Cartesian imaging </auto_examples/04-non-cartesian/01-trajectories-and-transforms>` | What are a trajectory, the NUFFT, density compensation and the point spread function? |
| {doc}`8. Radial SENSE </auto_examples/04-non-cartesian/02-radial-sense>` | How do I reconstruct an undersampled golden-angle radial scan? |
| {doc}`9. Nonlinear inversion for T2 mapping </auto_examples/05-model-based/01-quantitative-models>` | Why fit a signal model to the raw data instead of to the images? |
| {doc}`10. Plug-and-play reconstruction </auto_examples/06-learning/01-plug-and-play>` | How do I plug a pretrained network into BART's iterations, with no training? |
| {doc}`11. Training an unrolled network </auto_examples/06-learning/02-modl-with-admm>` | How do I train a network through BART's ADMM? |

By the end you can reconstruct Cartesian and non-Cartesian data with BART's
methods, write an encoding BART has no command for, and put a network inside
the reconstruction.

Lessons 10 and 11 also need:

```bash
pip install lightning torchio monai deepinv
```

Lesson 10 downloads the DRUNet weights that `deepinv` distributes.

```{toctree}
:hidden:

/auto_examples/01-basics/01-tensors-and-commands
/auto_examples/01-basics/02-from-kspace-to-image
/auto_examples/02-parallel-imaging/01-coil-calibration
/auto_examples/02-parallel-imaging/02-nonlinear-inversion
/auto_examples/03-regularization/01-regularized-reconstruction
/auto_examples/03-regularization/02-operators-and-solvers
/auto_examples/04-non-cartesian/01-trajectories-and-transforms
/auto_examples/04-non-cartesian/02-radial-sense
/auto_examples/05-model-based/01-quantitative-models
/auto_examples/06-learning/01-plug-and-play
/auto_examples/06-learning/02-modl-with-admm
```
