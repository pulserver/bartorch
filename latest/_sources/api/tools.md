# BART commands as functions

`bartorch.tools` exposes BART's command-line applications that have no
operator or pipeline counterpart here, one function per command, run in this
process.  Array arguments and results are C-order tensors in BART's dimension
order reversed, and axis arguments are axis indices rather than BART bitmasks;
the results of a BART command are not recorded by autograd.  Functions marked *wrapped* have a
hand-written signature; the others are generated from BART's declaration of the
command and take its options under their long names.  Reconstructions are
{mod}`bartorch.apps`, or an encoding from {mod}`bartorch.linop` under a solver
from {mod}`bartorch.optim`; {doc}`../explanation/execution-model` compares the
two routes.  The last two sections, corrections and rigid motion, hold
functions and classes with no BART command behind them.

```{eval-rst}
.. currentmodule:: bartorch.tools
```

## Simulation

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.phantom` | Analytical phantom as an image or as k-space, optionally with coils or along a trajectory (wrapped) |
| {obj}`~bartorch.tools.coils` | Analytical coil sensitivities |
| {obj}`~bartorch.tools.fakeksp` | K-space from an image and sensitivities |
| {obj}`~bartorch.tools.noise` | Additive complex Gaussian noise |

## Sampling and trajectories

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.traj` | Cartesian, radial and golden-angle trajectories in grid units (wrapped) |
| {obj}`~bartorch.tools.grid` | Sampling grid coordinates in image space or k-space |
| {obj}`~bartorch.tools.pattern` | Sampling pattern of a k-space array |
| {obj}`~bartorch.tools.poisson` | Poisson-disc sampling pattern |
| {obj}`~bartorch.tools.upat` | Regular undersampling pattern with a fully sampled centre |
| {obj}`~bartorch.tools.raga` | Indices of a RAGA (rational approximation of the golden angle) radial ordering |
| {obj}`~bartorch.tools.grog` | Radial data gridded onto a Cartesian grid by GROG |
| {obj}`~bartorch.tools.psf` | Point spread function of a trajectory |
| {obj}`~bartorch.tools.wavepsf` | Wave-CAIPI point spread function in hybrid space |
| {obj}`~bartorch.tools.estdims` | Image dimensions implied by a non-Cartesian trajectory |
| {obj}`~bartorch.tools.estdelay` | Gradient-delay estimation from radial data |
| {obj}`~bartorch.tools.trajcor` | Gradient-delay correction of a trajectory |
| {obj}`~bartorch.tools.rmfreq` | Removal of an angle-dependent frequency from radial data |
| {obj}`~bartorch.tools.bin` | Binning of data by labels, including respiratory and cardiac quadrature binning |
| {obj}`~bartorch.tools.ssa` | Singular spectrum analysis (SSA-FARY) of a time series |

## Coil calibration and compression

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.ecalib` | Sensitivities by ESPIRiT (wrapped) |
| {obj}`~bartorch.tools.caldir` | Sensitivities directly from the low-resolution k-space centre (wrapped) |
| {obj}`~bartorch.tools.calmat` | Calibration matrix of the k-space centre |
| {obj}`~bartorch.tools.ecaltwo` | Second stage of ESPIRiT calibration |
| {obj}`~bartorch.tools.walsh` | Walsh coil combination, for use with {obj}`~bartorch.tools.ecaltwo` |
| {obj}`~bartorch.tools.nlinv` | Image and sensitivities jointly by nonlinear inversion (wrapped) |
| {obj}`~bartorch.tools.ncalib` | Sensitivities from non-Cartesian data by nonlinear inversion (ENLIVE) |
| {obj}`~bartorch.tools.cc` | Coil compression matrix (SVD, geometric or ESPIRiT) |
| {obj}`~bartorch.tools.ccapply` | Application of a coil compression matrix |
| {obj}`~bartorch.tools.rovir` | Coil compression by region-optimized virtual coils (ROVir) |
| {obj}`~bartorch.tools.whiten` | Noise prewhitening from a noise measurement |
| {obj}`~bartorch.tools.estvar` | Noise variance of white Gaussian noise |
| {obj}`~bartorch.tools.phasepole` | Detection of phase poles in sensitivities |

## Preprocessing, registration and metrics

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.fovshift` | Field-of-view shift of k-space by a linear phase (wrapped) |
| {obj}`~bartorch.tools.homodyne` | Homodyne partial-Fourier reconstruction of asymmetrically sampled k-space |
| {obj}`~bartorch.tools.affine_transform` | Resampling on an affinely mapped grid (wrapped) |
| {obj}`~bartorch.tools.warp` | Resampling through a displacement field (wrapped) |
| {obj}`~bartorch.tools.register_affine` | Affine registration by mutual information (wrapped) |
| {obj}`~bartorch.tools.register_nonrigid` | Non-rigid registration by greedy SyN or TV-L1 optical flow (wrapped) |
| {obj}`~bartorch.tools.estimate_shift` | Sub-voxel translation from the phase of the cross-spectrum (wrapped) |
| {obj}`~bartorch.tools.nrmse` | Normalized root-mean-square error, optionally after least-squares scaling (wrapped) |
| {obj}`~bartorch.tools.mse` | Mean squared error (wrapped) |
| {obj}`~bartorch.tools.psnr` | Peak signal-to-noise ratio of the magnitudes (wrapped) |
| {obj}`~bartorch.tools.ssim` | Structural similarity of the magnitudes (wrapped) |
| {obj}`~bartorch.tools.roi_stat` | Statistic over a region of interest (wrapped) |

## Low-rank completion

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.sake` | SAKE: low-rank matrix completion of k-space |
| {obj}`~bartorch.tools.lrmatrix` | Multi-scale low-rank matrix completion |

## Corrections

Corrections applied to data or images outside the reconstruction.  Bias field
and gradient nonlinearity correction resample with SimpleITK
(`pip install 'bartorch[correct]'`), and spiral deblurring uses Triton on a
CUDA device when it is installed.  Susceptibility correction runs PyHySCO,
which is GPL-3.0-only, is not distributed with bartorch and is imported only
when {obj}`~bartorch.tools.correct_susceptibility` is called
(`pip install 'bartorch[pyhysco]'`).

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.epi_ramp_operator` | Band-limited resampling of a ramp-sampled EPI readout onto a uniform grid |
| {obj}`~bartorch.tools.estimate_epi_phase` | Odd/even phase of an EPI readout, fitted to a three-line navigator |
| {obj}`~bartorch.tools.correct_lines` | Reversal of EPI lines into forward readout order, with the odd/even phase removed |
| {obj}`~bartorch.tools.bias_field_correct` | Receive-field (bias) correction by N4 |
| {obj}`~bartorch.tools.GradientCoefficients` | Spherical-harmonic coefficient table of a gradient coil |
| {obj}`~bartorch.tools.CoefficientAccessor` | Protocol for a coefficient table that is not a file |
| {obj}`~bartorch.tools.Gradunwarp` | Gradient nonlinearity correction of images from a coefficient table |
| {obj}`~bartorch.tools.field_map_from_phase` | Off-resonance field map from the phase of single-echo coil images |
| {obj}`~bartorch.tools.ReadoutTiming` | Readout time of a spiral arm as a function of k-space radius |
| {obj}`~bartorch.tools.SpiralTransfer` | Separable factorization of the off-resonance transfer of a spiral readout |
| {obj}`~bartorch.tools.fit_transfer` | Fit of a {obj}`~bartorch.tools.SpiralTransfer` to a readout's time map |
| {obj}`~bartorch.tools.deblur` | Off-resonance deblurring of a spiral image |
| {obj}`~bartorch.tools.correct_susceptibility` | Susceptibility distortion correction from a reversed phase-encoding pair, by PyHySCO |
| {obj}`~bartorch.tools.SusceptibilityCorrection` | Corrected pair and displacement field from {obj}`~bartorch.tools.correct_susceptibility` |

## Rigid motion from navigators

Plane reconstruction by the non-uniform transform of this package, rigid
registration by SimpleITK (`pip install 'bartorch[motion]'`), and a
constant-velocity extended Kalman filter over the registered poses.

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.reconstruct_navigator` | Magnitude images of a navigator's planes by density-compensated adjoint NUFFT |
| {obj}`~bartorch.tools.RigidRegistration` | Multi-resolution rigid registration of magnitude images |
| {obj}`~bartorch.tools.RigidMotionEstimate` | 2D or 3D rigid transform with its covariance |
| {obj}`~bartorch.tools.RigidMotionEKF` | Constant-velocity extended Kalman filter over registered rigid poses |
| {obj}`~bartorch.tools.NavigatorMotionTracker` | Six-degree-of-freedom pose from a navigator's 2D planes, filtered over time |
