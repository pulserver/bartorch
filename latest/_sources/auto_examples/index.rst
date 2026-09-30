:orphan:

Examples
========

Reconstructions executed when the documentation is built: a course in six
sections read in order, and standalone tours of the corrections applied around
a reconstruction.  :doc:`/examples/index` lists every lesson.

Running them needs a built ``bartorch``, ``brainweb-dl``, which downloads the
BrainWeb phantoms several examples build their images from, and
``matplotlib`` and ``cmap`` for the figures::

    pip install bartorch brainweb-dl matplotlib cmap

The learned-regularization section additionally requires ``lightning``,
``torchio``, ``monai`` and ``deepinv``, and downloads the DRUNet weights
``deepinv`` distributes; the tours require ``SimpleITK`` and ``PyHySCO``,
which is GPL-3.0-only and not distributed with bartorch::

    pip install lightning torchio monai deepinv SimpleITK PyHySCO


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Basics
------

The course starts from the data a scanner delivers, multichannel k-space, and
reconstructs an image from it.  The first lesson relates PyTorch tensors to
BART's arrays and commands, simulates an acquisition with BART's analytical
phantom and verifies the centring and scaling of BART's FFT against NumPy.  The
second reconstructs an undersampled Cartesian acquisition through the three
steps every later section builds on: coil compression, calibration of the coil
sensitivity maps by ESPIRiT, and a regularized SENSE reconstruction.  The
signal model is derived in :doc:`/explanation/encoding`, and the array
conventions are stated in :doc:`/explanation/data-layout`.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This first lesson establishes how MR data from BART appear in Python: how a multichannel image and its k-space are laid out as tensors, which Fourier convention BART uses to go from k-space to the image, and how a BART command is called. Every later lesson relies on these conventions. A reconstruction that is off by a factor of \sqrt{N}, or by a half-voxel shift from a misplaced k-space centre, is a consequence of misreading one of them.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_01-tensors-and-commands_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/01-tensors-and-commands`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tensors and commands</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson reconstructs an undersampled Cartesian brain acquisition from its multichannel k-space to a coil-combined image, and shows what each step of a parallel-imaging and compressed-sensing pipeline contributes. Scan time in Cartesian MRI is proportional to the number of phase-encoding lines; skipping lines shortens the scan by the acceleration factor R, but violates the Nyquist criterion and folds the image onto itself. Recovering an unaliased image from such data is what the receive coil array, and prior knowledge of the image, are used for.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_02-from-kspace-to-image_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/02-from-kspace-to-image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">From k-space to image</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Parallel imaging
----------------

Parallel imaging recovers an image from k-space undersampled along the
phase-encoding directions by exploiting the spatial sensitivities of a receive
array.  A SENSE reconstruction is only as accurate as its coil sensitivity
maps, and it propagates the thermal noise of the channels, amplified by the
g-factor.  This section treats both: sensitivity estimation from the
autocalibration (ACS) region, by direct division and by ESPIRiT; joint
estimation of image and sensitivities by nonlinear inversion when the ACS
region is too small for a separate calibration; and prewhitening of correlated
channel noise, evaluated by the SNR of the reconstruction.
:doc:`/explanation/encoding` derives the SENSE model and
:doc:`/explanation/nonlinear` the joint estimation.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson compares three ways of estimating the receive sensitivities of a coil array from the undersampled acquisition itself, and shows how the size of the fully sampled calibration region decides which of them can be used. A SENSE reconstruction [#sense]_ inverts">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_01-coil-calibration_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/01-coil-calibration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Coil sensitivity calibration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson reconstructs an image and the coil sensitivities together from undersampled data whose fully sampled central region is too small for a separate calibration, and then writes the same reconstruction out as a nonlinear operator and a Gauss-Newton solver.">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_02-nonlinear-inversion_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/02-nonlinear-inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonlinear inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson measures how correlated noise between receive channels lowers the signal-to-noise ratio (SNR) of a SENSE reconstruction, and how much of it prewhitening with a noise-only acquisition recovers.">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_03-noise-prewhitening_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/03-noise-prewhitening`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Noise prewhitening</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Regularization
--------------

Beyond the acceleration factor the coil geometry supports, the SENSE problem
is ill-conditioned: a least-squares solution amplifies noise, and the aliasing
of the undersampling is not fully resolved.  A regularized reconstruction adds
prior knowledge of the image as a penalty.  This section compares Tikhonov,
wavelet-sparsity (compressed sensing) and total-variation penalties on one
undersampled acquisition and shows how the regularization weight is chosen.  It
then writes the same reconstruction as an explicit encoding operator and an
iterative solver, the form required by an encoding for which BART has no
application.  :doc:`/explanation/inverse-problems` introduces the formulations
and the algorithms.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson compares three regularization terms on the same undersampled, noisy SENSE acquisition, shows how the choice of the regularization weight trades residual noise and aliasing against loss of detail, and combines two terms in one reconstruction.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_01-regularized-reconstruction_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/01-regularized-reconstruction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Regularized reconstruction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson rebuilds the reconstruction of the previous lessons from its parts -- the encoding operator, the regularization term and the iterative algorithm -- instead of calling a BART application, and shows that the result is identical.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_02-operators-and-solvers_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/02-operators-and-solvers`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Operators and solvers</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Non-Cartesian imaging
---------------------

Radial and spiral trajectories sample k-space off the Cartesian grid.  Their
Fourier transform is a non-uniform FFT (NUFFT), their adjoint approximates an
inverse only after density compensation, and the normal operator of an
iterative reconstruction becomes a convolution with the point spread function
of the trajectory.  This section introduces trajectories, the NUFFT, density
compensation and the point spread function; reconstructs an undersampled
golden-angle radial acquisition by non-Cartesian SENSE, with coil sensitivities
estimated from the radial data; and reconstructs a continuous golden-angle
acquisition as a time series with a temporal regularizer.
:doc:`/explanation/non-cartesian` defines the transform and its accuracy.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson introduces the building blocks of non-Cartesian reconstruction: radial, golden-angle and spiral trajectories, the non-uniform fast Fourier transform (NUFFT) that samples an image along them, the density compensation that an adjoint (gridding) reconstruction needs, and the point spread function (PSF) that describes the undersampling artefacts.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_01-trajectories-and-transforms_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/01-trajectories-and-transforms`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Trajectories and transforms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson reconstructs an undersampled golden-angle radial acquisition with eight receive coils: the density-compensated gridding reconstruction first, then an iterative SENSE reconstruction with coil sensitivities estimated from the radial data themselves, with and without a total-variation penalty. The aim is to see which of the streak artefacts of radial undersampling the coil encoding removes, which the regularization removes, and what each costs.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_02-radial-sense_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/02-radial-sense`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radial SENSE reconstruction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson reconstructs a dynamic contrast-enhanced series from one continuous golden-angle radial acquisition, cut into frames of thirteen spokes each. Each frame on its own is undersampled fifteenfold and cannot be reconstructed; the series can, because consecutive frames are strongly correlated, and a total-variation penalty along the time axis states that correlation. The lesson compares frame-by-frame gridding with this joint reconstruction on the images and on the time-intensity curve a perfusion analysis would use.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_03-dynamic-golden-angle_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/03-dynamic-golden-angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dynamic golden-angle radial MRI</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Model-based reconstruction
--------------------------

Quantitative MRI estimates tissue parameters such as :math:`T_1` and
:math:`T_2` from a series of images acquired at different contrasts.
Reconstructing each contrast separately and fitting a signal model afterwards
ignores the relation between the contrasts that the signal model states.  A
model-based reconstruction places that relation in the forward operator, so
that every contrast constrains the same unknowns.  This section treats the two
standard formulations: a linear subspace model, in which inversion-recovery
signal curves are represented by a few temporal basis functions and
:math:`T_1` is fitted to the coefficient maps, and a nonlinear signal model,
through which :math:`T_2` maps are estimated directly from multi-echo k-space.
:doc:`/explanation/nonlinear` compares the two with reconstruction followed by
a voxel-wise fit.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson estimates a T_1 map from a single continuous inversion-recovery acquisition in which each of four hundred time points is encoded by one radial spoke. The aim is to show how a signal model turns a hopelessly undersampled time series into a well-posed reconstruction: the recovery curves of all plausible T_1 values span a subspace of low dimension, and reconstructing the few coefficients of that subspace instead of the individual frames reduces the number of unknowns by two orders of magnitude.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_01-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/01-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson estimates a T_2 map from an undersampled multi-echo spin-echo acquisition in two ways, and compares them: reconstructing an image per echo and fitting the decay voxel by voxel afterwards, and fitting the signal model directly to the k-space data. The aim is to show why the second, model-based reconstruction, tolerates undersampling that ruins the first.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_02-quantitative-models_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/02-quantitative-models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter maps straight from k-space</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Learned regularization
----------------------

A learned reconstruction replaces the hand-specified regularization term by a
neural network and keeps the encoding operator and the data consistency of the
iterative reconstruction.  This section starts with plug-and-play
reconstruction, in which a pretrained denoiser takes the place of the proximal
operator of ADMM and FISTA without any training, and proceeds to an unrolled
network trained through BART's ADMM (MoDL); networks for complex
multi-contrast volumes, applied patch by patch; staged and self-supervised
training of an unrolled network; plug-and-play with an annealed noise level;
and calibrated voxel-wise uncertainty.
:doc:`/explanation/learned-reconstruction` describes where a network enters a
reconstruction.

The section additionally requires::

    pip install lightning torchio monai deepinv

The first lesson downloads the DRUNet weights ``deepinv`` distributes.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson regularizes an undersampled, noisy Cartesian SENSE reconstruction with a pretrained image denoiser in place of a specified penalty, and compares the result with total-variation regularization of the same data. The aim is to show how a denoiser enters a proximal iteration, what it improves on a hand-crafted penalty, and how its noise level plays the role of the regularization weight.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_01-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/01-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plug-and-play denoisers</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson trains an unrolled reconstruction network for undersampled Cartesian SENSE: a small convolutional denoiser placed in the proximal step of BART&#x27;s alternating-direction iteration, with the whole iteration trained end to end against fully sampled images. The aim is to show how a learned regularizer is combined with the physical encoding model -- coil sensitivities, Fourier transform and sampling pattern -- so that the network only has to remove what the data leave undetermined, and how such a network is trained with standard tools.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_02-modl-with-admm_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/02-modl-with-admm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MoDL, on BART's ADMM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train a 3D convolutional denoiser on patches of a complex, multi-contrast brain volume, apply it to a whole volume of another subject patch by patch, as it would run on a scanner GPU too small for the volume, and check that the patch boundaries leave no visible seams.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_03-networks-for-complex-volumes_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/03-networks-for-complex-volumes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Networks for complex volumes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train an unrolled reconstruction network for fourfold undersampled, eight-channel Cartesian brain data within the memory of one iteration, and show that it removes the residual aliasing and the g-factor noise that CG-SENSE leaves at this acceleration.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_04-staged-training_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/04-staged-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Staged training of an unrolled network</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train the unrolled network of 04-staged-training from undersampled k-space alone, with no fully sampled reference, and measure how much of the supervised network&#x27;s image quality it retains.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_05-self-supervised-training_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/05-self-supervised-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Training without a reference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train one denoiser on images alone, without any encoding, and use it as the regularizer of an ADMM reconstruction at any undersampling, with the denoising strength decreasing over the iterations; show that it holds up at an acceleration where CG-SENSE breaks down.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_06-annealed-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/06-annealed-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Annealed plug-and-play</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Attach a voxel-wise error bar to a learned reconstruction of undersampled data, calibrated so that it contains the true error in a stated fraction of voxels, and see where in the head the reconstruction is least certain.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_07-uncertainty_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/07-uncertainty`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Uncertainty estimation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Tours
-----

Corrections applied to the data before reconstruction or to the image after
it, each shown on its own and independent of the course: removal of readout
oversampling and apodization, EPI Nyquist-ghost correction and regridding of
ramp-sampled readouts, receive bias-field correction, correction of the
geometric distortion caused by gradient nonlinearity, off-resonance deblurring
of spiral images, and rigid head-motion tracking from navigators.  Each tour
simulates the artefact from a known ground truth, so that the correction is
evaluated against it.

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
of an EPI image from a reversed phase-encoding pair.

The section additionally requires::

    pip install SimpleITK PyHySCO


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two operations are applied to Cartesian k-space before the image is reconstructed. Readout oversampling doubles the field of view along the frequency-encoding direction, and is removed so that the image has the prescribed matrix. Truncation of k-space at the edge of the acquired matrix convolves the image with a sinc, whose side lobes appear as Gibbs ringing parallel to every sharp edge; an apodization window reduces the ringing at the cost of spatial resolution.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_01-readout-oversampling-and-apodization_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/01-readout-oversampling-and-apodization`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Readout oversampling and apodization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An echo-planar readout acquires k-space in a train of lines of alternating readout gradient polarity, and two corrections are applied before the lines form a Cartesian k-space. A timing error between the readout gradient and the ADC, and eddy currents, displace the echoes of the reversed lines relative to the forward ones; the resulting odd/even phase produces a Nyquist ghost, a copy of the object displaced by half the field of view along the phase-encoding direction. Sampling during the ramps of the readout gradient shortens the echo spacing, and the samples, uniform in time, are not uniform in k_x.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_02-epi-ghost-and-ramp-sampling_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/02-epi-ghost-and-ramp-sampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">EPI Nyquist ghost and ramp sampling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A root-sum-of-squares combination of the images of a receive array is the object weighted by the root sum of squares of the coil sensitivities. That weighting is smooth, multiplicative and largest near the elements, and it shades the image: the same tissue appears brighter near the array than far from it, which biases segmentation, intensity-based registration and any quantitative comparison across the field of view.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_03-bias-field_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/03-bias-field`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Receive bias field</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spatial encoding assumes that each gradient field varies linearly with position. The field of a real gradient coil departs from linearity with the distance from isocentre, so a spin is encoded at a position displaced from its true one: the image is warped, by a few millimetres at the edge of a head-sized field of view and by centimetres at the edge of a body-sized one, and the voxel volume changes with the warp. The displacement is a property of the coil, stated by its manufacturer as the coefficients of a spherical-harmonic expansion of each gradient field [#janke]_; the correction evaluates that expansion at every voxel and resamples the image at the positions where the voxels were encoded.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_04-gradient-nonlinearity_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/04-gradient-nonlinearity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient nonlinearity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A spiral readout acquires each k-space radius at its own time, so a spin off resonance accrues a phase that varies over k-space: its image is blurred into a ring rather than shifted, as it would be along the readout of a Cartesian acquisition. With readouts of tens of milliseconds, the B_0 inhomogeneity near air-tissue interfaces is enough to smear the temporal and orbitofrontal cortex over several voxels.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_05-spiral-deblurring_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/05-spiral-deblurring`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Off-resonance correction of spiral imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Head motion during a scan changes the position of the anatomy between the readouts that encode it, and the image acquires blurring and ghosting. Prospective correction updates the imaging field of view with the measured head pose before each readout; it requires a measurement of the rigid pose with six degrees of freedom, repeated during the scan. A navigator is a short, low-resolution acquisition interleaved with the imaging readouts, and its registration against the first navigator of the scan measures how the head has moved since [#ehman]_.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_06-navigator-motion_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/06-navigator-motion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rigid head motion from navigators</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In an echo-planar image, the phase-encoding direction is sampled at the echo spacing rather than the dwell time, so its bandwidth per pixel is a few tens of hertz. A spin off resonance by \Delta f is displaced along the phase-encoding axis by \Delta f divided by that bandwidth, which near the frontal sinus and the petrous bone amounts to several voxels at 3 T: the orbitofrontal cortex is compressed or stretched, and signal piles up where neighbouring voxels are displaced onto the same location.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_07-epi-susceptibility-distortion_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/07-epi-susceptibility-distortion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Susceptibility distortion in EPI</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_examples/01-basics/index.rst
   /auto_examples/02-parallel-imaging/index.rst
   /auto_examples/03-regularization/index.rst
   /auto_examples/04-non-cartesian/index.rst
   /auto_examples/05-model-based/index.rst
   /auto_examples/06-learning/index.rst
   /auto_examples/07-tours/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
