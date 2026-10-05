:orphan:

Examples
========

Reconstructions executed when the documentation is built, so every figure and
printed number on these pages is produced by the code shown.  Every page can be
downloaded as a Python script or a notebook, or opened in Colab.  The concepts
are in :doc:`/explanation/index`, and the interfaces in :doc:`/api/index`.

The Course is the shortest coherent path that gives a new user the framework's
core mental model and enough practical competence to work independently.  Tours
are useful applications, advanced branches or specialised workflows that are
not necessary for that core competence.

Course
------

The first six sections are the course, read in order.  Each lesson states its
aim and learning objectives and links to the lesson before and after it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Subject
   * - :doc:`01-basics/index`
     - Tensors and BART's dimensions, and a first reconstruction from undersampled Cartesian k-space
   * - :doc:`02-parallel-imaging/index`
     - Coil sensitivity estimation and nonlinear inversion
   * - :doc:`03-regularization/index`
     - Tikhonov, wavelet and total-variation penalties, and the operator-and-solver form
   * - :doc:`04-non-cartesian/index`
     - The NUFFT, density compensation and radial SENSE
   * - :doc:`05-model-based/index`
     - :math:`T_2` estimation through a signal model fitted directly to k-space
   * - :doc:`06-learning/index`
     - Plug-and-play reconstruction and an unrolled network trained through ADMM

Tours
-----

The tours are standalone.  Each opens with its objective and the course
lessons it assumes, and has no previous or next page.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Subject
   * - :doc:`07-tours/index`
     - Corrections applied before and after a reconstruction, and prewhitening of channel noise
   * - :doc:`08-workflows/index`
     - Dynamic golden-angle imaging, subspace :math:`T_1` mapping and maps from DICOM images
   * - :doc:`09-learning-workflows/index`
     - Networks for complex volumes, staged and self-supervised training, annealed plug-and-play and uncertainty

The scripts need a built ``bartorch`` with its ``io`` extra, ``brainweb-dl``, which downloads the
BrainWeb phantoms several examples build their images from, and
``matplotlib`` and ``cmap`` for the figures::

    pip install 'bartorch[io]' brainweb-dl matplotlib cmap

The learned-regularization lessons and tours additionally require ``lightning``,
``torchio``, ``monai`` and ``deepinv``, and the corrections tours ``SimpleITK``
and ``PyHySCO``; each section page names what it needs.


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
maps.  This section treats their estimation from the autocalibration (ACS)
region, by direct division and by ESPIRiT, and the joint estimation of image
and sensitivities by nonlinear inversion when the ACS region is too small for a
separate calibration.  The prewhitening of correlated channel noise, a
correction applied before this estimation, is the tour
:doc:`/auto_examples/07-tours/08-noise-prewhitening`.
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
compensation and the point spread function, and reconstructs an undersampled
golden-angle radial acquisition by non-Cartesian SENSE, with coil sensitivities
estimated from the radial data.  The extension to a time series with a temporal
regularizer is the tour :doc:`/auto_examples/08-workflows/01-dynamic-golden-angle`.
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
that every contrast constrains the same unknowns.  This section fits a
nonlinear signal model directly to multi-echo k-space to estimate :math:`T_2`
maps, and compares the result with a voxel-wise fit of reconstructed images.
The linear subspace formulation and the fit to DICOM images are the tours
:doc:`/auto_examples/08-workflows/02-subspace-t1-mapping` and
:doc:`/auto_examples/08-workflows/03-maps-from-scanner-images`.
:doc:`/explanation/nonlinear` compares the formulations with reconstruction
followed by a voxel-wise fit.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson estimates a T_2 map from an undersampled multi-echo spin-echo acquisition in two ways, and compares them: reconstructing an image per echo and fitting the decay voxel by voxel afterwards, and fitting the signal model directly to the k-space data. The aim is to show why the second, model-based reconstruction, tolerates undersampling that ruins the first.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_01-quantitative-models_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/01-quantitative-models`

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
iterative reconstruction.  This section covers the two ways a network enters
BART's iterations: plug-and-play reconstruction, in which a pretrained denoiser
takes the place of the proximal operator of ADMM and FISTA without any
training, and an unrolled network trained through BART's ADMM (MoDL).
Networks for complex volumes, staged and self-supervised training, annealed
plug-and-play and uncertainty are the tours of
:doc:`/auto_examples/09-learning-workflows/index`.
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


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Tours: acquisition and corrections
----------------------------------

Corrections applied to the data before reconstruction or to the image after
it, each shown on its own and independent of the course: removal of readout
oversampling and apodization, EPI Nyquist-ghost correction and regridding of
ramp-sampled readouts, receive bias-field correction, correction of the
geometric distortion caused by gradient nonlinearity, off-resonance deblurring
of spiral images, rigid head-motion tracking from navigators, correction of
susceptibility distortion in EPI, and prewhitening of correlated channel
noise.  The first six tours and the last simulate the artefact or the noise
from a known ground truth, so that the correction is evaluated against it; the
seventh corrects measured data and evaluates the correction against an
independent field map.

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
of a 3 T EPI pair with reversed phase encoding, downloaded from OpenNeuro. The
eighth measures the signal-to-noise ratio gained by prewhitening a SENSE
reconstruction with a noise-only acquisition.

The section additionally requires the following packages, and the seventh tour
downloads about 2 MB of data into ``~/.cache/bartorch-examples``::

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

    <div class="sphx-glr-thumbcontainer" tooltip="In an echo-planar image, the phase-encoding direction is sampled at the echo spacing rather than the dwell time, so its bandwidth per pixel is a few tens of hertz. A spin off resonance by \Delta f is displaced along the phase-encoding axis by \Delta f divided by that bandwidth, which near the frontal sinus and the petrous bone amounts to several millimetres at 3 T: the orbitofrontal cortex and the temporal poles are compressed or stretched, and signal piles up where neighbouring voxels are displaced onto the same location.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_07-epi-susceptibility-distortion_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/07-epi-susceptibility-distortion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Susceptibility distortion in EPI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example measures how correlated noise between receive channels lowers the signal-to-noise ratio (SNR) of a SENSE reconstruction, and how much of it prewhitening with a noise-only acquisition recovers.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_08-noise-prewhitening_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/08-noise-prewhitening`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Noise prewhitening</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Tours: dynamic and quantitative workflows
-----------------------------------------

Applications of the course's encodings to a time series and to quantitative
mapping, each shown on its own.  The first reconstructs a dynamic
contrast-enhanced series from one continuous golden-angle radial acquisition
with a total-variation penalty along time.  The second estimates a
:math:`T_1` map from a single continuous inversion-recovery acquisition by
reconstructing the coefficients of a linear subspace of the signal curves.  The
third fits a :math:`T_2` decay to magnitude images read from DICOM, as a
scanner exports them, and writes the map back.
:doc:`/explanation/non-cartesian` and :doc:`/explanation/nonlinear` give the
background.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a dynamic contrast-enhanced series from one continuous golden-angle radial acquisition, cut into frames of thirteen spokes each. Each frame on its own is undersampled fifteenfold and cannot be reconstructed; the series can, because consecutive frames are strongly correlated, and a total-variation penalty along the time axis states that correlation. The example compares frame-by-frame gridding with this joint reconstruction on the images and on the time-intensity curve a perfusion analysis would use.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_01-dynamic-golden-angle_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/01-dynamic-golden-angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dynamic golden-angle radial MRI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example estimates a T_1 map from a single continuous inversion-recovery acquisition in which each of four hundred time points is encoded by one radial spoke. The aim is to show how a signal model turns a hopelessly undersampled time series into a well-posed reconstruction: the recovery curves of all plausible T_1 values span a subspace of low dimension, and reconstructing the few coefficients of that subspace instead of the individual frames reduces the number of unknowns by two orders of magnitude.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_02-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/02-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example estimates a T_2 map from the magnitude images a scanner exports, without access to the raw data: a multi-echo spin-echo series is read from DICOM, the decay is fitted voxel by voxel, and the map is written back as a DICOM series of the same study and as a NIfTI volume. The aim is to show the geometry and the acquisition timings passing from the scanner&#x27;s files to the fit and on to the output unchanged, so that the map overlays the images it was computed from.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_03-maps-from-scanner-images_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/03-maps-from-scanner-images`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter maps from scanner images</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Tours: learning workflows
-------------------------

Training strategies and uses of a learned regularizer that follow from the two
lessons of :doc:`/auto_examples/06-learning/index`: networks for complex
multi-contrast volumes applied patch by patch, staged training of an unrolled
network within the memory of one iteration, self-supervised training from
undersampled data alone, plug-and-play with an annealed noise level, and
calibrated voxel-wise uncertainty.
:doc:`/explanation/learned-reconstruction` describes where a network enters a
reconstruction.

The section additionally requires::

    pip install lightning torchio monai deepinv


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train a 3D convolutional denoiser on patches of a complex, multi-contrast brain volume, apply it to a whole volume of another subject patch by patch, as it would run on a scanner GPU too small for the volume, and check that the patch boundaries leave no visible seams.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_01-networks-for-complex-volumes_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/01-networks-for-complex-volumes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Networks for complex volumes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train an unrolled reconstruction network for fourfold undersampled, eight-channel Cartesian brain data within the memory of one iteration, and show that it removes the residual aliasing and the g-factor noise that CG-SENSE leaves at this acceleration.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_02-staged-training_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/02-staged-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Staged training of an unrolled network</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train the unrolled network of 02-staged-training from undersampled k-space alone, with no fully sampled reference, and measure how much of the supervised network&#x27;s image quality it retains.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_03-self-supervised-training_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/03-self-supervised-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Training without a reference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train one denoiser on images alone, without any encoding, and use it as the regularizer of an ADMM reconstruction at any undersampling, with the denoising strength decreasing over the iterations; show that it holds up at an acceleration where CG-SENSE breaks down.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_04-annealed-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/04-annealed-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Annealed plug-and-play</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Attach a voxel-wise error bar to a learned reconstruction of undersampled data, calibrated so that it contains the true error in a stated fraction of voxels, and see where in the head the reconstruction is least certain.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_05-uncertainty_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/05-uncertainty`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Uncertainty estimation</div>
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
   /auto_examples/08-workflows/index.rst
   /auto_examples/09-learning-workflows/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
