:orphan:

Gallery source pages
====================

The pages sphinx-gallery generates from the example scripts. The documentation
presents them through :doc:`/gallery/index` and the Course and Tours pages
under it.


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

You start from the data a scanner gives you, multichannel k-space, and end
with an image. Lesson 1 shows how BART's arrays look as PyTorch tensors and how
you call a BART command on them. Lesson 2 reconstructs an undersampled
Cartesian scan through the three steps the rest of the Course builds on:
channel compression, coil sensitivities from ESPIRiT, and a regularized SENSE
reconstruction. :doc:`/explanation/encoding` derives the signal model, and
:doc:`/explanation/data-layout` lists the array layouts.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="If you have used BART before, you know its arrays as CFL files on disk and its commands as programs you call from a shell. bartorch runs the same BART inside your Python process: every command is a Python function that takes and returns torch.Tensor objects, with no files and no subprocess in between. In this lesson you see what your data look like on that side: how a multichannel image and its k-space are laid out as tensors, which Fourier transform takes one to the other, and how a BART command line maps onto a function call.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_01-tensors-and-commands_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/01-tensors-and-commands`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">1. Your data as tensors</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 1 you saw how BART&#x27;s arrays look as tensors and how its FFT takes a fully sampled k-space to an image. A real scan is rarely fully sampled: you skip phase-encoding lines to shorten it, and the image folds onto itself. In this lesson you reconstruct such a scan, one step at a time, and after each step you look at what it did to the image.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_02-from-kspace-to-image_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/02-from-kspace-to-image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2. From k-space to an image</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Parallel imaging
----------------

A SENSE reconstruction is only as good as its coil sensitivities. In lesson 3
you estimate them from the autocalibration region, by direct division and by
ESPIRiT, and see what a poor estimate does to the image. In lesson 4 the
calibration region is too small for that, and you estimate the image and the
sensitivities together by nonlinear inversion. If your channels carry
correlated noise, whiten them first: the tour
:doc:`/auto_examples/07-tours/08-noise-prewhitening` shows how.
:doc:`/explanation/encoding` and :doc:`/explanation/nonlinear` give the
background.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 2 you estimated the coil sensitivities with ESPIRiT from a block of 24 central lines and passed them to pics without examining them. SENSE [#sense]_ solves">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_01-coil-calibration_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/01-coil-calibration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3. Coil sensitivities</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 3 you saw ESPIRiT fail with eight ACS lines while nonlinear inversion kept working, because it fits the sensitivities to every acquired sample rather than to the ACS block alone. In this lesson you keep only six central lines, too few for any separate calibration, and reconstruct the image and the sensitivities together.">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_02-nonlinear-inversion_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/02-nonlinear-inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">4. Joint reconstruction of image and sensitivity maps</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Regularization
--------------

Past the acceleration your coils can support, SENSE amplifies noise and leaves
aliasing behind. A regularizer adds what you know about the image. In lesson 5
you compare Tikhonov, wavelet and total-variation penalties on one scan and
choose their weight. In lesson 6 you write the same reconstruction yourself, as
an encoding operator and a solver, which is how you reconstruct anything BART
has no command for. :doc:`/explanation/regularizers` and :doc:`/explanation/solvers` explain
the penalties and the iterations.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 2 you reconstructed an undersampled scan with a wavelet penalty at a weight of 0.004, both fixed without justification. In this lesson you choose them. You reconstruct one noisy, undersampled SENSE acquisition with three different penalties, sweep the weight of each, and compare under- and over-regularized images. Then you combine two penalties in one reconstruction.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_01-regularized-reconstruction_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/01-regularized-reconstruction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">5. Choosing a regularizer</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 5 you handed pics the data, the maps and a penalty, and it did the rest. In this lesson you assemble the same reconstruction from its parts. You build the SENSE encoding operator, check its adjoint, run the solver on it, and obtain exactly what pics returns, bit for bit. Then you extend the operator with one written in Python, and differentiate through it.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_02-operators-and-solvers_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/02-operators-and-solvers`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6. Custom encoding and solver</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Non-Cartesian imaging
---------------------

Radial and spiral trajectories sample k-space off the grid, so the Fourier
transform becomes a NUFFT. In lesson 7 you meet trajectories, the NUFFT,
density compensation and the point spread function. In lesson 8 you
reconstruct an undersampled golden-angle radial scan by non-Cartesian SENSE,
with coil sensitivities estimated from the radial data themselves. The tour
:doc:`/auto_examples/08-workflows/01-dynamic-golden-angle` extends it to a time
series. :doc:`/explanation/non-cartesian` defines the transform and its
accuracy.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lessons 2 to 6 every sample sat on the Cartesian grid, so the FFT and its inverse were all the Fourier transforms you needed. In this lesson you leave the grid. You sample the same brain slice along radial and spiral trajectories, reconstruct it by gridding, and see what each step of that reconstruction does to the image.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_01-trajectories-and-transforms_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/01-trajectories-and-transforms`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">7. Basics of non-Cartesian imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 7 you computed radial trajectories, the NUFFT, density compensation and the point spread function, and saw that omitting spokes produces streak artefacts. In this lesson you reconstruct an undersampled golden-angle radial scan with eight receive coils: you estimate the coil sensitivities from the radial data, then reconstruct by gridding, by iterative SENSE, and by SENSE with a total-variation penalty.">

.. only:: html

  .. image:: /auto_examples/04-non-cartesian/images/thumb/sphx_glr_02-radial-sense_thumb.png
    :alt:

  :doc:`/auto_examples/04-non-cartesian/02-radial-sense`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">8. Radial SENSE</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Model-based reconstruction
--------------------------

To map :math:`T_2` you usually reconstruct one image per echo and fit a decay
in every voxel. In lesson 9 you put the decay into the forward model instead,
and fit the maps directly to the multi-echo k-space, so that every echo
constrains the same unknowns. The tours
:doc:`/auto_examples/08-workflows/02-subspace-t1-mapping` and
:doc:`/auto_examples/08-workflows/03-maps-from-scanner-images` show the linear
subspace alternative and a fit to DICOM images. :doc:`/explanation/nonlinear`
compares the formulations.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 8 you reconstructed one image from undersampled radial data, and in lesson 4 you solved a model that is nonlinear in its unknowns, image times coil sensitivities, by Gauss-Newton. In this lesson you apply both to quantitative imaging. You estimate a T_2 map from an undersampled multi-echo spin-echo scan twice: first by reconstructing an image per echo and fitting the decay voxel by voxel, then by fitting the signal model directly to the k-space data, and you compare the accuracy of the two at the same undersampling.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_01-quantitative-models_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/01-quantitative-models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">9. Nonlinear inversion for T2 mapping</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Learned regularization
----------------------

A learned reconstruction keeps the encoding operator and the data consistency
of an iterative reconstruction, and puts a network where the regularizer was.
In lesson 10 you take a pretrained denoiser and use it inside BART's ADMM and
FISTA, with no training at all: plug-and-play. In lesson 11 you train a network
through BART's ADMM, end to end (MoDL). The learning
workflows among the :doc:`/gallery/tours` go further.
:doc:`/explanation/learned-reconstruction` explains where a network enters a
reconstruction.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 9 the prior knowledge was a signal model; in lesson 5 it was an explicit penalty, a wavelet or total-variation term with a weight. In this lesson the prior is a pretrained denoising network, used in place of the penalty inside the same ADMM and FISTA iterations.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_01-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/01-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">10. Plug-and-play reconstruction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 10 the proximal step of ADMM was a denoiser pretrained on photographs, never trained on MR images or on the residual aliasing of an undersampled scan. In this lesson you train the denoiser through the iteration: a fixed number of ADMM steps is unrolled into one network, which is fitted end to end against fully sampled images.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_02-modl-with-admm_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/02-modl-with-admm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">11. Training an unrolled network</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Tours: acquisition and corrections
----------------------------------

Corrections you apply to the data before the reconstruction, or to the image
after it, each on its own. Most of them simulate the artefact from a known
truth, so you can see how much the correction recovers; the EPI
susceptibility tour corrects measured data instead, and checks the result
against an independent field map.

- Remove readout oversampling, and compare Fermi and Hann apodization.
- Correct the Nyquist ghost of an EPI train, and regrid a ramp-sampled readout.
- Estimate the receive bias field of a head array with N4.
- Undo the distortion of gradient nonlinearity from the coil's
  spherical-harmonic coefficients.
- Deblur a spiral image off resonance, by multifrequency interpolation and by
  a time-segmented reconstruction.
- Track rigid head motion from three orthogonal navigator planes.
- Correct the susceptibility distortion of a 3 T EPI pair with reversed phase
  encoding, downloaded from OpenNeuro.
- Prewhiten correlated channel noise, and measure the SNR it buys.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you apply two corrections to Cartesian k-space before reconstruction. The readout is usually sampled at twice the density the matrix requires; removing that oversampling gives the image the prescribed matrix. And the truncation of k-space at the edge of the acquired matrix causes Gibbs ringing next to every sharp edge, which an apodization window reduces at the cost of resolution.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_01-readout-oversampling-and-apodization_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/01-readout-oversampling-and-apodization`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Readout oversampling and apodization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An EPI readout acquires k-space as a train of lines with alternating readout gradient polarity, and before those lines make a Cartesian k-space you correct two things. A timing error between the readout gradient and the ADC, and eddy currents, shift the echoes of the reversed lines against the forward ones; the odd/even phase this leaves gives you a Nyquist ghost, a copy of the object shifted by half the field of view along the phase-encoding direction. And if you sample on the ramps of the readout gradient, to shorten the echo spacing, your samples are uniform in time but not in k_x.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_02-epi-ghost-and-ramp-sampling_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/02-epi-ghost-and-ramp-sampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">EPI Nyquist ghost and ramp sampling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When you combine the images of a receive array by root sum of squares, what you get is the object weighted by the root sum of squares of the coil sensitivities. That weight is smooth, multiplicative and largest near the elements, so it shades the image: the same tissue looks brighter near the array than far from it. The shading biases segmentation, intensity-based registration and any quantitative comparison across the field of view.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_03-bias-field_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/03-bias-field`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Receive bias field</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spatial encoding assumes that every gradient field changes linearly with position. A real gradient coil departs from that more and more with the distance from isocentre, so each spin is encoded at a position displaced from its true one. The image is warped, by a few millimetres at the edge of a head-sized field of view and by centimetres at the edge of a body-sized one, and the voxel volume changes with the warp. The displacement is a property of the coil: the manufacturer states it as the coefficients of a spherical-harmonic expansion of each gradient field [#janke]_. To correct it you evaluate that expansion at every voxel and resample the image at the positions where the voxels were actually encoded.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_04-gradient-nonlinearity_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/04-gradient-nonlinearity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient nonlinearity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you correct the blurring that B_0 off-resonance causes in a spiral image. A spiral readout reaches each k-space radius at its own time, so a spin off resonance by f picks up a phase 2\pi f\, t(k) that grows from the centre of k-space outwards. In a Cartesian scan the same phase is linear along the readout and only shifts the spin; on a spiral it is radial, and it spreads the spin into a ring. With a readout of tens of milliseconds, the field near the sinuses and the ear canals is enough to smear the orbitofrontal and temporal cortex over several voxels.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_05-spiral-deblurring_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/05-spiral-deblurring`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Off-resonance correction of spiral imaging</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you measure how a head moves during a scan, from navigators. If the head moves, the anatomy is in a different place for the readouts that encode it, and the image is blurred and ghosted. Prospective correction moves the imaging field of view with the head before each readout, and for that it needs the rigid pose of the head, six degrees of freedom, measured again and again during the scan. A navigator is a short, low-resolution acquisition interleaved with the imaging readouts; registering it against the first navigator of the scan tells you how the head has moved since [#ehman]_.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_06-navigator-motion_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/06-navigator-motion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rigid head motion from navigators</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you correct the susceptibility distortion of a measured EPI pair. In an echo-planar image, the phase-encoding direction is sampled once per echo spacing rather than once per dwell time, so its bandwidth per pixel is only a few tens of hertz. A spin off resonance by \Delta f is displaced along phase encoding by \Delta f divided by that bandwidth, which near the frontal sinus and the petrous bone comes to several millimetres at 3 T. The orbitofrontal cortex and the temporal poles are compressed or stretched, and signal piles up where neighbouring voxels are displaced onto the same place.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_07-epi-susceptibility-distortion_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/07-epi-susceptibility-distortion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Susceptibility distortion in EPI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you measure how much signal-to-noise ratio (SNR) a SENSE reconstruction loses when the noise of the receive channels is correlated, and how much of it you get back by prewhitening with a noise-only acquisition.">

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

The Course's encodings applied to a time series and to quantitative mapping.

- Reconstruct a dynamic contrast-enhanced series from one continuous
  golden-angle radial scan, with total variation along time.
- Map :math:`T_1` from one continuous inversion-recovery scan, by
  reconstructing the coefficients of a subspace of the signal curves.
- Fit a :math:`T_2` decay to magnitude images read from DICOM, as the scanner
  exports them, and write the map back.

:doc:`/explanation/non-cartesian` and :doc:`/explanation/nonlinear` give the
background.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you reconstruct a dynamic contrast-enhanced series from one continuous golden-angle radial acquisition, which you cut into frames of thirteen spokes each after the scan. On its own, each frame is undersampled about fifteenfold, and its gridding reconstruction is dominated by streaks. Consecutive frames, however, show the same anatomy and differ by a slow change of contrast, so you reconstruct all frames jointly with a penalty on the temporal variation of the series. You compare frame-by-frame gridding with that joint reconstruction on the images, on an x-t profile, and on the time-intensity curve a perfusion analysis would read.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_01-dynamic-golden-angle_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/01-dynamic-golden-angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dynamic golden-angle radial MRI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you estimate a T_1 map from a single continuous inversion-recovery acquisition in which each of four hundred time points is encoded by one radial spoke, so each frame on its own is undersampled about two-hundredfold. The recovery curves of all plausible T_1 values span a subspace of low dimension; reconstructing the few coefficients of that subspace instead of the individual frames reduces the number of unknowns by two orders of magnitude.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_02-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/02-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you estimate a T_2 map from the magnitude images a scanner exports, without the raw data: you read a multi-echo spin-echo series from DICOM, fit the decay voxel by voxel, and write the map back as a DICOM series of the same study and as a NIfTI volume. The geometry and the echo times pass from the scanner&#x27;s files through the fit to the output unchanged, so the map overlays the images it was computed from.">

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

What comes after lessons 10 and 11, when you train and use a learned
regularizer on real problems.

- Networks for complex multi-contrast volumes, applied patch by patch.
- Training an unrolled network in stages, within the memory of one iteration.
- Self-supervised training from undersampled data alone.
- Plug-and-play with a noise level that anneals over the iterations.
- Voxel-wise uncertainty, calibrated.

:doc:`/explanation/learned-reconstruction` explains where a network enters a
reconstruction.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lessons 10 and 11 your network denoised one complex 2D slice. The data you most need a learned reconstruction for are larger: a 3D volume of several contrasts, the subspace coefficients of MR fingerprinting, the frames of a cine. In this tour you train a 3D denoiser on patches of a complex, multi-contrast brain volume, apply it to the whole volume of another subject patch by patch, the way it would run on a scanner GPU too small for the volume, and check that the patch boundaries leave no seams.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_01-networks-for-complex-volumes_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/01-networks-for-complex-volumes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Networks for complex volumes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 11 you trained an unrolled network end to end, which keeps the activations of every iteration in memory for the backward pass. For a 2D slice that memory is affordable; for a 3D volume, or a series of them, it exceeds a single GPU. In this tour you train an unrolled network for fourfold undersampled, eight-channel Cartesian brain data in three stages whose memory is bounded by one iteration, and you check that it removes the residual aliasing and the g-factor noise that CG-SENSE leaves at this acceleration.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_02-staged-training_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/02-staged-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Staged training of an unrolled network</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The networks of lesson 11 and of 02-staged-training learn from fully sampled reference images. Dynamic and high-dimensional acquisitions -- a cine, a functional run, a fingerprinting series -- are rarely acquired fully sampled: undersampling is what makes them fit in a breath-hold or a scan time, so you have no reference to train against. In this tour you train the unrolled network of 02-staged-training from undersampled k-space alone and measure how much of the supervised network&#x27;s image quality it keeps.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_03-self-supervised-training_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/03-self-supervised-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Training without a reference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 10 you plugged a denoiser someone else had trained, on photographs, into ADMM at one noise level you chose. In this tour you train the denoiser yourself, on MR images alone and without any encoding, and use it as the regularizer of an ADMM reconstruction whose denoising strength decreases over the iterations. You then apply the same network, unchanged, at an acceleration where CG-SENSE breaks down.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_04-annealed-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/04-annealed-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Annealed plug-and-play</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A learned reconstruction returns an image without saying where it may be wrong. Where the undersampling leaves the image underdetermined, the network fills in what its training data suggest, and a hallucinated structure looks like real anatomy. In this tour you attach a voxel-wise error bar to the unrolled reconstruction of 02-staged-training, calibrate it so that it contains the true error in a stated fraction of voxels, and look at where in the head the reconstruction is least certain.">

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
