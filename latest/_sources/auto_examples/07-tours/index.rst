

.. _sphx_glr_auto_examples_07-tours:

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

These tours also need the following packages, and the susceptibility tour
downloads about 2 MB of data into ``~/.cache/bartorch-examples``::

    pip install SimpleITK PyHySCO


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


.. toctree::
   :hidden:

   /auto_examples/07-tours/01-readout-oversampling-and-apodization
   /auto_examples/07-tours/02-epi-ghost-and-ramp-sampling
   /auto_examples/07-tours/03-bias-field
   /auto_examples/07-tours/04-gradient-nonlinearity
   /auto_examples/07-tours/05-spiral-deblurring
   /auto_examples/07-tours/06-navigator-motion
   /auto_examples/07-tours/07-epi-susceptibility-distortion
   /auto_examples/07-tours/08-noise-prewhitening

