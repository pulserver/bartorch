

.. _sphx_glr_auto_examples_07-tours:

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

