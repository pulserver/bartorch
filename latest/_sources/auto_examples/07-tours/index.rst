

.. _sphx_glr_auto_examples_07-tours:

Tours
-----

Standalone examples of the corrections and the rigid-motion tracking in
:mod:`bartorch.tools`, each simulated from a known ground truth so that the
correction can be measured against it. They do not depend on one another or
on the course.

The section additionally requires::

    pip install SimpleITK


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two operations on Cartesian k-space before it is reconstructed: removing the readout oversampling, and weighting the measurement by an apodization window to suppress the Gibbs ringing of its truncation.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_01-readout-oversampling-and-apodization_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/01-readout-oversampling-and-apodization`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Readout oversampling and apodization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two corrections an echo-planar readout needs before its lines form a Cartesian k-space: the odd/even phase that produces the Nyquist ghost, estimated from a three-line navigator, and the resampling of samples taken on the gradient ramps onto the uniform grid.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_02-epi-ghost-and-ramp-sampling_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/02-epi-ghost-and-ramp-sampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">EPI Nyquist ghost and ramp sampling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The intensity shading that a surface receive array leaves in a root-sum-of-squares image, estimated and removed by N4 bias field correction.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_03-bias-field_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/03-bias-field`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Receive bias field</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The geometric distortion a gradient coil&#x27;s nonlinearity produces over a large field of view, simulated from a spherical-harmonic description of the coil and corrected with bartorch.tools.Gradunwarp.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_04-gradient-nonlinearity_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/04-gradient-nonlinearity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient nonlinearity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The blurring that off-resonance produces in a spiral image, simulated for a known field map and removed with bartorch.tools.deblur.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_05-spiral-deblurring_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/05-spiral-deblurring`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Off-resonance blurring in spirals</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six-degree-of-freedom rigid head motion measured from three orthogonal radial navigator planes, and carried across a scan by a Kalman filter, with the rigid-motion functions of bartorch.tools.">

.. only:: html

  .. image:: /auto_examples/07-tours/images/thumb/sphx_glr_06-navigator-motion_thumb.png
    :alt:

  :doc:`/auto_examples/07-tours/06-navigator-motion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rigid head motion from navigators</div>
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

