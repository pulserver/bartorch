

.. _sphx_glr_auto_examples_02-parallel-imaging:

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


.. toctree::
   :hidden:

   /auto_examples/02-parallel-imaging/01-coil-calibration
   /auto_examples/02-parallel-imaging/02-nonlinear-inversion

