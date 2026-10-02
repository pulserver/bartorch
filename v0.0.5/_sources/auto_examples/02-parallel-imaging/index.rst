

.. _sphx_glr_auto_examples_02-parallel-imaging:

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


.. toctree::
   :hidden:

   /auto_examples/02-parallel-imaging/01-coil-calibration
   /auto_examples/02-parallel-imaging/02-nonlinear-inversion
   /auto_examples/02-parallel-imaging/03-noise-prewhitening

