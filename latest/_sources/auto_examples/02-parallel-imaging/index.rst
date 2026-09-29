

.. _sphx_glr_auto_examples_02-parallel-imaging:

Parallel imaging
----------------

The coil sensitivities a SENSE reconstruction depends on, and the channel
noise it propagates.

The first lesson compares three sensitivity estimates from the same
acquisition, :func:`bartorch.tools.caldir`, ESPIRiT and nonlinear inversion.
The second estimates the image and the sensitivities jointly where the
calibration region is too small for a separate calibration. The third measures
the effect of correlated channel noise on the reconstruction and removes it by
prewhitening.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three estimates of the receive sensitivities of a coil array from the same undersampled Cartesian acquisition, and the SENSE reconstructions they lead to.">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_01-coil-calibration_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/01-coil-calibration`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Coil sensitivity calibration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Estimating the image and the coil sensitivities together, from undersampled data whose fully sampled central region is too small for a separate calibration.">

.. only:: html

  .. image:: /auto_examples/02-parallel-imaging/images/thumb/sphx_glr_02-nonlinear-inversion_thumb.png
    :alt:

  :doc:`/auto_examples/02-parallel-imaging/02-nonlinear-inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonlinear inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The effect of channel-noise correlation on a SENSE reconstruction, and its removal by prewhitening with a noise measurement.">

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

