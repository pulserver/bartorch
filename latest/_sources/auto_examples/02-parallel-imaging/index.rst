

.. _sphx_glr_auto_examples_02-parallel-imaging:

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


.. toctree::
   :hidden:

   /auto_examples/02-parallel-imaging/01-coil-calibration
   /auto_examples/02-parallel-imaging/02-nonlinear-inversion

