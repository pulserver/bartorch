

.. _sphx_glr_auto_examples_05-model-based:

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


.. toctree::
   :hidden:

   /auto_examples/05-model-based/01-quantitative-models

