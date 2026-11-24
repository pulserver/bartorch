

.. _sphx_glr_auto_examples_05-model-based:

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


.. toctree::
   :hidden:

   /auto_examples/05-model-based/01-quantitative-models

