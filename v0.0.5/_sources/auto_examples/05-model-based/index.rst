

.. _sphx_glr_auto_examples_05-model-based:

Model-based reconstruction
--------------------------

Quantitative MRI estimates tissue parameters such as :math:`T_1` and
:math:`T_2` from a series of images acquired at different contrasts.
Reconstructing each contrast separately and fitting a signal model afterwards
ignores the relation between the contrasts that the signal model states.  A
model-based reconstruction places that relation in the forward operator, so
that every contrast constrains the same unknowns.  This section treats the two
standard formulations: a linear subspace model, in which inversion-recovery
signal curves are represented by a few temporal basis functions and
:math:`T_1` is fitted to the coefficient maps, and a nonlinear signal model,
through which :math:`T_2` maps are estimated directly from multi-echo k-space.
:doc:`/explanation/nonlinear` compares the two with reconstruction followed by
a voxel-wise fit.  The last lesson fits a signal model to magnitude
images read from DICOM, as a scanner exports them, and writes the map back.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson estimates a T_1 map from a single continuous inversion-recovery acquisition in which each of four hundred time points is encoded by one radial spoke. The aim is to show how a signal model turns a hopelessly undersampled time series into a well-posed reconstruction: the recovery curves of all plausible T_1 values span a subspace of low dimension, and reconstructing the few coefficients of that subspace instead of the individual frames reduces the number of unknowns by two orders of magnitude.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_01-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/01-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson estimates a T_2 map from an undersampled multi-echo spin-echo acquisition in two ways, and compares them: reconstructing an image per echo and fitting the decay voxel by voxel afterwards, and fitting the signal model directly to the k-space data. The aim is to show why the second, model-based reconstruction, tolerates undersampling that ruins the first.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_02-quantitative-models_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/02-quantitative-models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter maps straight from k-space</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson estimates a T_2 map from the magnitude images a scanner exports, without access to the raw data: a multi-echo spin-echo series is read from DICOM, the decay is fitted voxel by voxel, and the map is written back as a DICOM series of the same study and as a NIfTI volume. The aim is to show the geometry and the acquisition timings passing from the scanner&#x27;s files to the fit and on to the output unchanged, so that the map overlays the images it was computed from.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_03-maps-from-scanner-images_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/03-maps-from-scanner-images`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter maps from scanner images</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/05-model-based/01-subspace-t1-mapping
   /auto_examples/05-model-based/02-quantitative-models
   /auto_examples/05-model-based/03-maps-from-scanner-images

