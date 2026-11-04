

.. _sphx_glr_auto_examples_08-workflows:

Tours: dynamic and quantitative workflows
-----------------------------------------

The Course's encodings applied to a time series and to quantitative mapping.

- Reconstruct a dynamic contrast-enhanced series from one continuous
  golden-angle radial scan, with total variation along time.
- Map :math:`T_1` from one continuous inversion-recovery scan, by
  reconstructing the coefficients of a subspace of the signal curves.
- Fit a :math:`T_2` decay to magnitude images read from DICOM, as the scanner
  exports them, and write the map back.

:doc:`/explanation/non-cartesian` and :doc:`/explanation/nonlinear` give the
background.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you reconstruct a dynamic contrast-enhanced series from one continuous golden-angle radial acquisition, which you cut into frames of thirteen spokes each after the scan. On its own, each frame is undersampled about fifteenfold, and its gridding reconstruction is dominated by streaks. Consecutive frames, however, show the same anatomy and differ by a slow change of contrast, so you reconstruct all frames jointly with a penalty on the temporal variation of the series. You compare frame-by-frame gridding with that joint reconstruction on the images, on an x-t profile, and on the time-intensity curve a perfusion analysis would read.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_01-dynamic-golden-angle_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/01-dynamic-golden-angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dynamic golden-angle radial MRI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you estimate a T_1 map from a single continuous inversion-recovery acquisition in which each of four hundred time points is encoded by one radial spoke, so each frame on its own is undersampled about two-hundredfold. The recovery curves of all plausible T_1 values span a subspace of low dimension; reconstructing the few coefficients of that subspace instead of the individual frames reduces the number of unknowns by two orders of magnitude.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_02-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/02-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tour you estimate a T_2 map from the magnitude images a scanner exports, without the raw data: you read a multi-echo spin-echo series from DICOM, fit the decay voxel by voxel, and write the map back as a DICOM series of the same study and as a NIfTI volume. The geometry and the echo times pass from the scanner&#x27;s files through the fit to the output unchanged, so the map overlays the images it was computed from.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_03-maps-from-scanner-images_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/03-maps-from-scanner-images`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter maps from scanner images</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/08-workflows/01-dynamic-golden-angle
   /auto_examples/08-workflows/02-subspace-t1-mapping
   /auto_examples/08-workflows/03-maps-from-scanner-images

