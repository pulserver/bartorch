

.. _sphx_glr_auto_examples_08-workflows:

Tours: dynamic and quantitative workflows
-----------------------------------------

Applications of the course's encodings to a time series and to quantitative
mapping, each shown on its own.  The first reconstructs a dynamic
contrast-enhanced series from one continuous golden-angle radial acquisition
with a total-variation penalty along time.  The second estimates a
:math:`T_1` map from a single continuous inversion-recovery acquisition by
reconstructing the coefficients of a linear subspace of the signal curves.  The
third fits a :math:`T_2` decay to magnitude images read from DICOM, as a
scanner exports them, and writes the map back.
:doc:`/explanation/non-cartesian` and :doc:`/explanation/nonlinear` give the
background.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a dynamic contrast-enhanced series from one continuous golden-angle radial acquisition, cut into frames of thirteen spokes each. Each frame on its own is undersampled fifteenfold and cannot be reconstructed; the series can, because consecutive frames are strongly correlated, and a total-variation penalty along the time axis states that correlation. The example compares frame-by-frame gridding with this joint reconstruction on the images and on the time-intensity curve a perfusion analysis would use.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_01-dynamic-golden-angle_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/01-dynamic-golden-angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dynamic golden-angle radial MRI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example estimates a T_1 map from a single continuous inversion-recovery acquisition in which each of four hundred time points is encoded by one radial spoke. The aim is to show how a signal model turns a hopelessly undersampled time series into a well-posed reconstruction: the recovery curves of all plausible T_1 values span a subspace of low dimension, and reconstructing the few coefficients of that subspace instead of the individual frames reduces the number of unknowns by two orders of magnitude.">

.. only:: html

  .. image:: /auto_examples/08-workflows/images/thumb/sphx_glr_02-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/08-workflows/02-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example estimates a T_2 map from the magnitude images a scanner exports, without access to the raw data: a multi-echo spin-echo series is read from DICOM, the decay is fitted voxel by voxel, and the map is written back as a DICOM series of the same study and as a NIfTI volume. The aim is to show the geometry and the acquisition timings passing from the scanner&#x27;s files to the fit and on to the output unchanged, so that the map overlays the images it was computed from.">

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

