

.. _sphx_glr_auto_examples_05-model-based:

Model-based reconstruction
--------------------------

Reconstructions that put a signal model into the encoding.

The first lesson constrains a series of four hundred radial frames to a
low-dimensional subspace of inversion-recovery curves and fits :math:`T_1`
from the coefficient maps. The second estimates :math:`T_2` maps from
multi-echo k-space directly, through a nonlinear forward operator, and
compares the result with fitting reconstructed echo images.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An inversion-recovery FLASH acquisition of four hundred frames, one spoke each, reconstructed into the coefficients of a signal subspace and fitted for T_1.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_01-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/01-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A multi-echo spin-echo acquisition fitted for T_2 in two ways: by reconstructing the echo images and fitting them afterwards, and by putting the signal model inside the forward operator and fitting the k-space directly.">

.. only:: html

  .. image:: /auto_examples/05-model-based/images/thumb/sphx_glr_02-quantitative-models_thumb.png
    :alt:

  :doc:`/auto_examples/05-model-based/02-quantitative-models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter maps straight from k-space</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/05-model-based/01-subspace-t1-mapping
   /auto_examples/05-model-based/02-quantitative-models

