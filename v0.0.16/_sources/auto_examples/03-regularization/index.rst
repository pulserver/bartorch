

.. _sphx_glr_auto_examples_03-regularization:

Regularization
--------------

Past the acceleration your coils can support, SENSE amplifies noise and leaves
aliasing behind. A regularizer adds what you know about the image. In lesson 5
you compare Tikhonov, wavelet and total-variation penalties on one scan and
choose their weight. In lesson 6 you write the same reconstruction yourself, as
an encoding operator and a solver, which is how you reconstruct anything BART
has no command for. :doc:`/explanation/regularizers` and :doc:`/explanation/solvers` explain
the penalties and the iterations.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 2 you reconstructed an undersampled scan with a wavelet penalty at a weight of 0.004, both fixed without justification. In this lesson you choose them. You reconstruct one noisy, undersampled SENSE acquisition with three different penalties, sweep the weight of each, and compare under- and over-regularized images. Then you combine two penalties in one reconstruction.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_01-regularized-reconstruction_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/01-regularized-reconstruction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">5. Choosing a regularizer</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 5 you handed pics the data, the maps and a penalty, and it did the rest. In this lesson you assemble the same reconstruction from its parts. You build the SENSE encoding operator, check its adjoint, run the solver on it, and obtain exactly what pics returns, bit for bit. Then you extend the operator with one written in Python, and differentiate through it.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_02-operators-and-solvers_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/02-operators-and-solvers`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">6. Custom encoding and solver</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/03-regularization/01-regularized-reconstruction
   /auto_examples/03-regularization/02-operators-and-solvers

