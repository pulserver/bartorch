

.. _sphx_glr_auto_examples_03-regularization:

Regularization
--------------

Beyond the acceleration factor the coil geometry supports, the SENSE problem
is ill-conditioned: a least-squares solution amplifies noise, and the aliasing
of the undersampling is not fully resolved.  A regularized reconstruction adds
prior knowledge of the image as a penalty.  This section compares Tikhonov,
wavelet-sparsity (compressed sensing) and total-variation penalties on one
undersampled acquisition and shows how the regularization weight is chosen.  It
then writes the same reconstruction as an explicit encoding operator and an
iterative solver, the form required by an encoding for which BART has no
application.  :doc:`/explanation/inverse-problems` introduces the formulations
and the algorithms.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson compares three regularization terms on the same undersampled, noisy SENSE acquisition, shows how the choice of the regularization weight trades residual noise and aliasing against loss of detail, and combines two terms in one reconstruction.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_01-regularized-reconstruction_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/01-regularized-reconstruction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Regularized reconstruction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson rebuilds the reconstruction of the previous lessons from its parts -- the encoding operator, the regularization term and the iterative algorithm -- instead of calling a BART application, and shows that the result is identical.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_02-operators-and-solvers_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/02-operators-and-solvers`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Operators and solvers</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/03-regularization/01-regularized-reconstruction
   /auto_examples/03-regularization/02-operators-and-solvers

