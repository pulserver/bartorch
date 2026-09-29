

.. _sphx_glr_auto_examples_03-regularization:

Regularization
--------------

Regularized least squares: the terms of :mod:`bartorch.priors`, and the
operators and solvers a reconstruction is assembled from.

The first lesson applies Tikhonov, wavelet and total-variation terms to one
undersampled SENSE problem and selects their weights. The second writes the
same reconstruction as a :mod:`bartorch.linop` encoding operator and a
:mod:`bartorch.optim` solver, the route for an encoding BART has no application
for.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three regularization terms applied to the same undersampled SENSE problem, and the dependence of the reconstruction error on the regularization weight.">

.. only:: html

  .. image:: /auto_examples/03-regularization/images/thumb/sphx_glr_01-regularized-reconstruction_thumb.png
    :alt:

  :doc:`/auto_examples/03-regularization/01-regularized-reconstruction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Regularized reconstruction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same reconstruction written as an encoding operator and a solver rather than as a call to a BART application.">

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

