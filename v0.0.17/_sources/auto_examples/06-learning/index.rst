

.. _sphx_glr_auto_examples_06-learning:

Learned regularization
----------------------

A learned reconstruction keeps the encoding operator and the data consistency
of an iterative reconstruction, and puts a network where the regularizer was.
In lesson 10 you take a pretrained denoiser and use it inside BART's ADMM and
FISTA, with no training at all: plug-and-play. In lesson 11 you train a network
through BART's ADMM, end to end (MoDL). The learning
workflows among the :doc:`/gallery/tours` go further.
:doc:`/explanation/learned-reconstruction` explains where a network enters a
reconstruction.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 9 the prior knowledge was a signal model; in lesson 5 it was an explicit penalty, a wavelet or total-variation term with a weight. In this lesson the prior is a pretrained denoising network, used in place of the penalty inside the same ADMM and FISTA iterations.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_01-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/01-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">10. Plug-and-play reconstruction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 10 the proximal step of ADMM was a denoiser pretrained on photographs, never trained on MR images or on the residual aliasing of an undersampled scan. In this lesson you train the denoiser through the iteration: a fixed number of ADMM steps is unrolled into one network, which is fitted end to end against fully sampled images.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_02-modl-with-admm_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/02-modl-with-admm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">11. Training an unrolled network</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/06-learning/01-plug-and-play
   /auto_examples/06-learning/02-modl-with-admm

