

.. _sphx_glr_auto_examples_06-learning:

Learned regularization
----------------------

A learned reconstruction replaces the hand-specified regularization term by a
neural network and keeps the encoding operator and the data consistency of the
iterative reconstruction.  This section covers the two ways a network enters
BART's iterations: plug-and-play reconstruction, in which a pretrained denoiser
takes the place of the proximal operator of ADMM and FISTA without any
training, and an unrolled network trained through BART's ADMM (MoDL).
Networks for complex volumes, staged and self-supervised training, annealed
plug-and-play and uncertainty are the tours of
:doc:`/auto_examples/09-learning-workflows/index`.
:doc:`/explanation/learned-reconstruction` describes where a network enters a
reconstruction.

The section additionally requires::

    pip install lightning torchio monai deepinv

The first lesson downloads the DRUNet weights ``deepinv`` distributes.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson regularizes an undersampled, noisy Cartesian SENSE reconstruction with a pretrained image denoiser in place of a specified penalty, and compares the result with total-variation regularization of the same data. The aim is to show how a denoiser enters a proximal iteration, what it improves on a hand-crafted penalty, and how its noise level plays the role of the regularization weight.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_01-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/01-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plug-and-play denoisers</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson trains an unrolled reconstruction network for undersampled Cartesian SENSE: a small convolutional denoiser placed in the proximal step of BART&#x27;s alternating-direction iteration, with the whole iteration trained end to end against fully sampled images. The aim is to show how a learned regularizer is combined with the physical encoding model -- coil sensitivities, Fourier transform and sampling pattern -- so that the network only has to remove what the data leave undetermined, and how such a network is trained with standard tools.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_02-modl-with-admm_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/02-modl-with-admm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MoDL, on BART's ADMM</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/06-learning/01-plug-and-play
   /auto_examples/06-learning/02-modl-with-admm

