

.. _sphx_glr_auto_examples_06-learning:

Learned regularization
----------------------

Reconstructions whose regularizer is a network rather than a specified term.

The first lesson uses a pretrained denoiser as the proximal step of BART's
ADMM and FISTA iterations, with no training. The second unrolls BART's ADMM
with a convolutional denoiser in its proximal step and trains the network
through the iteration against fully sampled images. The third builds networks
for complex multi-contrast volumes and applies them patch by patch. The fourth
and fifth train an unrolled network in stages, and without references. The
sixth anneals the noise level of a plug-and-play denoiser, and the seventh
attaches calibrated error bars to a learned reconstruction.

The section additionally requires::

    pip install lightning torchio monai deepinv

The first lesson downloads the DRUNet weights ``deepinv`` distributes.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A pretrained image denoiser used as the proximal step of BART&#x27;s iterations, in place of a specified regularization term, for an undersampled Cartesian SENSE acquisition.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_01-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/01-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plug-and-play denoisers</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An unrolled network for undersampled Cartesian SENSE: a convolutional denoiser in the proximal step of BART&#x27;s alternating-direction iteration, trained end to end against fully sampled images.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_02-modl-with-admm_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/02-modl-with-admm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MoDL, on BART's ADMM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A convolutional denoiser for complex, multi-contrast volumes, trained on patches and applied to a whole volume patch by patch.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_03-networks-for-complex-volumes_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/03-networks-for-complex-volumes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Networks for complex volumes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An unrolled proximal-gradient network with one denoiser shared by every iteration and told which iteration it is in, trained in three stages: the denoiser alone, the iterations one at a time, and the whole stack.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_04-staged-training_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/04-staged-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Staged training of an unrolled network</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The unrolled network of 04-staged-training, trained from undersampled k-space alone by holding out part of the acquired samples and scoring the reconstruction on them.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_05-self-supervised-training_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/05-self-supervised-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Training without a reference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A denoiser conditioned on the noise level, trained once on images and used in ADMM with a noise level that decreases over the iterations.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_06-annealed-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/06-annealed-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Annealed plug-and-play</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A voxel-wise error bar for a learned reconstruction, from the spread of randomized reconstructions, calibrated on references to a stated coverage.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_07-uncertainty_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/07-uncertainty`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Uncertainty estimation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/06-learning/01-plug-and-play
   /auto_examples/06-learning/02-modl-with-admm
   /auto_examples/06-learning/03-networks-for-complex-volumes
   /auto_examples/06-learning/04-staged-training
   /auto_examples/06-learning/05-self-supervised-training
   /auto_examples/06-learning/06-annealed-plug-and-play
   /auto_examples/06-learning/07-uncertainty

