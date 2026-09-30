

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

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train a 3D convolutional denoiser on patches of a complex, multi-contrast brain volume, apply it to a whole volume of another subject patch by patch, as it would run on a scanner GPU too small for the volume, and check that the patch boundaries leave no visible seams.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_03-networks-for-complex-volumes_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/03-networks-for-complex-volumes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Networks for complex volumes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train an unrolled reconstruction network for fourfold undersampled, eight-channel Cartesian brain data within the memory of one iteration, and show that it removes the residual aliasing and the g-factor noise that CG-SENSE leaves at this acceleration.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_04-staged-training_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/04-staged-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Staged training of an unrolled network</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train the unrolled network of 04-staged-training from undersampled k-space alone, with no fully sampled reference, and measure how much of the supervised network&#x27;s image quality it retains.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_05-self-supervised-training_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/05-self-supervised-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Training without a reference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train one denoiser on images alone, without any encoding, and use it as the regularizer of an ADMM reconstruction at any undersampling, with the denoising strength decreasing over the iterations; show that it holds up at an acceleration where CG-SENSE breaks down.">

.. only:: html

  .. image:: /auto_examples/06-learning/images/thumb/sphx_glr_06-annealed-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/06-learning/06-annealed-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Annealed plug-and-play</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Attach a voxel-wise error bar to a learned reconstruction of undersampled data, calibrated so that it contains the true error in a stated fraction of voxels, and see where in the head the reconstruction is least certain.">

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

