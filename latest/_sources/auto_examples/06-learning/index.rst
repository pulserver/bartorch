

.. _sphx_glr_auto_examples_06-learning:

Learned regularization
----------------------

A learned reconstruction replaces the hand-specified regularization term by a
neural network and keeps the encoding operator and the data consistency of the
iterative reconstruction.  This section starts with plug-and-play
reconstruction, in which a pretrained denoiser takes the place of the proximal
operator of ADMM and FISTA without any training, and proceeds to an unrolled
network trained through BART's ADMM (MoDL); networks for complex
multi-contrast volumes, applied patch by patch; staged and self-supervised
training of an unrolled network; plug-and-play with an annealed noise level;
and calibrated voxel-wise uncertainty.
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

