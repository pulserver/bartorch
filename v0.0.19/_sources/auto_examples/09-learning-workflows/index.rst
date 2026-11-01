

.. _sphx_glr_auto_examples_09-learning-workflows:

Tours: learning workflows
-------------------------

What comes after lessons 10 and 11, when you train and use a learned
regularizer on real problems.

- Networks for complex multi-contrast volumes, applied patch by patch.
- Training an unrolled network in stages, within the memory of one iteration.
- Self-supervised training from undersampled data alone.
- Plug-and-play with a noise level that anneals over the iterations.
- Voxel-wise uncertainty, calibrated.

:doc:`/explanation/learned-reconstruction` explains where a network enters a
reconstruction.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lessons 10 and 11 your network denoised one complex 2D slice. The data you most need a learned reconstruction for are larger: a 3D volume of several contrasts, the subspace coefficients of MR fingerprinting, the frames of a cine. In this tour you train a 3D denoiser on patches of a complex, multi-contrast brain volume, apply it to the whole volume of another subject patch by patch, the way it would run on a scanner GPU too small for the volume, and check that the patch boundaries leave no seams.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_01-networks-for-complex-volumes_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/01-networks-for-complex-volumes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Networks for complex volumes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 11 you trained an unrolled network end to end, which keeps the activations of every iteration in memory for the backward pass. For a 2D slice that memory is affordable; for a 3D volume, or a series of them, it exceeds a single GPU. In this tour you train an unrolled network for fourfold undersampled, eight-channel Cartesian brain data in three stages whose memory is bounded by one iteration, and you check that it removes the residual aliasing and the g-factor noise that CG-SENSE leaves at this acceleration.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_02-staged-training_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/02-staged-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Staged training of an unrolled network</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The networks of lesson 11 and of 02-staged-training learn from fully sampled reference images. Dynamic and high-dimensional acquisitions -- a cine, a functional run, a fingerprinting series -- are rarely acquired fully sampled: undersampling is what makes them fit in a breath-hold or a scan time, so you have no reference to train against. In this tour you train the unrolled network of 02-staged-training from undersampled k-space alone and measure how much of the supervised network&#x27;s image quality it keeps.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_03-self-supervised-training_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/03-self-supervised-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Training without a reference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 10 you plugged a denoiser someone else had trained, on photographs, into ADMM at one noise level you chose. In this tour you train the denoiser yourself, on MR images alone and without any encoding, and use it as the regularizer of an ADMM reconstruction whose denoising strength decreases over the iterations. You then apply the same network, unchanged, at an acceleration where CG-SENSE breaks down.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_04-annealed-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/04-annealed-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Annealed plug-and-play</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A learned reconstruction returns an image without saying where it may be wrong. Where the undersampling leaves the image underdetermined, the network fills in what its training data suggest, and a hallucinated structure looks like real anatomy. In this tour you attach a voxel-wise error bar to the unrolled reconstruction of 02-staged-training, calibrate it so that it contains the true error in a stated fraction of voxels, and look at where in the head the reconstruction is least certain.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_05-uncertainty_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/05-uncertainty`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Uncertainty estimation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/09-learning-workflows/01-networks-for-complex-volumes
   /auto_examples/09-learning-workflows/02-staged-training
   /auto_examples/09-learning-workflows/03-self-supervised-training
   /auto_examples/09-learning-workflows/04-annealed-plug-and-play
   /auto_examples/09-learning-workflows/05-uncertainty

