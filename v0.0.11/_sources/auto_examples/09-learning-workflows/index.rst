

.. _sphx_glr_auto_examples_09-learning-workflows:

Tours: learning workflows
-------------------------

Training strategies and uses of a learned regularizer that follow from the two
lessons of :doc:`/auto_examples/06-learning/index`: networks for complex
multi-contrast volumes applied patch by patch, staged training of an unrolled
network within the memory of one iteration, self-supervised training from
undersampled data alone, plug-and-play with an annealed noise level, and
calibrated voxel-wise uncertainty.
:doc:`/explanation/learned-reconstruction` describes where a network enters a
reconstruction.

The section additionally requires::

    pip install lightning torchio monai deepinv


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train a 3D convolutional denoiser on patches of a complex, multi-contrast brain volume, apply it to a whole volume of another subject patch by patch, as it would run on a scanner GPU too small for the volume, and check that the patch boundaries leave no visible seams.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_01-networks-for-complex-volumes_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/01-networks-for-complex-volumes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Networks for complex volumes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train an unrolled reconstruction network for fourfold undersampled, eight-channel Cartesian brain data within the memory of one iteration, and show that it removes the residual aliasing and the g-factor noise that CG-SENSE leaves at this acceleration.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_02-staged-training_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/02-staged-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Staged training of an unrolled network</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train the unrolled network of 02-staged-training from undersampled k-space alone, with no fully sampled reference, and measure how much of the supervised network&#x27;s image quality it retains.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_03-self-supervised-training_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/03-self-supervised-training`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Training without a reference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Train one denoiser on images alone, without any encoding, and use it as the regularizer of an ADMM reconstruction at any undersampling, with the denoising strength decreasing over the iterations; show that it holds up at an acceleration where CG-SENSE breaks down.">

.. only:: html

  .. image:: /auto_examples/09-learning-workflows/images/thumb/sphx_glr_04-annealed-plug-and-play_thumb.png
    :alt:

  :doc:`/auto_examples/09-learning-workflows/04-annealed-plug-and-play`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Annealed plug-and-play</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Aim. Attach a voxel-wise error bar to a learned reconstruction of undersampled data, calibrated so that it contains the true error in a stated fraction of voxels, and see where in the head the reconstruction is least certain.">

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

