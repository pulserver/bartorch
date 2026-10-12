

.. _sphx_glr_auto_examples_01-basics:

Basics
------

You start from the data a scanner gives you, multichannel k-space, and end
with an image. Lesson 1 shows how BART's arrays look as PyTorch tensors and how
you call a BART command on them. Lesson 2 reconstructs an undersampled
Cartesian scan through the three steps the rest of the Course builds on:
channel compression, coil sensitivities from ESPIRiT, and a regularized SENSE
reconstruction. :doc:`/explanation/encoding` derives the signal model, and
:doc:`/explanation/data-layout` lists the array layouts.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="If you have used BART before, you know its arrays as CFL files on disk and its commands as programs you call from a shell. bartorch runs the same BART inside your Python process: every command is a Python function that takes and returns torch.Tensor objects, with no files and no subprocess in between. In this lesson you see what your data look like on that side: how a multichannel image and its k-space are laid out as tensors, which Fourier transform takes one to the other, and how a BART command line maps onto a function call.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_01-tensors-and-commands_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/01-tensors-and-commands`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">1. Your data as tensors</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In lesson 1 you saw how BART&#x27;s arrays look as tensors and how its FFT takes a fully sampled k-space to an image. A real scan is rarely fully sampled: you skip phase-encoding lines to shorten it, and the image folds onto itself. In this lesson you reconstruct such a scan, one step at a time, and after each step you look at what it did to the image.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_02-from-kspace-to-image_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/02-from-kspace-to-image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2. From k-space to an image</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/01-basics/01-tensors-and-commands
   /auto_examples/01-basics/02-from-kspace-to-image

