

.. _sphx_glr_auto_examples_01-basics:

Basics
------

The course starts from the data a scanner delivers, multichannel k-space, and
reconstructs an image from it.  The first lesson relates PyTorch tensors to
BART's arrays and commands, simulates an acquisition with BART's analytical
phantom and verifies the centring and scaling of BART's FFT against NumPy.  The
second reconstructs an undersampled Cartesian acquisition through the three
steps every later section builds on: coil compression, calibration of the coil
sensitivity maps by ESPIRiT, and a regularized SENSE reconstruction.  The
signal model is derived in :doc:`/explanation/encoding`, and the array
conventions are stated in :doc:`/explanation/data-layout`.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This first lesson establishes how MR data from BART appear in Python: how a multichannel image and its k-space are laid out as tensors, which Fourier convention BART uses to go from k-space to the image, and how a BART command is called. Every later lesson relies on these conventions. A reconstruction that is off by a factor of \sqrt{N}, or by a half-voxel shift from a misplaced k-space centre, is a consequence of misreading one of them.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_01-tensors-and-commands_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/01-tensors-and-commands`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tensors and commands</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This lesson reconstructs an undersampled Cartesian brain acquisition from its multichannel k-space to a coil-combined image, and shows what each step of a parallel-imaging and compressed-sensing pipeline contributes. Scan time in Cartesian MRI is proportional to the number of phase-encoding lines; skipping lines shortens the scan by the acceleration factor R, but violates the Nyquist criterion and folds the image onto itself. Recovering an unaliased image from such data is what the receive coil array, and prior knowledge of the image, are used for.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_02-from-kspace-to-image_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/02-from-kspace-to-image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">From k-space to image</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/01-basics/01-tensors-and-commands
   /auto_examples/01-basics/02-from-kspace-to-image

